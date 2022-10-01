from typing import Optional, Dict

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class ImageLogger(pl.Callback):
    def __init__(self,
                 RGB_image: Optional[Dict] = {'image': [0, 1, 2]},
                 RGB_target: Optional[Dict] = {'image': [0, 0, 0]},
                 RGB_output: Optional[Dict] = {'image': [0, 0, 0]},
                 normalize: Optional[bool] = True,
                 num_images: Optional[int] = None,
                 log_probabilities: Optional[bool] = False,
                 on_train_epoch_end: Optional[bool] = True,
                 on_validation_epoch_end: Optional[bool] = True,
                 ) -> None:
        """
        Inputs:
        ------
        RGB_image: dict
            Dict[List[int]]: len(RGB) is the amount of images to show in RGB.
            The combination of bands to associate to RGB is given in the values of the dictionary.
            The key string is unimportant. default: {'image': [0,1,2]}

        RGB_target: dict
            Dict[List[int]]: len(RGB) is the amount of target images to show in RGB.
            The combination of bands to associate to RGB is given in the values of the dictionary.
            The key string is unimportant. default: {'image': [0,0,0]}
            To use in conjunction with log_probabilities=True.

        RGB_output: dict
            Dict[List[int]]: len(RGB) is the amount of output images to show in RGB.
            The combination of bands to associate to RGB is given in the values of the dictionary.
            The key string is unimportant. default: {'image': [0,0,0]}
            To use in conjunction with log_probabilities=True.

        log_probabilities: bool
            if True, RGB_target and RGB_output are used.
            if False, the classes are obtained and given unique colors (max 27 classes).

        normalize: bool
            if True, normalizes all the bands of all the images in the batch between 0.1 and 99.9 percentiles

        num_images: int
            number of images in the batch to log.

        on_train_epoch_end: bool

        on_validation_epoch_end: bool
        """
        super().__init__()
        self.RGB_image = RGB_image if RGB_image is not None else {}
        self.RGB_target = RGB_target if RGB_target is not None else {}
        self.RGB_output = RGB_output if RGB_output is not None else {}
        self.normalize = normalize
        self.num_images = num_images
        self.log_valid = on_validation_epoch_end
        self.log_train = on_train_epoch_end
        self.log_probabilities = log_probabilities
        self.cmap = self._get_colormap()

    def _normalize(self, image: torch.Tensor):
        """ Normalize image

        Normalize the image by mapping the 0.1 and 99.9 percentiles between 0 and 1.
        Each channel is normalized independently.

        Input:
        -----
        image [torch.Tensor]: of shape [N, C, H, W]

        Output:
        ------
        normalize image [torch.Tensor] of shape [N, C, H, W]
        """
        shape = image.shape
        min_im = image.view(shape[0], shape[1], -1).quantile(0.001, dim=2).repeat(*shape[2:], 1, 1).permute((2, 3, 0, 1))
        max_im = image.view(shape[0], shape[1], -1).quantile(0.999, dim=2).repeat(*shape[2:], 1, 1).permute((2, 3, 0, 1))

        return (image - min_im) / (max_im - min_im)

    def _log_batch_predictions(self, trainer, plModel, images, targets, probabilities, mode):
        ''' Log all images in a batch, targets and probabilities

        Log each images in the batch as RGB images, their targets and the predicted probabilities
        '''
        if self.normalize:
            images = self._normalize(images)

        # Loop though each images in batch
        N = min(self.num_images, images.shape[0]) if self.num_images is not None else images.shape[0]

        for i in range(N):
            image = images[i]
            rgb_img = []
            for rgb in self.RGB_image.values():
                rgb_img.append(image[rgb, ...])

            target_rgb = []
            pred_rgb = []
            # If log probabilities, we want to show the classes with the same colors as for the predictions
            if self.log_probabilities:
                for rgb in self.RGB_target.values():
                    target_rgb.append(targets[i][rgb])
                for rgb in self.RGB_output.values():
                    pred_rgb.append(probabilities[i][rgb])
            else:
                target = torch.tensor([self.cmap[x.item()] for x in targets[i].reshape(-1)], device=image.device)
                target = target.reshape(image.shape[-2], image.shape[-1], 3).permute(2, 0, 1)   # (3,H,W)
                target_rgb.append(target)

                # Convert proba to classes first
                cls = plModel.probs_to_classes(probabilities[[i]])
                if cls.shape[1] != 1:   # i.e. multilabel
                    cls = cls.argmax(dim=1)
                pred = torch.tensor([self.cmap[x.item()] for x in cls.reshape(-1)], device=image.device)
                pred = pred.reshape(image.shape[-2], image.shape[-1], 3).permute(2, 0, 1)
                pred_rgb.append(pred)

            if i == 0:
                data = torch.stack([*rgb_img, *target_rgb, *pred_rgb], dim=0)
            else:
                data = torch.cat([data, torch.stack([*rgb_img, *target_rgb, *pred_rgb], dim=0)], dim=0)

        grid = torchvision.utils.make_grid(tensor=data,
                                           nrow=len(rgb_img) + len(target_rgb) + len(pred_rgb))

        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(f"{mode}_images", grid, global_step=trainer.global_step)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, plModel: pl.LightningModule) -> None:
        if self.log_valid and plModel.log_images['valid'] is not None:
            images, targets, prob = plModel.log_images['valid']
            self._log_batch_predictions(trainer, plModel, images, targets, prob, 'valid')

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: pl.Trainer, plModel: pl.LightningModule) -> None:
        if self.log_train and plModel.log_images['train'] is not None:
            images, targets, prob = plModel.log_images['train']
            self._log_batch_predictions(trainer, plModel, images, targets, prob, 'train')

    def _get_colormap(self):
        cmap = {
            -1: [0, 0, 0],
            0: [255, 255, 255],
            1: [240, 163, 255],
            2: [0, 117, 220],
            3: [153, 63, 0],
            4: [76, 0, 92],
            5: [25, 25, 25],
            6: [0, 92, 49],
            7: [43, 206, 72],
            8: [255, 204, 153],
            9: [128, 128, 128],
            10: [148, 255, 181],
            11: [143, 124, 0],
            12: [157, 204, 0],
            13: [194, 0, 136],
            14: [0, 51, 128],
            15: [255, 164, 5],
            16: [255, 168, 187],
            17: [66, 102, 0],
            18: [255, 0, 16],
            19: [94, 241, 242],
            20: [0, 153, 143],
            21: [224, 255, 102],
            22: [116, 10, 255],
            23: [153, 0, 0],
            24: [255, 255, 128],
            25: [255, 225, 0],
            26: [255, 80, 5],
        }
        return cmap
