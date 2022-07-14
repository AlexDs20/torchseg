from typing import Optional, Dict

import torch
import torchvision
import pytorch_lightning as pl


class ImageLogger(pl.Callback):
    def __init__(self,
                 RGB: Optional[Dict]={'image': [0,1,2]},
                 normalize: Optional[bool]=True,
                 num_images: Optional[int]=None,
                 on_train_epoch_end: Optional[bool]=True,
                 on_validation_epoch_end: Optional[bool]=True,
                 ) -> None:
        """
        Inputs:
        ------
        RGB: Dict[List[int]]: len(RGB) is the amount of images to show in RGB. The combination of bands to associate to RGB is given in the values of the dictionary. The key string is unimportant. default: {'image': [3,2,1]}
        normalize: bool: if True, normalizes all the bands of all the images in the batch between 0.1 and 99.9 percentiles
        num_images: int: number of images in the batch to log.

        """
        super().__init__()
        self.RGB = RGB if RGB is not None else {}
        self.normalize = normalize
        self.num_images = num_images
        self.log_valid = on_validation_epoch_end
        self.log_train = on_train_epoch_end

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

        return (image-min_im)/(max_im-min_im)


    def _log_batch_predictions(self, trainer, images, targets, probabilities, mode):
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
            for rgb in self.RGB.values():
                rgb_img.append(image[rgb,...])

            target= targets[i].repeat(3,1,1)
            prob  = probabilities[i].repeat(3,1,1)

            if i == 0:
                data = torch.stack([*rgb_img, target, prob], dim=0)
            else:
                data = torch.cat([data, torch.stack([*rgb_img, target, prob], dim=0)], dim=0)

        grid = torchvision.utils.make_grid(tensor=data, nrow=len(self.RGB)+2)

        try:    # TODO: Better implementation of this. pl is changing multi logger api...
            trainer.logger.experiment.add_image(f"{mode}_images", grid, global_step=trainer.global_step)
        except:
            print('Image logger failed!')
        try:    # This works for tensorboard as 1st logger if several
            trainer.logger.experiment[0].add_image(f"{mode}_images", grid, global_step=trainer.global_step)
        except:
            print('Image logger failed!')


    def on_validation_epoch_end(self, trainer: pl.Trainer, plModel: pl.LightningModule) -> None:
        if self.log_valid and plModel.log_images['valid'] is not None:
            images, targets, prob = plModel.log_images['valid']
            self._log_batch_predictions(trainer, images, targets, prob, 'valid')

    def on_train_epoch_end(self, trainer: pl.Trainer, plModel: pl.LightningModule) -> None:
        if self.log_train and plModel.log_images['train'] is not None:
            images, targets, prob = plModel.log_images['train']
            self._log_batch_predictions(trainer, images, targets, prob, 'train')

