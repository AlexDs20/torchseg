import torch

from torchseg.utils import make_subtiles, unmake_subtiles


class TestSubtiles():
    def test_make_subtiles(self):
        shape = (12, 512, 1024)
        h, w = 16, 16
        data = torch.rand(*shape)

        out = make_subtiles(data, h, w)
        assert out.shape == ((shape[1] // h) * (shape[2] // w), shape[0], h, w)

    def test_unmake_subtiles(self):
        m, n = 512, 1024
        c, h, w = 13, 128, 128
        shape = ((m // h) * (n // w), c, h, w)
        data = torch.rand(*shape)

        out = unmake_subtiles(data, m, n)
        assert out.shape == (c, m, n)

    def test_make_unmake_id(self):
        h, w = 16, 16
        c, m, n = (13, 512, 1024)
        shape = (c, m, n)
        data = torch.rand(*shape)

        out = make_subtiles(data, h, w)
        out = unmake_subtiles(out, m, n)
        assert torch.all(out == data)
