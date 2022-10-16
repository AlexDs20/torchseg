class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, target):
        return target.to(self.dtype)
