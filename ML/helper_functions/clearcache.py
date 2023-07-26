from ML import *


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, *args):
        torch.cuda.empty_cache()
