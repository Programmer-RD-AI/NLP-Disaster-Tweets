from ML import *


class ClearCache:
    def __enter__(self) -> None:
        torch.cuda.empty_cache()

    def __exit__(self, *args) -> None:
        torch.cuda.empty_cache()
