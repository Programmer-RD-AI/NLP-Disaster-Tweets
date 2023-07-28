from ML import *

"""This file contains the main loader class, which is inherited from in `main_loaders` and `valid_loaders`"""


class Loader(Dataset):
    def __init__(self, path: str, transform: torchtext.transforms) -> None:
        """initalization of the Loader class

        Keyword arguments:
        path -- path of the .csv file to load
        transform -- the transformation to be applied to the data
        Return: None
        """
        self.path = path
        self.transform = transform
        self.data: pd.DataFrame = pd.read_csv(self.path)

    def __len__(self) -> int:
        """returns the length of the dataset"""
        return len(self.data)
