from ML import *
from ML.dataset.loader import *

"""Contains the validation dataloader used to load the validation"""


class Valid_Loader(Loader):
    def __init__(self, *args) -> None:
        """Initialization of the Valid Loader which inherits from the Loader class"""
        super().__init__(*args)
        self.data["id"].dropna(inplace=True)
        self.X = self.data["text"].to_numpy()
        self.ids = self.data["id"].to_numpy()
        print(len(self.X), len(self.ids))

    def __getitem__(self, index) -> np.array:
        """get and specific item according to the index given

        Keyword arguments:
        index -- The index of the item
        Return: Tuple
        """
        return (self.ids[index], [self.transform(self.X[index])])
