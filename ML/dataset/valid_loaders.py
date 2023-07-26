from ML import *
from ML.dataset.loader import *


class Valid_Loader(Loader):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.X = self.data["text"].to_numpy()

    def __getitem__(self, index) -> np.array:
        return self.transform(self.X[index])
