from ML import *


class Valid_Loader(Loader):
    def __init__(self) -> None:
        super().__init__()
        self.X = self.data["text"].to_numpy()

    def __getitem__(self, index) -> np.array:
        return self.X[index]
