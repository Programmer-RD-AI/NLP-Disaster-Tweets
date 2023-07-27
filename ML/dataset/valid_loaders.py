from ML import *
from ML.dataset.loader import *


class Valid_Loader(Loader):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.data["id"].dropna(inplace=True)
        self.X = self.data["text"].to_numpy()
        self.ids = self.data["id"].to_numpy()
        print(len(self.X), len(self.ids))

    def __getitem__(self, index) -> np.array:
        return (self.ids[index], [self.transform(self.X[index])])
