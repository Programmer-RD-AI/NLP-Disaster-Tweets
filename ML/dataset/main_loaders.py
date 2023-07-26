from ML import *


class Main_DL(Loader):
    def __init__(self, train: bool = True, test_split: float = 0.125, seed: int = 42) -> None:
        super().__init__()
        self.X = self.data["text"].to_numpy()
        self.y = self.data["target"].to_numpy()
        self.train = train
        self.test_split = test_split
        self.seed = seed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_split, random_state=seed
        )

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        if self.train:
            return (
                self.transform(self.X_train[index]) if self.transform else self.X_train[index],
                self.y_train[index],
            )
        return (
            self.transform(self.X_test[index]) if self.transform else self.X_test[index],
            self.y_test[index],
        )

    def __len__(self) -> int:
        return len(self.y_train) if self.train else len(self.y_test)
