from ML import *
from ML.dataset.loader import *


class Main_DL(Loader):
    def __init__(
        self,
        train: bool = True,
        test_split: float = 0.125,
        seed: int = 42,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.X = self.data["text"].to_numpy()
        self.y = self.data["target"].to_numpy()
        self.train = train
        self.test_split = test_split
        self.seed = seed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_split, random_state=seed
        )
        self.X_train, self.X_test, self.y_train, self.y_test = (
            np.array(self.X_train),
            np.array(self.X_test),
            np.array(self.y_train),
            np.array(self.y_test),
        )
        self.batch_size = batch_size
        # self.get_batches()
        

    def get_batches(self):
        X = self.X_train if self.train else self.X_test
        y = self.y_train if self.train else self.y_test
        X_batches = []
        y_batches = []
        for i in range(0, len(X), self.batch_size):
            X_iter = X[i : i + self.batch_size]
            y_iter = y[i : i + self.batch_size]
            X_batches.append(X_iter)
            y_batches.append(y_iter)
        if self.train:
            self.X_train = F.to_tensor(X_batches, padding_value=1)
            self.y_train = np.array(y_batches)
        else:
            self.X_test = F.to_tensor(X_batches, padding_value=1)
            self.y_test = np.array(y_batches)

        print(X_batches[0], y_batches[0])

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        if self.train:
            return (
                self.transform(self.X_train[index]),
                [self.y_train[index]],
            )
        return (
            self.transform(self.X_test[index]),
            [self.y_test[index]],
        )

    def __len__(self) -> int:
        return len(self.y_train) if self.train else len(self.y_test)
