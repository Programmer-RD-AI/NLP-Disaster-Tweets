from ML import *
from ML.dataset.loader import *

"""Contains the main dataloader used to load the train and testing data"""


class Main_DL(Loader):
    def __init__(
        self,
        train: bool = True,
        test_split: float = 0.125,
        seed: int = 42,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """initalization of the Main Dataloader which inherits from the `Loader` class

        Keyword arguments:
        train -- bool, if the data is for training or testing
        test_split -- float between 0 and 1
        seed -- int, to prevent change of results
        batch_size -- int, the size of the batches
        Return: None
        """
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
        self.get_batches()

    def get_batches(self) -> None:
        """create the batches for training"""
        X = self.X_train if self.train else self.X_test
        y = self.y_train if self.train else self.y_test
        X_batches = []
        y_batches = []
        iterator = tqdm(
            range(0, (round(len(X) / self.batch_size) - 1) * self.batch_size, self.batch_size)
        )
        for i in iterator:
            X_iter = X[i : i + self.batch_size]
            y_iter = y[i : i + self.batch_size]
            new_X_iter = []
            for j in X_iter:
                new_X_iter.append(self.transform(j))
            X_batches.append(new_X_iter)
            y_batches.append([y_iter])
        if self.train:
            self.X_train = X_batches
            self.y_train = np.array(y_batches)
        else:
            self.X_test = X_batches
            self.y_test = np.array(y_batches)

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        """get an specific item using an specific index

        Keyword arguments:
        index -- the index of the item to retrieve
        Return: Tuple
        """
        if self.train:
            return (
                self.X_train[index],
                [self.y_train[index]],
            )
        return (
            self.X_test[index],
            [self.y_test[index]],
        )

    def __len__(self) -> int:
        """get the length / no. of batches of the dataset

        Return: Int
        """
        return len(self.y_train) if self.train else len(self.y_test)
