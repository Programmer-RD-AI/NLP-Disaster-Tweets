from ML import *


class Load_Data:
    def __init__(
        self,
        dataset_main: Dataset,
        dataset_valid: Dataset,
        main: list,
        valid: list,
        test_split,
        seed,
    ) -> None:
        self.dataset_main = dataset_main
        self.dataset_valid = dataset_valid
        self.main_path = main[0]
        self.main_batch_size = main[1]
        self.main_transform = main[2]
        self.test_split = test_split
        self.seed = seed
        self.valid_path = valid[0]
        self.valid_batch_size = valid[1]
        self.main = main
        self.valid = valid

    def ld(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        self.train_data_loader = DataLoader(
            self.dataset_main(
                path=self.main_path,
                transform=self.main_transform,
                train=True,
                test_split=self.test_split,
                seed=self.seed,
            ),
            batch_size=None,
            shuffle=True,
            num_workers=round(os.cpu_count() / 2),
        )
        self.test_data_loader = DataLoader(
            self.dataset_main(
                path=self.main_path,
                transform=self.main_transform,
                train=False,
                test_split=self.test_split,
                seed=self.seed,
            ),
            batch_size=None,
            shuffle=True,
            num_workers=round(os.cpu_count() / 2),
        )
        self.valid_data_loader = DataLoader(
            self.dataset_valid(self.valid_path, self.main_transform),
            batch_size=None,
            shuffle=False,
            num_workers=round(os.cpu_count() / 2),
        )
        return (self.train_data_loader, self.test_data_loader, self.valid_data_loader)
