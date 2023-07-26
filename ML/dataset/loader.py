from ML import *


class Loader(Dataset):
    def __init__(self, path: str, transform=None) -> None:
        self.path = path
        self.transform = transform
        self.data: pd.DataFrame = pd.read_csv(self.path)

    def __len__(self) -> int:
        return len(self.data)
