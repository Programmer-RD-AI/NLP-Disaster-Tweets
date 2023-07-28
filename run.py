from ML import *


def train(
    batch_size: int = 32,
    lr: float = 0.01,
    test_split: float = 0.25,
    optimizer=optim.Adam,
    epochs: int = 5,
    name: str = "",
    lr_schedular=None,
    transforms=None,
):
    train_data_loader, test_data_loader, valid_data_loader = Load_Data(
        Main_DL,
        Valid_Loader,
        [
            "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/ML/data/train.csv",
            batch_size,
            transforms,
        ],
        [
            "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/ML/data/test.csv",
            1,
        ],
        test_split,
        42,
    ).ld()
    model = TL().to(device)
    optimizer = optimizer(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    config = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "learning_rate": lr,
    }
    Train(
        model,
        epochs,
        config,
        train_data_loader,
        test_data_loader,
        valid_data_loader,
        criterion,
        optimizer,
    ).train(f"{name}")


train(
    transforms=Transformer().transform(),
    batch_size=16,
    lr=1e-3,
    test_split=0.25,
    optimizer=optim.Adam,
    lr_schedular=None,
    name=f"1e-3",
)
train(
    transforms=Transformer().transform(),
    batch_size=16,
    lr=1e-4,
    test_split=0.25,
    optimizer=optim.Adam,
    lr_schedular=None,
    name=f"1e-4",
)
train(
    transforms=Transformer().transform(),
    batch_size=16,
    lr=1e-5,
    test_split=0.25,
    optimizer=optim.Adam,
    lr_schedular=None,
    name=f"1e-5",
)
train(
    transforms=Transformer().transform(),
    batch_size=16,
    lr=1e-6,
    test_split=0.25,
    optimizer=optim.Adam,
    lr_schedular=None,
    name=f"1e-6",
)
train(
    transforms=Transformer().transform(),
    batch_size=16,
    lr=1e-7,
    test_split=0.25,
    optimizer=optim.Adam,
    lr_schedular=None,
    name=f"1e-7",
)
