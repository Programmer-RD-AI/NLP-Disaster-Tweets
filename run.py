from ML import *

lrs = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for lr in lrs:
    train_data_loader, test_data_loader, valid_data_loader = Load_Data(
        Main_DL,
        Valid_Loader,
        [
            "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/ML/data/train.csv",
            32,
            Transformer().transform(),
        ],
        [
            "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/ML/data/test.csv",
            1,
        ],
        0.25,
        42,
    ).ld()
    model = TL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    config = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "learning_rate": lr,
    }
    Train(
        model,
        5,
        config,
        train_data_loader,
        test_data_loader,
        valid_data_loader,
        criterion,
        optimizer,
    ).train(f"{lr}")
