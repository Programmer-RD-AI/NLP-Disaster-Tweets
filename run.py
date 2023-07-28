from ML import *

print(ROBERTA_BASE_ENCODER)


train_data_loader, test_data_loader, valid_data_loader = Load_Data(
    Main_DL,
    Valid_Loader,
    [
        "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/ML/data/train.csv",
        16,
        Transformer().transform(),
    ],
    [
        "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/ML/data/test.csv",
        1,
    ],
    0.125,
    42,
).ld()
model = TL().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
config = {
    "model": model,
    "criterion": criterion,
    "optimizer": optimizer,
    "learning_rate": 1e-5,
}
Train(
    model,
    25,
    config,
    train_data_loader,
    test_data_loader,
    valid_data_loader,
    criterion,
    optimizer,
).train(f"final")
