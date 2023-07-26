from ML import *

train_data_loader, test_data_loader, valid_data_loader = Load_Data(
    Main_DL,
    Valid_Loader,
    [
        "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/learning/train.csv",
        32,
        Transformer().transform(),
    ],
    [
        "/media/user/Main/Programmer-RD-AI/Programming/Learning/JS/NLP-Disaster-Tweets/learning/test.csv",
        1,
    ],
    0.125,
    42,
)
model = TL().to(device)
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
config = {
    "model": model,
    "criterion": criterion,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
}
Train(
    model, 10, config, train_dataloader, test_dataloader, valid_dataloader, criterion, optimizer
).train("baseline")
