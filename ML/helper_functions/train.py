from ML import *


class Train:
    def __init__(
        self,
        model: torchtext.models,
        epochs: int,
        config: dict,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.config = config
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, run_name):
        print(torchinfo.summary(self.model))
        wandb.init(project=PROJECT_NAME, entity=run_name)
        wandb.watch(self.model, log="all")
        iterator = tqdm(range(self.epochs))
        for _ in iterator:
            for i, (X, y) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step(f"{i}/{len(self.train_dataloader)}")
                iterator.set_description()
            if self.lr_schedular:
                self.lr_schedular.step()
            wandb.log(
                Test(
                    self.test_dataloader, self.valid_dataloader, self.criterion, self.model, "Test"
                ).test()
            )
            wandb.log(
                Test(
                    self.train_dataloader,
                    self.valid_dataloader,
                    self.criterion,
                    self.model,
                    "Train",
                ).test()
            )
        wandb.save()
        wandb.finish()
