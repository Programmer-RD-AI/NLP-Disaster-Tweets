from ML import *
import torchtext.functional as F
from ML.helper_functions.test import *


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
        lr_schedular: bool = None,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.config = config
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_schedular = lr_schedular

    def train(self, run_name):
        print(torchinfo.summary(self.model))
        wandb.init(project=PROJECT_NAME, name=run_name, config=self.config)
        wandb.watch(self.model, log="all")
        iterator = tqdm(range(self.epochs))
        for _ in iterator:
            torch.cuda.empty_cache()
            for i, (X, y) in enumerate(self.train_dataloader):
                y = y[0]
                torch.cuda.empty_cache()
                X = F.to_tensor(X, padding_value=1).to("cuda")
                y = torch.tensor(y).to(dtype=torch.long, device="cuda")
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y.view(-1, 1).squeeze(1))
                loss.backward()
                self.optimizer.step()
                iterator.set_description(f"{i}/{len(self.train_dataloader)}")
            if self.lr_schedular:
                self.lr_schedular.step()
            iterator.set_description(f"Testing...")
            self.model.eval()
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
            iterator.set_description(f"Testing Done")
            self.model.train()
        wandb.save()
        wandb.finish()
