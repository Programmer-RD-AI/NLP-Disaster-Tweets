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
        lr_schedular=None,
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

    def train(self, run_name: str) -> None:
        print(torchinfo.summary(self.model))
        wandb.init(project=PROJECT_NAME, name=run_name, config=self.config)
        wandb.watch(self.model, log="all")
        iterator = tqdm(range(self.epochs))
        for epoch in iterator:
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
            Test(
                self.train_dataloader,
                self.valid_dataloader,
                self.criterion,
                self.model,
                "Train",
            ).make_predictions(run_name, epoch)
            iterator.set_description(f"Testing Done")
            self.model.train()
        wandb.save()
        wandb.finish()
        self.save_model(run_name)

    def save_model(self, run_name: str) -> None:
        if run_name not in os.listdir("./ML/predictions/"):
            os.mkdir(f"./ML/predictions/{run_name}")
        torch.save(self.model, f"./ML/predictions/{run_name}/model.pt")
        torch.save(self.model, f"./ML/predictions/{run_name}/model.pth")
        torch.save(self.model.state_dict(), f"./ML/predictions/{run_name}/model_state_dict.pt")
        torch.save(self.model.state_dict(), f"./ML/predictions/{run_name}/model_state_dict.pth")
