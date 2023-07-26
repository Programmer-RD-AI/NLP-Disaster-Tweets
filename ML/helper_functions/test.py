from ML import *


class Test:
    def __init__(
        self,
        test_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        model: torch.nn,
        name: str,
    ) -> None:
        self.test_dataloader = test_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.model = model
        self.name = name

    def test(self):
        a_tot = 0
        p_tot = 0
        r_tot = 0
        f1_tot = 0
        n = 0
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                y = y[0]
                X = F.to_tensor(X, padding_value=1).to("cuda")
                y = torch.tensor(y).to("cuda")
                preds = torch.argmax(torch.softmax(self.model(X), dim=1), dim=1)
                results = classification_report(
                    preds, y.view(-1, 1).squeeze(1), class_names=["0", "1"]
                )
                precision = results["weighted avg"]["precision"]
                recall = results["weighted avg"]["recall"]
                f1score = results["weighted avg"]["f1-score"]
                accuracy = results["accuracy"]
                a_tot += accuracy
                p_tot += precision
                r_tot += recall
                f1_tot += f1score
                n += 1
        return {
            f"{self.name} precision": p_tot / n,
            f"{self.name} recall": r_tot / n,
            f"{self.name} f1-score": f1_tot / n,
            f"{self.name} accuracy": a_tot / n,
        }
