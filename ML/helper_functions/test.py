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

    def test(self) -> Dict:
        a_tot = 0
        p_tot = 0
        r_tot = 0
        f1_tot = 0
        l_tot = 0
        n = 0
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                y = y[0]
                X = F.to_tensor(X, padding_value=1).to("cuda")
                y = torch.tensor(y).to("cuda")
                preds = torch.argmax(torch.softmax(self.model(X), dim=1), dim=1)
                loss = self.criterion(preds.float(), y.view(-1, 1).squeeze(1).float())
                results = classification_report(
                    preds.cpu(), y.view(-1, 1).squeeze(1).cpu(), output_dict=True
                )
                precision = results["weighted avg"]["precision"]
                recall = results["weighted avg"]["recall"]
                f1score = results["weighted avg"]["f1-score"]
                accuracy = results["accuracy"]
                a_tot += accuracy
                p_tot += precision
                r_tot += recall
                f1_tot += f1score
                l_tot += loss.item()
                n += 1
        return {
            f"{self.name} precision": p_tot / n,
            f"{self.name} recall": r_tot / n,
            f"{self.name} f1-score": f1_tot / n,
            f"{self.name} accuracy": a_tot / n,
            f"{self.name} loss": l_tot / n,
        }

    def make_predictions(self, run_name: str, epoch: int) -> pd.DataFrame:
        ids = []
        target = []
        for _id, X in self.valid_dataloader:
            X = F.to_tensor(X, padding_value=1).to("cuda")
            pred = torch.argmax(torch.softmax(self.model(X), dim=1), dim=1).squeeze().cpu().item()
            ids.append(_id.item())
            target.append(pred)
        if run_name not in os.listdir("./ML/predictions/"):
            os.mkdir(f"./ML/predictions/{run_name}")
        df = pd.DataFrame({"id": ids, "target": target})
        df.to_csv(f"./ML/predictions/{run_name}/{epoch}.csv", index=False)
        return df
