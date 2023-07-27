from ML import *


class Randomize(Module):
    def __init__(self, p: float = 0.5) -> None:
        self.p = 0.5 if p > 1 else p

    def __call__(
        self,
        X: torch.tensor,
    ) -> torch.tensor:
        """it will randamize the order of a given tensor / list, this will make it harder for the model to understand inturn hopefully understand as for the

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        if torch.rand(1).item() < self.p:
            np.random.shuffle(X)
            return X
        return X
