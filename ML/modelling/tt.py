from ML import *


class TL(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_dim: int = 768,
        classifier_head: torchtext.models = RobertaClassificationHead,
        model: torchtext.models = XLMR_LARGE_ENCODER,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.classifier_head = classifier_head(num_classes, input_dim)
        self.model = model.get_model(head=self.classifier_head).to(device)

    def forward(self, X):
        return self.model(X)
