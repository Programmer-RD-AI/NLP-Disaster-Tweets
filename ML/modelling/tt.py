from ML import *


class TL(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        input_dim: int = 768,
        classifier_head: torchtext.models = RobertaClassificationHead,
        model: torchtext.models = XLMR_BASE_ENCODER,
    ) -> None:
        """The initialization of the Transfer Learning Model

        Keyword arguments:
        num_classes -- the number of classes to be outputted
        input_dim -- the input dimension for the Encoder
        classifier_head -- the head of the Encoder model
        model -- the encoder model it self
        Return: None
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.classifier_head = classifier_head(num_classes, input_dim)
        self.model = model.get_model(head=self.classifier_head).to(device)

    def freeze(self):
        pass

    def forward(self, X) -> torch.tensor:
        """the forward function where the input / X data is inputed and the logits are outputed

        Keyword arguments:
        X -- the input data
        Return: the logits of the each input_data in the shape of ((num_classes),len(X))
        """

        return self.model(X)
