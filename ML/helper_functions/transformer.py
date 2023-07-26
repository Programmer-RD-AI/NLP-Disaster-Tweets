from ML import *


class Transformer:
    def __init__(
        self,
        padding_idx: int = 1,
        beg_idx: int = 0,
        end_idx: int = 2,
        max_seq_len: int = 256 - 2,
        vocab_path=r"https://download.pytorch.org/models/text/xlmr.vocab.pt",
        spm_model_path=r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model",
        tokenizer: torchtext.transforms = SentencePieceTokenizer,
        vocab_transform: torchtext.transforms = VocabTransform,
        truncate: torchtext.transforms = Truncate,
    ) -> None:
        self.padding_idx = padding_idx
        self.beg_idx = beg_idx
        self.end_idx = end_idx
        self.max_seq_len = max_seq_len
        self.vocab_path = vocab_path
        self.spm_model_path = spm_model_path
        self.tokenizer = tokenizer
        self.vocab_transform = vocab_transform
        self.truncate = truncate

    def transform(self) -> torchtext.transforms.Sequential:
        t = torchtext.transforms.Sequential(
            self.tokenizer(self.spm_model_path),
            self.vocab_transform(load_state_dict_from_url(self.vocab_path)),
            self.truncate(self.max_seq_len),
            AddToken(self.beg_idx, begin=True),
            AddToken(self.end_idx, begin=False),
        )
        return t

    def model_transform(self, model):
        return model.transforms()
