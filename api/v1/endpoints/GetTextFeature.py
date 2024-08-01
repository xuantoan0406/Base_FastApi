import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.util import cos_sim


class GetTextFeature:
    def __init__(self, weights_model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(weights_model)
        self.model = AutoModel.from_pretrained(weights_model).to(self.device)

    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_feature(self, text):
        batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            self.device)
        outputs = self.model(**batch_dict)
        text_features = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).detach()
        text_features=text_features.to("cpu").view(-1).tolist()
        return text_features

    @staticmethod
    def cosine_distance(tensor1, tensor2):
        return cos_sim(tensor1, tensor2).item()
