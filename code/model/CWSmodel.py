import torch
import torch.nn as nn
import transformers


class CWSmodel(nn.Module):
    def __init__(
        self, model_name: str, total_token: int, drop: float = 0.0, criteria: int = 10
    ):
        super(CWSmodel, self).__init__()

        self.pretrained_model = transformers.BertModel.from_pretrained('../../bert-base-chinese',)
        self.pretrained_model.resize_token_embeddings(total_token)

        # Define layers for word segmentation
        self.segmentor_layers = nn.Sequential(
            nn.Linear(in_features=768, out_features=48),
            nn.LayerNorm(48),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=48, out_features=4)
        )

        # Criteria classifier
        self.criteria = nn.Linear(in_features=768, out_features=criteria)
        self.dropout = nn.Dropout(p=drop, inplace=False)

    def forward(self, batch_x):
        model_predict = self.pretrained_model(**batch_x)

        # Criteria classification
        criteria = self.criteria(self.dropout(model_predict.last_hidden_state[:, 1, :]))

        # Remove the [CLS] token
        last_hidden_state = self.dropout(model_predict.last_hidden_state[:, 1:, :])

        # Apply word segmentation layers
        last_hidden_state = self.segmentor_layers(last_hidden_state)

        return criteria, last_hidden_state

    def inference(self, batch_x):
        model_predict = self.pretrained_model(**batch_x)

        # The rest of the tokens
        last_hidden_state = self.segmentor_layers(model_predict.last_hidden_state[:, 2:, :])

        return last_hidden_state
