from sentence_transformers import SentenceTransformer
from torch import nn
import torch
import torch.optim as optim
import numpy as np


class SiameseBert(nn.Module):
    def __init__(self):
        super(SiameseBert, self).__init__()
        # base bert layer.
        self.bert = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        # add linear layers
        self.fc = nn.Sequential(
            nn.Linear(768, 200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward_one(self, word):
        embedding = torch.from_numpy(self.bert.encode(word))
        return embedding

    def forward(self, word1, word2):
        # siamese network embeds two words separately, then learns from their distance
        word1_embedding = self.forward_one(word1)
        word2_embedding = self.forward_one(word2)
        distance = word1_embedding.sub(word2_embedding).pow(2)
        x = self.fc(distance)
        return x
