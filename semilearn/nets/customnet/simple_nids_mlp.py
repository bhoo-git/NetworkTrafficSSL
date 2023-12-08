# Edit: Robin Bhoo

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMLP, self).__init__()

        # Specify simple MLP to be used for Payload_Byte data
        self.dropout = torch.nn.Dropout(p=0.25)
        self.num_features = 1504
        self.mlp = nn.Sequential(*[
            nn.Linear(self.num_features, 1000),
            nn.GELU(), self.dropout,
            nn.Linear(1000, 500),
            nn.GELU(), self.dropout,
            nn.Linear(500, 500),
            nn.GELU(), self.dropout,
            nn.Linear(500, 250),
            nn.GELU(), self.dropout,
            nn.Linear(250, 250),
            nn.GELU(), self.dropout,
            nn.Linear(250, 100)
        ])

        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            logits = self.classifier(x)
            return logits

        pooled_output = self.extract(x)

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}
        return result_dict

    def extract(self, x):
        #TODO: Not Applicable for MLP, but for other architectures
        pooled_output = self.mlp(x)
        return pooled_output


def simple_nids_mlp(pretrained:bool=False, pretrained_path=None, **kwargs):
    model = SimpleMLP(**kwargs)
    return model

if __name__ == '__main__':
    model = simple_nids_mlp(**model_kwargs)
    print(model)