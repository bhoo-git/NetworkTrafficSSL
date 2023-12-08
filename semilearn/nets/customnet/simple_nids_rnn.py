# Edit: Robin Bhoo

import torch
import torch.nn as nn

class SimpleRNN(nn.Module):

    def __init__(self, in_channels=1, hid_dim=128, num_classes=10, n_layers=3, dropout=0.2):
        super(SimpleRNN, self).__init__()
        # Specify simple RNN to be used for Payload_Byte data
        self.rnn = nn.LSTM(in_channels, hid_dim, 
                           num_layers=n_layers, batch_first=True, 
                           dropout=dropout)
        self.classifier = nn.Linear(hid_dim, num_classes)
        #self.num_features = 1504
        self._init_weights()

    def _init_weights(self):
        pass
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.xavier_uniform_(m.weight, gain=1)
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        # Because nn.LSTM expects (N, L, H_in) as the shape for batched inputs, we check and process input x accordingly.
        # N: batch size | L: sequence length | H_in: input_size. For Payload_byte data, L = 1504 and H_in = 1.
        if x.dim() == 2 and x.shape[1] == 1504:
            x = x.unsqueeze(dim=-1)
            assert x.dim() == 3, "Input to RNN is not the expected shape of (N, L, H_in)."

        if only_fc:
            logits = self.classifier(x)
            return logits[:,-1,:]

        pooled_output = self.extract(x)

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits[:,-1,:], 'feat':pooled_output}
        return result_dict

    def extract(self, x):
        #TODO: Not Applicable for MLP, but for other architectures
        self.rnn.flatten_parameters()
        pooled_output, _ = self.rnn(x)
        return pooled_output


def simple_nids_rnn(pretrained:bool=False, pretrained_path=None, **kwargs):
    model = SimpleRNN(**kwargs)
    return model

if __name__ == '__main__':
    model = simple_nids_rnn(**model_kwargs)
    print(model)