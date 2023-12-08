# Edit: Robin Bhoo

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        #Stacking Residual Blocks
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding='same'),
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            #torch.nn.Dropout(0.2)
            ) 
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, padding='same'),
            nn.MaxPool2d(kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #torch.nn.Dropout(0.2)
            ) 
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding='same'),
            nn.MaxPool2d(kernel_size=3, padding=1),
            #nn.BatchNorm2d(192),
            nn.ReLU(),
            #torch.nn.Dropout(0.2)
            ) 
        self.conv4= torch.nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=4, padding='same'),
            nn.MaxPool2d(kernel_size=3, padding=1),
            #nn.BatchNorm2d(384),
            nn.ReLU(),
            torch.nn.Dropout(0.2)
            )  

        # Final Classification Layer
        self.classifier = torch.nn.Sequential(
                # Prep features from conv_5
                #nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                #nn.LayerNorm(768),
                nn.Linear(7296, 64),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
         )
        
        # Weight Initialization for conv/linear layers
        self.blocks = [self.conv1, self.conv2, self.conv3, self.conv4]
        for block in self.blocks: #Conv Layer Weights
            for layer in block:
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.xavier_normal_(layer.weight)

        for layer in self.classifier: #Linear Layer Weights
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight)

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
        x = self.conv1(x)
        x = self.conv2(x)  
        x = self.conv3(x)
        pooled_output = self.conv4(x)
        return pooled_output


def simple_nids_cnn(pretrained:bool=False, pretrained_path=None, **kwargs):
    model = SimpleCNN(**kwargs)
    return model

if __name__ == '__main__':
    model = simple_nids_cnn(**model_kwargs)
    print(model)