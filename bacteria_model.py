from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torch import nn

class BacteriaModel(nn.Module):
    def __init__(self, num_classes, start_frozen=False):
        super(BacteriaModel, self).__init__()

        # Model architecture
        self.vit_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.fc_layer = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        # For frozen start
        if start_frozen:
            for param in self.res_model.parameters():
                param.requires_grad = False
        

    def unfreeze(self, n_layers):
        # For unfreezing. Didn't work as well for us
        child_list = list(self.res_model.children())
        for child in child_list[len(child_list) - n_layers - 1:]:
           for param in child.parameters():
               param.requires_grad = True

    def forward(self, x):
        # TODO: Add a layer for the fine resolution
        x = self.vit_model(x)
        return self.fc_layer(x)
