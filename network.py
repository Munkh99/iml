import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights


class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()

        # Pre-trained ResNet-50 as feature extractor
        self.feature_extractor = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor.fc = nn.Identity()
        # Fully connected layer for embedding
        self.embedding = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x)
        return x

