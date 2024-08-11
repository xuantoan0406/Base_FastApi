import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn as nn


class GetImageFeature:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True).to(self.device)
        self.model.eval()
        self.fc = None
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_features(self, image):
        img = Image.open(image)
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = {}

        def hook_fn(module, input, output):
            features['intermediate'] = output

        # Register hook on the desired layer
        hook = self.model.patch_embed.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = self.model(img_tensor)
        print(_.size())

        # Remove the hook
        hook.remove()
        intermediate_features = features['intermediate']
        num_features = intermediate_features.size(1) * intermediate_features.size(2) * intermediate_features.size(3)
        self.fc = nn.Linear(num_features, 1024).to(self.device)
        flattened_features = intermediate_features.view(intermediate_features.size(0), -1)
        output_features = self.fc(flattened_features)

        return output_features
