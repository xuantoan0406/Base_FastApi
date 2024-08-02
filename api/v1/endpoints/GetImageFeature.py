import torch
import timm
from torchvision import transforms
from collections import OrderedDict
from PIL import Image


class GetImageFeature:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_features(self, image):
        img = Image.open(image)
        img_tensor = self.preprocess(img).unsqueeze(0)
        features = {}

        def hook_fn(module, input, output):
            features['intermediate'] = output

        # Register hook on the desired layer
        hook = self.model.patch_embed.register_forward_hook(hook_fn)

        # Extract features
        with torch.no_grad():
            _ = self.model(img_tensor)

        # Remove the hook
        hook.remove()

        # Print the shape of the intermediate features
        print(features['intermediate'].shape)
        return features
# Load pre-trained Swin Transformer model
# model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
# model.eval()
#
# # Define preprocessing transformations
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # Load and preprocess image
# img = Image.open('assets/images/00001-3171169063.png')
# img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
#
# # Extract features
# with torch.no_grad():
#     features = model(img_tensor)
#
# # Print feature shape
# print(features.shape)
