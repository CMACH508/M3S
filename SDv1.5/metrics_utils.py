import torch
from PIL import Image
import clip 
from torchvision import transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from warnings import filterwarnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
#import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image, ImageFile
import torch.nn.functional as F
from torch.nn import CosineSimilarity
import timm

class CLIPScorer:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval() 

    def __call__(self, image, text):
        if isinstance(image, str):
            image = Image.open(image)
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T)
        return torch.max(similarity, torch.tensor(0.0)).item()

class VGGCalculator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()

        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def _get_features(self, img_tensor):
        features = {}
        x = img_tensor
        for index, (name, layer) in enumerate(self.vgg._modules.items()):
            x = layer(x)
            if index in [0,5,10,19,28]:
                features[index] = x.clone()
                if index == 28:
                    break
        return features
    
    @staticmethod
    def _gram_matrix(input_tensor):
        b, c, h, w = input_tensor.size()
        features = input_tensor.view(c, h * w)
        return torch.mm(features, features.t()) / (c * h * w)
    
    def calculate_similarity(self, img_path1, img_path2):
        img1 = self.preprocess(Image.open(img_path1).convert('RGB')).unsqueeze(0).to(self.device)
        img2 = self.preprocess(Image.open(img_path2).convert('RGB')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features1 = self._get_features(img1)
            features2 = self._get_features(img2)
        gram1 = {layer: self._gram_matrix(feat) for layer, feat in features1.items()}
        gram2 = {layer: self._gram_matrix(feat) for layer, feat in features2.items()}

        loss = 0.0
        for layer in [0,5,10,19,28]:#self.style_layers.keys():
            loss += torch.mean((gram1[layer] - gram2[layer])**2)
        
        return loss
    

class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("./notebooks/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
model.load_state_dict(s)
model.to("cuda")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   


def calculate_aesthetic_score(img_path):
    with torch.no_grad():
        pil_image = Image.open(img_path)
        image = preprocess(pil_image).unsqueeze(0).to(device)
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
    return prediction

class DINOStyleScorer:
    def __init__(self, model_name="vit_base_patch8_224_dino", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
        self.model.eval()  

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_image(self, image_input):
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError("String or PIL.Image")

    def _extract_features(self, image_input):
        image = self._load_image(image_input)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        with torch.no_grad():
            features = self.model(image_tensor)  # [1, 768]
        return features.squeeze(0)  # [768]

    def __call__(self, image1, image2):
        features1 = self._extract_features(image1)
        features2 = self._extract_features(image2)
        return CosineSimilarity(dim=0)(features1, features2).item()