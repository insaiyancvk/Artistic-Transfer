from PIL import Image
import os, cv2, torch, json
import torch.nn as nn
from torchvision import datasets, transforms, models
import numpy as np
from math import sqrt
import torch.optim as optim
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self, vgg_path="vgg16.pth"):
        super(VGG16, self).__init__()
        # Load VGG Skeleton, Pretrained Weights
        vgg16_features = models.vgg16(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16_features.features

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if (name=='22'):
                    break

        return features

class TransformerNetworkNN(nn.Module):
    """Feedforward Transformation Network without Tanh
    reference: https://arxiv.org/abs/1603.08155 
    exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """
    def __init__(self):
        super(TransformerNetworkNN, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3), 
            ResidualLayer(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class ResidualLayer(nn.Module):
    """
    Deep Residual Learning for Image Recognition

    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x                     # preserve residual
        out = self.relu(self.conv1(x))   # 1st conv layer + activation
        out = self.conv2(out)            # 2nd conv layer
        out = out + identity             # add residual
        return out

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution 
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

# Gram Matrix
def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H*W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C*H*W)

# Load image file
def load_image(path):
    # Images loaded as BGR
    img = cv2.imread(path)
    return img

def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)

# Preprocessing ~ Image to Tensor
def itot(img, max_size=None):
    # Rescale the image
    if (max_size==None):
        itot_t = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])    
    else:
        H, W, C = img.shape
        image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])
        itot_t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    # Convert image to tensor
    tensor = itot_t(img)

    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor

# Preprocessing ~ Tensor to Image
def ttoi(tensor):

    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    #img = ttoi_t(tensor)
    img = tensor.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

TRAIN_IMAGE_SIZE = 512
DATASET_PATH = "/content/train"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "/content/style.jpg"
BATCH_SIZE = 4
CONTENT_WEIGHT = 17
STYLE_WEIGHT = 50
TV_WEIGHT = 1e-6
ADAM_LR = 0.001
SAVE_MODEL_PATH = "/content/models/"
SAVE_IMAGE_PATH = "/content/images/"
SAVE_MODEL_EVERY = 200 # 800 Images with batch size 4
SEED = 68
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = ("cuda" if torch.cuda.is_available() else "cpu")


# Dataset and Dataloader
transform = transforms.Compose([
    transforms.Resize(TRAIN_IMAGE_SIZE),
    transforms.CenterCrop(TRAIN_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Get Style Features
imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float16).reshape(1,3,1,1).to(device)
imagenet_mean = torch.tensor([103.939, 116.779, 123.68], dtype=torch.float16).reshape(1,3,1,1).to(device)

## ----------------------------------------------     Segmentation stuff     ---------------------------------------------------------- ##
from utils.yolact import Yolact
from utils.output_utils import postprocess
from utils.config import cfg
import torch.backends.cudnn as cudnn

def calc_size_preserve_ar(img_w, img_h, max_size):
    ratio = sqrt(img_w / img_h)
    w = max_size * ratio
    h = max_size / ratio
    return int(w), int(h)

class FastBaseTransform(torch.nn.Module):

    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
          self.mean = torch.Tensor((103.94, 116.78, 123.68)).float().cuda()[None, :, None, None]
          self.std  = torch.Tensor( (57.38, 57.12, 58.40) ).float().cuda()[None, :, None, None]
    
        else:
          self.mean = torch.Tensor((103.94, 116.78, 123.68)).float()[None, :, None, None]
          self.std  = torch.Tensor( (57.38, 57.12, 58.40) ).float()[None, :, None, None]
    
        self.transform = cfg.backbone.transform
    
    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = (img_size[1], img_size[0]) # Pytorch needs h, w
        else:
            img_size = (cfg.max_size, cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        return img

def evalimage(net:Yolact, path:str):
    if torch.cuda.is_available():
      frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    else:
      frame = torch.from_numpy(cv2.imread(path)).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)
    h, w, _ = frame.shape
    t = postprocess(preds, w, h, score_threshold = 0.15)
    idx = t[1].argsort(0, descending=True)[:15]
    masks = t[3][idx]
    np.save('utils/mask_data', masks.cpu().numpy())

def evaluate(net:Yolact, inp):
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    cfg.mask_proto_debug = False
    evalimage(net, inp)
    return


def segments():
    global solutions
    with torch.no_grad():
      if torch.cuda.is_available():
          cudnn.fastest = True
          torch.set_default_tensor_type('torch.cuda.FloatTensor')
      else:
          torch.set_default_tensor_type('torch.FloatTensor')

      net = Yolact()
      net.load_weights("yolact_base_54_800000.pth")
      net.eval()

      if torch.cuda.is_available():
          net = net.cuda()

      evaluate(net, "content.jpg")

    mask = np.load('./utils/mask_data.npy')
    sizes = {}
    for x,i in enumerate(mask):
        sizes[x] = len((np.argwhere(i!=0)))
    sizes = {k: v for k, v in sorted(sizes.items(), key=lambda item: item[1], reverse=True)}

    solutions = {}
    for x,i in enumerate(list(sizes.keys())[:5]):
        solutions[x] = np.argwhere(mask[i] != 0)

    img = cv2.imread('content.jpg')[:, :, ::-1]

    segments = {}
    for i in range(len(solutions)):
        segments[i] = np.full((img.shape), 255.)

    for x,i in solutions.items():
        for j in i:
            segments[x][j[0]][j[1]] = img[j[0]][j[1]]

    if not os.path.exists('segments'):
        os.mkdir('segments')

    for i in range(len(segments)):
        im = Image.fromarray(segments[i].astype('uint8'))
        im.save(f'segments/{i}.jpg')
    
    return solutions

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)