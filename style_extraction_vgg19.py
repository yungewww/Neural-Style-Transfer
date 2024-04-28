from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

'''
LOAD IMAGE & MODEL
'''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

image_transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((400, 400)),  # Resize the image to 28 * 28 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor 1 * 28 * 28
    transforms.Normalize(mean=0., std=1.)  # Normalize the image tensor
])

style = Image.open("content/style.jpg")
style = image_transform(style).unsqueeze(0)
print(style.shape)

'''
GET FEATURE MAPS FROM MODEL
'''
def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    if layers is None:
        layers = {'0': 'conv1_1', # 64
                  '5': 'conv2_1', # 128
                  '10': 'conv3_1', # 256
                  '19': 'conv4_1', # 512
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'} # 512

    features = {}
    x = image

    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

style_features = get_features(style, vgg)

'''
VISUALIZE EXTRACTED STYLE FEATURE MAPS
'''
def show_all_feature_map(feature_map):
    for _, feature in feature_map.items():
        print(feature.shape)

        # 创建一个 8x8 的子图布局，共 64 个子图
        fig, axes = plt.subplots(8, 8, figsize=(16, 16))

        # 遍历所有特征图并显示在子图中
        for i in range(8):
            for j in range(8):
                feature_map = feature[0, i * 8 + j].cpu().detach().numpy()
                axes[i, j].imshow(feature_map, cmap='gray')
                axes[i, j].axis('off')  # 关闭坐标轴

        plt.tight_layout()  # 自动调整子图布局
        plt.show()


def show_first_feature_map(feature_map):
    target = feature_map['conv1_1']
    first_feature_map = target[0, 0].cpu().detach().numpy()

    plt.imshow(first_feature_map)
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# show_first_feature_map(style_features)
# show_all_feature_map(style_features)

'''
VISUALIZE FEATURE MAPS AFTER GRAM
'''
def gram_matrix(tensor):
    _, depth, height, width = tensor.size()

    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# gram = style_grams['conv1_1']

def show_all_gram(gram_map):
    for _, gram in gram_map.items():
        plt.imshow(gram)
        plt.axis('off')
        plt.show()

show_all_gram(style_grams)


# show_first_feature_map(style_grams)


# style_weights = {'conv1_1': 1.,
#                  'conv2_1': 0.8,
#                  'conv3_1': 0.5,
#                  'conv4_1': 0.3,
#                  'conv5_1': 0.1}

