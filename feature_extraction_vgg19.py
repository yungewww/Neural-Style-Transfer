import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


'''
LOAD & PREPROCESS IMAGE
'''
# Define the image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=0., std=1.)  # Normalize the image tensor
])
# Load the image
# input_image = Image.open('content/cat.jpg') # add your image path
input_image = Image.open('content/style.jpg')
plt.imshow(input_image)
plt.show()

# Preprocess the image and move it to GPU
input_image = image_transform(input_image)
input_image = input_image.unsqueeze(0)  # Add a batch dimension

'''
LOAD & EXTRACT MODEL CONV LAYERS
'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# model = models.vgg16(weights='imagenet')
model = models.vgg19(pretrained=True)
print(model)

# Extract convolutional layers and their weights
conv_weights = []  # List to store convolutional layer weights
conv_layers = []  # List to store convolutional layers
total_conv_layers = 0  # Counter for total convolutional layers

# Traverse through the model to extract convolutional layers and their weights
for module in model.features.children():
    if isinstance(module, nn.Conv2d):
        total_conv_layers += 1
        conv_weights.append(module.weight)
        conv_layers.append(module)

print(f"Total convolution layers: {total_conv_layers}")

'''
GET FEATURE MAPS
'''
# Extract feature maps
feature_maps = []  # List to store feature maps
layer_names = []  # List to store layer names
for layer in conv_layers:
    input_image = layer(input_image)
    feature_maps.append(input_image)
    layer_names.append(str(layer))

# Display feature maps shapes
print("\nFeature maps shape")
for feature_map in feature_maps:
	print(feature_map.shape)

# Process and visualize feature maps
processed_feature_maps = [] # List to store processed feature maps
for feature_map in feature_maps:
	feature_map = feature_map.squeeze(0) # Remove the batch dimension
	mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0] # ⭕️ Compute mean across channels
	processed_feature_maps.append(mean_feature_map.data.cpu().numpy())


# Display processed feature maps shapes
print("\n Processed feature maps shape")
for fm in processed_feature_maps:
	print(fm.shape)

# Plot the feature maps
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed_feature_maps)):
	ax = fig.add_subplot(5, 4, i + 1)
	ax.imshow(processed_feature_maps[i])
	ax.axis("off")
	ax.set_title(layer_names[i].split('(')[0], fontsize=30)
plt.show()
