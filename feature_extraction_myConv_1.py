import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # input image size = 1 * 28 * 28,
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output 16 feature maps of 14px * 14px * 1channel
        )

    def forward(self, x):
        output = self.conv1(x)
        return output

model = FeatureExtractor()

# Define the image transformations
image_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),  # Resize the image to 28 * 28 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=0., std=1.)  # Normalize the image tensor
])

# Load the image
input_image = Image.open('content/cat.jpg') # add your image path
# plt.imshow(input_image)
# plt.show()
input_image = image_transform(input_image)
input_image = input_image.unsqueeze(0)  # Add a batch dimension

# Pass the image through the model to get the feature map
with torch.no_grad():
    feature_map = model(input_image)

print("Shape of feature map:", feature_map.shape) # Print the shape of the feature map
feature_map = feature_map[:, :16, :, :] # Get the 16 feature maps
feature_map = feature_map.squeeze().numpy() # Convert to numpy array and reshape

# Display the feature maps
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_map[i], cmap='gray')
    ax.axis('off')
plt.show()