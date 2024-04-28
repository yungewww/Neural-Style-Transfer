'''
Gram Matrix of a single feature map
'''
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

image_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),  # Resize the image to 28 * 28 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor 1 * 28 * 28
    transforms.Normalize(mean=0., std=1.)  # Normalize the image tensor
])

input_image = Image.open('content/cat.jpg') # add your image path
# input_image = Image.open('content/ytb.png') # add your image path
# input_image = Image.open('content/style.jpg') # add your image path
input_image = image_transform(input_image)
input_image = input_image.squeeze(0) # 28 * 28
print(input_image.shape)

plt.imshow(input_image)
plt.axis('off')
plt.show()


transpose = input_image.t()
plt.imshow(transpose)
plt.axis('off')
plt.show()

def gram_matrix(tensor):
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

gram = gram_matrix(input_image)

plt.imshow(gram)
plt.axis('off')
plt.show()