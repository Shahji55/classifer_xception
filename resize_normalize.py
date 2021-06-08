import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

cwd = os.getcwd()
data_dir = cwd + '/data'

print(data_dir)

employee_data_dir = data_dir + '/Employee/'
person_data_dir = data_dir + '/Person/'

# load the image
img_path = employee_data_dir + 'srlpackages_ch5_20210505150807_2021050518032812.jpg'
img = Image.open(img_path)
print(img.size)

transform_resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((150, 50))
])

# get resize image
img_resized = transform_resize(img)

# calculate mean and std
mean, std = img_resized.mean([1, 2]), img_resized.std([1, 2])

# convert tis image to numpy array
img_resized = np.array(img_resized)

# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)

# transpose from shape of (3,,) to shape of (,,3)
img_resized = img_resized.transpose(1, 2, 0)

print("Resized image shape: ", img_resized.shape)

# display the normalized image
plt.imshow(img_resized)
plt.xticks([])
plt.yticks([])
plt.show()

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

image_normalized = transform_norm(img_resized)

# calculate mean and std
mean, std = image_normalized.mean([1, 2]), image_normalized.std([1, 2])

# print mean and std
print("Mean and Std of normalized image:")
print("Mean of the image:", mean)
print("Std of the image:", std)

image_normalized = np.array(image_normalized)

# transpose from shape of (3,,) to shape of (,,3)
image_normalized = image_normalized.transpose(1, 2, 0)

print("Normalized image shape: ", image_normalized.shape)

plt.imshow(image_normalized)
plt.xticks([])
plt.yticks([])
plt.show()






