import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import cv2

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

# convert tis image to numpy array
img_resized = np.array(img_resized)

# transpose from shape of (3,,) to shape of (,,3)
img_resized = img_resized.transpose(1, 2, 0)

print(img_resized.shape)

# display the normalized image
plt.imshow(img_resized)
plt.xticks([])
plt.yticks([])
plt.show()


