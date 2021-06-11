import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchvision

cwd = os.getcwd()
data_dir = cwd + '/data'

print(data_dir)

transform = transforms.Compose([
    transforms.Resize((150, 50)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    # transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(data_dir, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))

# Display image and label.
print(f"Feature batch shape: {images.size()}")
print(f"Labels batch shape: {labels.size()}")
img = images[0].squeeze()
print(img)
img = np.array(img)
img = img.transpose(1, 2, 0)
label = labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Split the dataset into train and test set
train_set, test_set = torch.utils.data.random_split(dataset, [7658, 1914])
print(len(train_set))
print(len(test_set))
print(train_set.indices)
print(test_set.indices)


batch_size = 4
# Dataloader for training set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Dataloader for test set
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Specify the classes of dataset
classes = ('Employee', 'Person')


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


