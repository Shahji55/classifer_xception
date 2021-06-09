import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def mean_std_single_image():
    # load the image
    img_path = employee_data_dir + 'srlpackages_ch5_20210505150807_2021050518032812.jpg'
    img = Image.open(img_path)
    print(img.size)

    # convert PIL image to numpy array
    img_np = np.array(img)

    '''
    # plot the pixel values
    # pixel values of RGB image range from 0-255
    plt.hist(img_np.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.show()
    '''

    # define custom transform function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # transform the pIL image to tensor
    # image
    img_tr = transform(img)

    # Convert tensor image to numpy array
    img_np = np.array(img_tr)

    '''
    # plot the pixel values
    # pixel values of tensor range from 0-1
    plt.hist(img_np.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.show()
    '''

    # calculate mean and std
    mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])

    # print mean and std
    print("mean and std before normalize:")
    print("Mean of the image:", mean)
    print("Std of the image:", std)

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # get normalized image
    img_normalized = transform_norm(img)

    # convert normalized image to numpy
    # array
    img_np = np.array(img_normalized)

    '''
    # plot the pixel values
    plt.hist(img_np.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    # plt.show()
    '''

    # convert tis image to numpy array
    img_normalized = np.array(img_normalized)

    # transpose from shape of (3,,) to shape of (,,3)
    img_normalized = img_normalized.transpose(1, 2, 0)

    # display the normalized image
    plt.imshow(img_normalized)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # get normalized image
    img_nor = transform_norm(img)

    # calculate mean and std
    mean, std = img_nor.mean([1, 2]), img_nor.std([1, 2])

    # print mean and std
    print("Mean and Std of normalized image:")
    print("Mean of the image:", mean)
    print("Std of the image:", std)


if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = cwd + '/data'

    print(data_dir)

    employee_data_dir = data_dir + '/Employee/'
    person_data_dir = data_dir + '/Person/'

    mean_std_single_image()
