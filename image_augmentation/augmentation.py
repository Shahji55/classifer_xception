import cv2
import numpy as np
from PIL import Image
import os


# Rotate
def rotate(image):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, .5)
    rotated_img = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imwrite(cwd + '/Rotated_image.jpg', rotated_img)


# Flip
def flip(image):
    # Flip around y-axis (horizontal)
    flipped_img = cv2.flip(image, 1)
    cv2.imwrite(cwd + '/Flipped_image.jpg', flipped_img)


# Calculate average images size
def avg_size_images():
    emp_widths = []
    per_widths = []
    emp_heights = []
    per_heights = []

    employee_dir = input_dir + 'Employee/'
    person_dir = input_dir + 'Person/'

    # Employee
    for emp_img in os.listdir(employee_dir):
        img_path = os.path.join(employee_dir, emp_img)
        im = Image.open(img_path)

        # print(im.size[0], im.size[1])
        emp_widths.append(im.size[0])
        emp_heights.append(im.size[1])

    # Average size of employee images
    avg_width_emp = round(sum(emp_widths)/len(emp_widths))
    avg_height_emp = round(sum(emp_heights)/len(emp_heights))
    print("Employee images avg size: ", avg_width_emp, avg_height_emp)

    # Person
    for per_img in os.listdir(person_dir):
        img_path = os.path.join(person_dir, per_img)
        im = Image.open(img_path)
        
        per_widths.append(im.size[0])
        per_heights.append(im.size[1])

    # Average size of person images
    avg_width_per = round(sum(per_widths)/len(per_widths))
    avg_height_per = round(sum(per_heights)/len(per_heights))
    print("Person images avg size: ", avg_width_per, avg_height_per)


if __name__ == "__main__":
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    print(parent_dir)

    input_dir = parent_dir + '/data/'
    print(input_dir)

    input_img = cwd + '/image.jpg'
    img = cv2.imread(input_img)

    # rotate(img)
    # flip(img)

    avg_size_images()

'''
cv2.imshow('Image', rotated_img)
cv2.waitKey(0)
cv2.imshow('Image', flipped_img)
cv2.destroyAllWindows()
'''



