import numpy as np
import random
import os
import cv2
import time


def random_subsample(image, subsample_width, subsample_height):
    
    image_height, image_width = image.shape[:2] 
    assert image_height >= subsample_height
    assert image_width >= subsample_width

    left = random.randint(0, image_width - subsample_width)
    top = random.randint(0, image_height - subsample_height)

    subsampled_image = image[top:top+subsample_height, left:left+subsample_width, :]
    return subsampled_image

train_image_path = 'dataset/val/'
train_images_names = [os.path.join(train_image_path, image_file_names) for image_file_names in os.listdir(train_image_path)]
subsample_image_save_path = 'dataset/val_augmented/'
start = time.time()
print("===> Sub-image data augmentation start")

for image_name in train_images_names:
    image_num = 0
    while image_num < 10:
        image = cv2.imread(image_name)
        subsample = random_subsample(image, 120, 120)
        subsample_name = os.path.basename(os.path.splitext(image_name)[0]) + '-' + str(image_num) + '.png'
        cv2.imwrite(subsample_image_save_path + subsample_name, subsample)
        image_num += 1

end = time.time()
print("===> Data augmentation finished. Time spent: {:.2f}s".format(end-start))
