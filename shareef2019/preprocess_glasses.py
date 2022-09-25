import os
import shutil

data_root = r'data\eyeglasses\eyeglasses'

processed_data_dir = r'data\eyeglasses_processed\eyeglasses'


if not os.path.isdir(processed_data_dir):
    os.makedirs(processed_data_dir)

image_names = []
with open(r'data\eyeglasses_training.txt') as f:
    image_names = f.read().split('\n')

for image_name in image_names:
    shutil.copyfile(os.path.join(data_root, image_name), os.path.join(processed_data_dir, image_name))

shutil.copyfile(r'data\eyeglasses\._eyeglasses', r'data\eyeglasses_processed\._eyeglasses')