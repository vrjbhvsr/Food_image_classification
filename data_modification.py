import json
import shutil
import os
import tqdm
from pathlib import Path
import random
from Food_Classification import logger

class data_modification:
    def __init__(self,path):
        self.path = path

    def get_labels(self,path:Path):
        with open(path) as f:
            return json.load(f)

## There are 101 classes and each class has 750 train images and 250 test images in the data. let's create dataset with 20 classes.


    def copy_images(self,source,file_name, destination, dataset, classes):
        ''' This function will take the selected dataset of selected class names and copy the data to the destination.'''
        labels = self.get_labels(self.path + dataset + '.json')

        for i in classes:
            os.makedirs(file_name + '/' + destination + '/' + dataset + '/' + i,exist_ok=True)

            for j in labels[i]:
                original_path = source + '/food-101/images/' + j + '.jpg'
                new_path = file_name + '/' + destination + '/' + dataset + '/' + j + '.jpg'
                shutil.copy(original_path, new_path)

    def get_percent_images(self,target_dir, new_dir, sample_amount = 0.2, random_seed = 42):
        """get sample amount of images from target_dir and copy to new_dir.

        Args:
            target_dir (str): file path to the target directory to extract images from.
            new_dir (str): path to the new directory to copy the images to.
            sample_amount (float): percentage of images to copy from target_dir. Defaults to 20.
            random_seed (int, optional): Defaults to 42.
        """

        #set random seed for reproducibility
        random.seed(random_seed)

        images = [{dirname: os.listdir(target_dir + dirname)} for dirname in os.listdir(target_dir)]
        images_moved=[]
        logger.info(f"Copying {sample_number} images from {x} to {new_dir}")
        for i in images:
            for x,y in i.items():
                sample_number = round(int(len(y) * sample_amount))
                
                random_images = random.sample(y,sample_number)

                #make new directory for each class
                new_target_dir = new_dir + x
                os.makedirs(new_target_dir, exist_ok=True)

                #copy images to new directory
                for images in random_images:
                    from_dir = target_dir + x + "/" + images
                    to_dir = new_target_dir + "/" + images

                    shutil.copy2(from_dir, to_dir)
                    images_moved.append(to_dir)
