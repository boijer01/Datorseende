import os
import shutil
import random

dataset_dir = "./train"
train_dir = "./split/train"
val_dir = "./split/validation"
test_dir = "./split/test"

#ratios
train_ratio = 0.7  
val_ratio = 0.15  
test_ratio = 0.15  

#skapar folders
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

#shufflar igenom bilderna
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    
    if os.path.isdir(category_path):
        images = os.listdir(category_path)
        random.shuffle(images) 

        train_split = int(len(images) * train_ratio)
        val_split = train_split + int(len(images) * val_ratio)

        #flyttar dom tills foldersen
        for i, img in enumerate(images):
            src_path = os.path.join(category_path, img)

            if i < train_split:
                dst_folder = os.path.join(train_dir, category)
            elif i < val_split:
                dst_folder = os.path.join(val_dir, category)
            else:
                dst_folder = os.path.join(test_dir, category)

            os.makedirs(dst_folder, exist_ok=True)
            shutil.move(src_path, os.path.join(dst_folder, img))

print("DOOOOOONEEEEEEE!! :)")
