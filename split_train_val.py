import os
import shutil
from pathlib import Path

with open("data/DAVIS/DAVIS_train/ImageSets/2017/train.txt") as f:
    for line in f.readlines():
        line = line.strip()
        print(line)
        file_jpg = Path("data/DAVIS/DAVIS_train/JPEGImages/480p/" + line)
        file_mp4 = Path("data/DAVIS/DAVIS_train/mp4/" + line + ".mp4")
        if file_jpg.is_dir() and file_mp4.is_file():
            shutil.move(file_jpg, "data/DAVIS/DAVIS_train/JPEGImages/480p/train_sequences/" + line)
            shutil.move(file_mp4, "data/DAVIS/DAVIS_train/mp4/train_mp4/" + line + ".mp4")
    
