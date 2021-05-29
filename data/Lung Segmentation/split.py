import os
import glob
import shutil

train_path = '.\gtrain'
test_path = ".\gtest"

filename_test = os.listdir(test_path)
for file in filename_test:
    os.remove(os.path.join(train_path,file))    