# preprocess AffectNet dataset (already unzipped)


import numpy as np
import shutil
import time
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageOps


# preprocess unzipped AffectNet (train)
def sort_data_train(path, new_path):
    files = [f for f in listdir(path + '/images/') if isfile(join(path + '/images/', f))]

    init_time = time.time()

    for i in range(0, len(files)):
        file = files[i]
        file = file[:-4]

        emotion = np.load(path + '/annotations/' + file + '_exp.npy')
        emotion = np.int(emotion)

        try:
            if emotion == 0:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/0_neutral/' + file + '.jpg')
            if emotion == 1:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/1_happy/' + file + '.jpg')
            if emotion == 2:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/2_sad/' + file + '.jpg')
            if emotion == 3:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/3_surprise/' + file + '.jpg')
            if emotion == 4:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/4_fear/' + file + '.jpg')
            if emotion == 5:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/5_disgust/' + file + '.jpg')
            if emotion == 6:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/6_anger/' + file + '.jpg')
            if emotion == 7:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/train/7_contempt/' + file + '.jpg')
        except shutil.SameFileError:
            pass

        if i % 1000 == 0 or i == len(files) - 1:
            print('Sorted: {}'.format(i))
            print('Time elapsed: {}'.format(time.time() - init_time))

    print('Done sorting training data.')


# preprocess unzipped AffectNet (validation)
def sort_data_val(path, new_path):
    files = [f for f in listdir(path + '/images/') if isfile(join(path + '/images/', f))]

    init_time = time.time()

    for i in range(0, len(files)):
        file = files[i]
        file = file[:-4]

        emotion = np.load(path + '/annotations/' + file + '_exp.npy')
        emotion = np.int(emotion)

        try:
            if emotion == 0:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/0_neutral/' + file + '.jpg')
            if emotion == 1:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/1_happy/' + file + '.jpg')
            if emotion == 2:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/2_sad/' + file + '.jpg')
            if emotion == 3:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/3_surprise/' + file + '.jpg')
            if emotion == 4:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/4_fear/' + file + '.jpg')
            if emotion == 5:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/5_disgust/' + file + '.jpg')
            if emotion == 6:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/6_anger/' + file + '.jpg')
            if emotion == 7:
                shutil.copy(path + '/images/' + file + '.jpg', new_path + '/val/7_contempt/' + file + '.jpg')
        except shutil.SameFileError:
            pass

        if i % 1000 == 0 or i == len(files) - 1:
            print('Sorted: {}'.format(i))
            print('Time elapsed: {}'.format(time.time() - init_time))
            print('------------')

    print('Done sorting validation data.')


# boost data for multiclass
def boost_data(path, emotions):
    for e in range(0, len(emotions)):
        init_time = time.time()

        emotion = emotions[e]

        files = [f for f in listdir(path + '/train/' + emotion) if isfile(join(path + '/train/' + emotion, f))]

        for i in range(0, len(files)):
            file = files[i]
            file = file[:-4]

            im = Image.open(path + '/train/' + emotion + '/' + file + '.jpg')
            mirror = ImageOps.mirror(im)
            mirror.save(path + '/train/' + emotion + '/' + file + '_mirror' + '.jpg', quality=100)

            if i % 250 == 0 or i == len(files) - 1:
                print('Sorted: {}'.format(i))
                print('Time elapsed: {}'.format(time.time() - init_time))
                print('------------')

    print('Done boosting data.')


# boost data for binary
def boost_binary_data(path):
    files = [f for f in listdir(path + '/train/true') if isfile(join(path + '/train/true', f))]

    init_time = time.time()

    for i in range(0, len(files)):
        file = files[i]
        file = file[:-4]

        im = Image.open(path + '/train/true/' + file + '.jpg')
        mirror = ImageOps.mirror(im)
        mirror.save(path + '/train/true/' + file + '_mirror' + '.jpg', quality=100)

        if i % 250 == 0 or i == len(files) - 1:
            print('Sorted: {}'.format(i))
            print('Time elapsed: {}'.format(time.time() - init_time))
            print('------------')

    print('Done boosting data.')
