# preprocessing csv output from OpenFace
# NOTE: Highly suggested that you stitch together images into a single video
# before running it through OpenFace to get 1 csv file (vs. # images)
# e.g. use ffmpeg


import numpy as np
import shutil
import time
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageOps


def create_openface_csv():
    # path for train/validation
    # path = 'TODO TRAIN'
    path = 'TODO VAL'

    # emotion classes
    neutral = path + '0_neutral/'
    happy = path + '1_happy/'
    sad = path + '2_sad/'
    surprise = path + '3_surprise/'
    fear = path + '4_fear/'
    disgust = path + '5_disgust/'
    anger = path + '6_anger/'
    contempt = path + '7_contempt/'

    paths = [neutral, happy, sad, surprise, fear, disgust, anger, contempt]
    emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

    df = pd.read_csv(neutral + '1.csv')
    df['file'] = 1
    df['emotion_label'] = 0

    for p in range(0, len(paths)):
        print(emotions[p])
        files = [f for f in listdir(paths[p]) if isfile(join(paths[p], f))]
        files = [f for f in files if f.split('.')[1] == 'csv']

        for file in range(0, len(files)):
            if file != '1.csv':
                temp_df = pd.read_csv(paths[p] + files[file])
                temp_df['file'] = int(files[file].split('.')[0])
                temp_df['emotion_label'] = p
                df = df.append(temp_df)
            if file % 1000 == 0:
                print('Completed {}'.format(file))

    df = df.sort_values('file', ascending=True)
    df = df.reset_index(drop=True)
    df = df.drop(columns='face')

    # saving for train vs. val dataset
    # df.to_csv('openface.csv', encoding='utf-8', index=False)
    df.to_csv('openface_val.csv', encoding='utf-8', index=False)


create_openface_csv()
