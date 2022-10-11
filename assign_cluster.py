# assign an image's AUs to a cluster (already trained)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


def assign():
    centroids = np.load('centroids_36.npy')
    c = centroids[0]

    path = 'TODO'
    new_path = 'TODO'

    df = pd.read_csv('openface_val.csv')
    X = df[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
            ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
            ' AU45_r']]
    y = df[['emotion_label']]
    filename = df[['file']]

    X = X.to_numpy()
    y = y.to_numpy()
    filename = filename.to_numpy()

    centroid_emotions = np.zeros((36, 8))

    for i in range(0, X.shape[0]):
        mse = (np.square(X[i] - c)).mean(axis=1)
        centroid = np.argmin(mse)
        shutil.copy(path + str(filename[i][0]) + '.jpg', new_path + 'centroid (' + str(centroid + 1) + ')/' + str(filename[i][0]) + '.jpg')
        centroid_emotions[centroid][y[i][0]] += 1

    print(centroid_emotions)

    # plt.figure(figsize=(10, 20))
    # sns.heatmap(centroid_emotions, annot=True, linewidths=.5)
    # plt.savefig('cluster_15_1.png')


assign()
