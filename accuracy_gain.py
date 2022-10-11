# calculates purity gain


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# testing based on human labels
# 0 neutral
# 1 happy
# 2 sad
# 3 surprise
# 4 fear
# 5 disgust
# 6 anger
# 7 contempt
# 8 anxiety
cluster_label = [1, 0, 1, 0, 1,
                 2, 5, 3, 0, 1,
                 1, 6, 1, 1, 5,
                 7, 8, 1, 8, 7,
                 4, 1, 0, 6, 5,
                 1, 5, 5, 3, 0]


# based on human labels
def get_labelled():
    centroids = np.load('centroids_30.npy')
    c = centroids[0]

    df = pd.read_csv('openface_val.csv')
    X = df[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
            ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
            ' AU45_r']]
    y = df[['emotion_label']]
    filename = df[['file']]

    X = X.to_numpy()
    filename = filename.to_numpy()

    labelled = []
    for i in range(0, X.shape[0]):
        mse = (np.square(X[i] - c)).mean(axis=1)
        centroid = np.argmin(mse)
        labelled.append([filename[i][0], cluster_label[centroid]])

    return labelled


# based on AffectNet labels
def get_labelled_true():
    centroids = np.load('centroids_30.npy')
    c = centroids[0]

    df = pd.read_csv('openface_val.csv')
    X = df[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
            ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
            ' AU45_r']]
    y = df[['emotion_label']]
    filename = df[['file']]

    X = X.to_numpy()
    y = y.to_numpy()
    print(y)
    print(y.shape)
    filename = filename.to_numpy()

    labelled = []
    for i in range(0, X.shape[0]):
        labelled.append([filename[i][0], y[i]])

    return labelled


def calc_purity(clusters):
    counts = 0
    for i in range(0, clusters.shape[0]):
        emotion = np.amax(clusters[i])
        counts += emotion
    return counts


def accuracy_gain():
    labelled = get_labelled()
    # labelled = get_labelled_true()
    n = len(labelled)

    prev_purity = 0
    purities = []
    purity_gains = []

    for cen in range(8, 51):
        centroids = np.load('centroids_' + str(cen) + '.npy')
        c = centroids[0]

        df = pd.read_csv('openface_val.csv')
        X = df[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
                ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
                ' AU45_r']]
        y = df[['emotion_label']]
        filename = df[['file']]

        X = X.to_numpy()

        centroid_emotions = np.zeros((cen, 9), dtype=int)
        # centroid_emotions = np.zeros((cen, 8), dtype=int)

        for i in range(0, X.shape[0]):
            mse = (np.square(X[i] - c)).mean(axis=1)
            centroid = np.argmin(mse)
            centroid_emotions[centroid][labelled[i][1]] += 1

        if cen == 8:
            prev_purity = calc_purity(centroid_emotions) / n
            purities.append(prev_purity)
        else:
            print('------------')
            print("Centroids: {}".format(cen))
            purity = calc_purity(centroid_emotions) / n
            purities.append(purity)
            purity_gains.append(purity - prev_purity)
            prev_purity = purity

    print(purities)
    print(purity_gains)

    plt.plot(range(8, 51), purities, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Purity')
    plt.savefig('purity.png')
    # plt.savefig('purity_true.png')

    plt.clf()

    plt.plot(range(9, 51), purity_gains, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Purity Gain')
    plt.savefig('purity_gain.png')
    # plt.savefig('purity_true_gain.png')


accuracy_gain()
