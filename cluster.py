# K-means clustering


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster():
    init = 'random'
    n_init = 10
    max_iter = 2500
    tol = 0.0001
    verbose = 10

    df = pd.read_csv('openface.csv')
    df_val = pd.read_csv('openface_val.csv')

    X = df[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
            ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
            ' AU45_r']]
    X_val = df_val[[' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
                    ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r',
                    ' AU45_r']]
    y = df[['emotion_label']]
    y_val = df_val[['emotion_label']]

    X = X.to_numpy()
    y = y.to_numpy()
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    distortions = []

    # looping for K=8 to K=50
    for i in range(8, 51):
        labels = []
        centroids = []

        distortion = 0

        # looping n_init times for each K (mitigate randomization)
        for j in range(0, n_init):
            kmeans = KMeans(n_clusters=i, init=init, max_iter=max_iter, tol=tol)
            kmeans.fit(X)

            distortion += kmeans.inertia_

            centroid = kmeans.cluster_centers_
            label = kmeans.predict(X)
            centroids.append(centroid)
            labels.append(label)

            print('Classes: {}  Run: {}  Distortion: {}'.format(i, j, kmeans.inertia_))

        np.save('centroids_' + str(i) + '.npy', np.array(centroids))
        distortions.append(distortion / n_init)

    plt.plot(range(8, 51), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('clustering.png')


cluster()
