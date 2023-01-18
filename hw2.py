import numpy as np
import pandas as pd

df = pd.read_csv('clustering_hw.csv')
X = df[['x1', 'x2']]
y = df['y']
print(X.head())

import matplotlib.pyplot as plt
plt.scatter(X.x1, X.x2, s=40, c=y, marker='o', alpha=0.8, label='data')
plt.scatter(5, 8, s=80, c='black', marker='o', alpha=0.8, label='data')
plt.scatter(0, 5, s=80, c='blue', marker='o', alpha=0.8, label='data')
plt.show()



from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

metrics = []
MAX_CLUSTERS = 7
for cluster_num in range(1, MAX_CLUSTERS):
    kmeans_model = KMeans(n_clusters=cluster_num, random_state=99).fit(X)
    centroids, labels = kmeans_model.cluster_centers_, kmeans_model.labels_
    metric = 0
    for centroid_label in range(cluster_num):
        metric += euclidean_distances(
            X[labels==centroid_label],
            centroids[centroid_label,:].reshape(1,-1)
        ).sum(axis=0)[0]
    print("cluster_num %s, metric %s" % (cluster_num, metric))
    metrics.append(metric)

D = []
for i in range(0, len(metrics)-1):
    d = abs(metrics[i+1]-metrics[i])/abs(metrics[i]-metrics[i-1])
    D.append(d)
print("best cluster num: %s" % (np.argmin(D)+1))

plt.plot([i+1 for i in range(len(metrics))], metrics)
plt.show()

from sklearn.metrics.cluster import adjusted_mutual_info_score

kmeans_model = KMeans(n_clusters=np.argmin(D)+1, random_state=42)
kmeans_model.fit(X)
X_test1 = [[5, 8]]
X_test2 = [[0, 5]]
y_test1 = kmeans_model.predict(X_test1)
y_test2 = kmeans_model.predict(X_test2)
print(y_test1)
print(y_test2)
score = adjusted_mutual_info_score(y, kmeans_model.labels_, average_method='arithmetic')
print("Качество кластеризации по метрике AMI %.3f" % score)

