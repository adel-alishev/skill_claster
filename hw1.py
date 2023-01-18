import pickle
import numpy as np

# данные получены с помощью функции make_classification
with open('clustering.pkl', 'rb') as f:
    data_clustering = pickle.load(f)

X = np.array(data_clustering['X'])
Y = np.array(data_clustering['Y'])

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score

for i in range(2,5,1):
    kmeans_model = KMeans(n_clusters=i, random_state=42)
    kmeans_model.fit(X)
    score = silhouette_score(X, kmeans_model.labels_)
    print("Качество кластеризации по метрике силуэтта %.3f" % score)
    model = KMeans(n_clusters=i).fit(X)
    score = adjusted_mutual_info_score(Y, model.labels_, average_method='arithmetic')
    print("Качество кластеризации по метрике AMI %.3f" % score)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=kmeans_model.labels_, marker='o', alpha=0.8, label='data')
    plt.show()


