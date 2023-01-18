import pickle
import numpy as np

# данные получены с помощью функции make_classification
with open('clustering.pkl', 'rb') as f:
    data_clustering = pickle.load(f)

X = np.array(data_clustering['X'])
Y = np.array(data_clustering['Y'])

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X)
plt.scatter(X[:, 0], X[:, 1], s=40, c=kmeans_model.labels_, marker='o', alpha=0.8, label='data')
plt.show()

from sklearn.metrics import silhouette_score
score = silhouette_score(X, kmeans_model.labels_)

print("Качество кластеризации по метрике силуэтта %.3f" % score)

model = KMeans(n_clusters=3).fit(X)
score = silhouette_score(X, model.labels_)
print("Качество кластеризации по метрике силуэтта  для трёх кластеров %.3f" % score)
plt.scatter(X[:, 0], X[:, 1], s=40, c=model.labels_, marker='o', alpha=0.8)
plt.show()

from sklearn.metrics.cluster import adjusted_mutual_info_score

model = KMeans(n_clusters=3).fit(X)
score = adjusted_mutual_info_score(Y, model.labels_, average_method='arithmetic')

print("Качество кластеризации по метрике AMI %.3f" % score)