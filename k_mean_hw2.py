import pickle
import numpy as np

# данные получены с помощью функции make_classification
with open('clustering.pkl', 'rb') as f:
    data_clustering = pickle.load(f)

X = np.array(data_clustering['X'])
Y = np.array(data_clustering['Y'])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

c = []
for i in range(11):
    kmeans_model = KMeans(n_clusters=2, n_init=1, random_state=None, algorithm='full', max_iter=2)
    kmeans_model.fit(X)
    c.append(kmeans_model.cluster_centers_)
    plt.scatter(c[i][:, 0], c[i][:, 1], s=80, c = 'black', marker='o', alpha=0.8, label='data1')
print(c[0])
print(c[0][:, 0])
plt.scatter(X[:, 0], X[:, 1], s=40, marker='o', alpha=0.8, label='data')
plt.show()