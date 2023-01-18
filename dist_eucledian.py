from scipy.spatial.distance import euclidean

c1 = [1.0, 1.5]
c2 = [-1.0, -0.5]
c3 = [1.0, -0.5]
dist = euclidean(c1, c3)
print("Расстояние между кластерами c1 и c3: %.4f" % dist)