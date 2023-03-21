import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = [0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print(knn.score(X, y))
print(knn.predict([[2, 2]]))
