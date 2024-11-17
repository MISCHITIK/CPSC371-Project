def euclidean_distance(point1, point2):
        distance = 0
        valid_features = 0  # Counter for valid features

        for i in range(len(point1)):
            # Only calculate distance for features that are not missing in both points
            if point1[i] != '?' and point2[i] != '?':
                try:
                    distance += (float(point1[i]) - float(point2[i])) ** 2
                    valid_features += 1  # Count valid comparisons
                    print('valid')
                except ValueError:
                    print('invalid')
                    # Skip the feature if it's not numeric and can't be converted
                    continue
        return distance ** 0.5

class KNN:
    def __init__(self, k=20):
        self.k = k
    
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
    
    def _find_neighbours(self, test_point):
        distances = []
        # Compute distances from the test point to all training points
        for i in range(len(self.train_x)):
            dist = euclidean_distance(test_point, self.train_x[i])
            print(dist)
            distances.append((dist, i))
        # Sort by distance and select the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = [self.train_y[i] for _, i in distances[:self.k]]
        return neighbors
    
    def predict(self, test_x):
        predictions = []
        for test_point in test_x:
            neighbors = self._find_neighbours(test_point)
            most_common_label = max(set(neighbors), key=neighbors.count)  # Majority vote
            predictions.append(most_common_label)
        return predictions