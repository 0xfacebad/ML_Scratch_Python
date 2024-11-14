import numpy as np
import pandas as pd

# Gini Impurity function
def gini_impurity(y):
    """
    Calculate the Gini Impurity for a set of labels.
    """
    classes = np.unique(y)
    gini = 1.0
    for cls in classes:
        prob = np.sum(y == cls) / len(y)
        gini -= prob ** 2
    return gini

# Function to split the dataset on a feature at a given threshold
def split_dataset(X, y, feature_index, threshold):
    """
    Split the dataset into two groups based on a feature and threshold.
    """
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    
    X_left, X_right = X[left_mask], X[right_mask]
    y_left, y_right = y[left_mask], y[right_mask]
    
    return X_left, X_right, y_left, y_right

# Function to find the best split for a dataset
def best_split(X, y):
    """
    Find the best split for a dataset by evaluating all features and possible thresholds.
    """
    best_gini = float('inf')
    best_split = None
    best_left_y = None
    best_right_y = None
    best_feature_index = None
    best_threshold = None
    
    n_samples, n_features = X.shape
    
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])  # Evaluate all unique values as thresholds
        
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            # Calculate Gini impurity for the split
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
            
            if gini < best_gini:
                best_gini = gini
                best_split = (X_left, X_right, y_left, y_right)
                best_feature_index = feature_index
                best_threshold = threshold
                
    return best_feature_index, best_threshold, best_split

# Decision Tree Classifier (Recursive Tree Builder)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data (X, y).
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if n_samples <= 1 or n_classes == 1:
            return np.bincount(y).argmax()
        
        if self.max_depth is not None and depth >= self.max_depth:
            return np.bincount(y).argmax()
        
        # Find the best split
        feature_index, threshold, (X_left, X_right, y_left, y_right) = best_split(X, y)
        
        if feature_index is None:
            return np.bincount(y).argmax()
        
        # Build the tree recursively for the left and right splits
        left_node = self._build_tree(X_left, y_left, depth + 1)
        right_node = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': left_node,
            'right': right_node
        }

    def predict(self, X):
        """
        Predict the class labels for a given set of samples.
        """
        predictions = [self._predict_sample(sample, self.tree) for sample in X]
        return np.array(predictions)

    def _predict_sample(self, sample, node):
        """
        Recursively predict a sample based on the tree structure.
        """
        if isinstance(node, dict):  # If we are at a decision node
            if sample[node['feature_index']] <= node['threshold']:
                return self._predict_sample(sample, node['left'])
            else:
                return self._predict_sample(sample, node['right'])
        else:  # If we are at a leaf node
            return node
#Choosing the small data sets 
humidity_values = [6.1, 5.7, 4.6, 6, 4.4, 2.5, 2.6, 2.5, 4.9, 4.5, 6.5, 2.4, 2.5]
wind_values = [4.5, 4.2, 2.5, 4.6, 1.2, 1.1, 1.2, 1.4, 1.4, 4.8, 2.3, 3.4, 2.6]
playgolf_values = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

# Repeat values to make sure there are 400 values
humidity_extended = np.tile(humidity_values, 400 // len(humidity_values))[:400]
wind_extended = np.tile(wind_values, 400 // len(wind_values))[:400]
playgolf_extended = np.tile(playgolf_values, 400 // len(playgolf_values))[:400]
np.random.shuffle(humidity_extended)
np.random.shuffle(wind_extended)
np.random.shuffle(playgolf_extended)

# Create the DataFrame
data = pd.DataFrame({
    "humidity": humidity_extended,
    "wind": wind_extended,
    "PlayGolf": playgolf_extended
})
# Data preparation

X = data[['humidity', 'wind']].values  # Features as a NumPy array
y = data['PlayGolf'].values  # Target variable as a NumPy array

# Create and train the decision tree
tree = DecisionTree(max_depth=4)
tree.fit(X, y)

# Make predictions
y_pred = tree.predict(X)
val = y_pred.ravel()
data = pd.DataFrame({
    "humidity": humidity_extended,
    "wind": wind_extended,
    "PlayGolf": val
})
data["PlayGolf"] = data["PlayGolf"].map({1:'Yes' , 0:'No'})
print(data.sample(20))
# Evaluate model accuracy
accuracy = np.mean(y_pred == y)
print(f'Accuracy: {accuracy:.2f}')
