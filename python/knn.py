"""
k-Nearest Neighbors (kNN) from scratch in Python 3
Beginner-friendly implementation using only standard library
No external ML libraries (no numpy/pandas), just lists, loops, and csv module
"""

import csv
import math


# ============================================================================
# Function: read_csv_all_but_last
# Reads CSV file, treating all columns except the last as features
# and the last column as the label.
# Returns: (features, labels) where features is a list of lists
# ============================================================================
def read_csv_all_but_last(filename):
    """
    Read CSV file and split into features and labels.
    
    Args:
        filename: Path to CSV file
    
    Returns:
        tuple: (features, labels) where:
            - features is a list of lists (each inner list is a data point)
            - labels is a list of integers
    """
    features = []
    labels = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:  # Skip empty rows
                # Convert all columns to float except the last one
                feature_row = [float(x) for x in row[:-1]]
                label = int(row[-1])
                
                features.append(feature_row)
                labels.append(label)
    
    return features, labels


# ============================================================================
# Function: distance_sq
# Computes squared Euclidean distance between two feature vectors
# ============================================================================
def distance_sq(features1, features2):
    """
    Calculate squared Euclidean distance between two points.
    
    Args:
        features1: List of feature values for first point
        features2: List of feature values for second point
    
    Returns:
        float: Squared distance
    """
    total = 0.0
    for i in range(len(features1)):
        diff = features1[i] - features2[i]
        total += diff * diff
    return total


# ============================================================================
# Function: predict_knn
# Predicts the label for a test point using k-nearest neighbors
# Returns the most common label among the k nearest neighbors
# ============================================================================
def predict_knn(test_features, train_features, train_labels, k):
    """
    Predict label for a test point using k-nearest neighbors.
    
    Args:
        test_features: List of feature values for test point
        train_features: List of lists containing training features
        train_labels: List of training labels
        k: Number of nearest neighbors to consider
    
    Returns:
        int: Predicted label
    """
    # Calculate distances from test point to all training points
    distances = []
    for i in range(len(train_features)):
        dist = distance_sq(test_features, train_features[i])
        distances.append((dist, i))
    
    # Sort by distance (beginner-friendly sorting)
    distances.sort(key=lambda x: x[0])
    
    # Get the k nearest neighbors
    k_nearest = distances[:k]
    
    # Count votes from k nearest neighbors
    votes = {}
    for _, idx in k_nearest:
        label = train_labels[idx]
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    
    # Find label with most votes
    predicted_label = max(votes, key=votes.get)
    
    return predicted_label


# ============================================================================
# Function: accuracy
# Calculates classification accuracy on a test dataset
# ============================================================================
def accuracy(test_features, test_labels, train_features, train_labels, k):
    """
    Calculate classification accuracy on test data.
    
    Args:
        test_features: List of lists containing test features
        test_labels: List of test labels
        train_features: List of lists containing training features
        train_labels: List of training labels
        k: Number of nearest neighbors to consider
    
    Returns:
        float: Accuracy as a value between 0 and 1
    """
    correct = 0
    
    for i in range(len(test_features)):
        predicted = predict_knn(test_features[i], train_features, train_labels, k)
        if predicted == test_labels[i]:
            correct += 1
    
    return correct / len(test_features)


# ============================================================================
# Optional Normalization Functions
# ============================================================================

def zstats(features):
    """
    Calculate mean and standard deviation for each feature.
    
    Args:
        features: List of lists containing feature values
    
    Returns:
        tuple: (means, std_devs) where each is a list
    """
    num_features = len(features[0])
    num_samples = len(features)
    
    # Calculate means
    means = [0.0] * num_features
    for feature_row in features:
        for i in range(num_features):
            means[i] += feature_row[i]
    
    for i in range(num_features):
        means[i] /= num_samples
    
    # Calculate standard deviations
    std_devs = [0.0] * num_features
    for feature_row in features:
        for i in range(num_features):
            diff = feature_row[i] - means[i]
            std_devs[i] += diff * diff
    
    for i in range(num_features):
        std_devs[i] = math.sqrt(std_devs[i] / num_samples)
        # Avoid division by zero
        if std_devs[i] < 1e-10:
            std_devs[i] = 1.0
    
    return means, std_devs


def apply_norm(features, means, std_devs):
    """
    Apply z-score normalization to features (in-place).
    
    Args:
        features: List of lists containing feature values (modified in-place)
        means: List of mean values for each feature
        std_devs: List of standard deviation values for each feature
    """
    for feature_row in features:
        for i in range(len(feature_row)):
            feature_row[i] = (feature_row[i] - means[i]) / std_devs[i]


# ============================================================================
# Main Function - Example Usage
# ============================================================================

def main():
    print("=== k-Nearest Neighbors (kNN) from Scratch in Python ===")
    print()
    
    # Load dataset
    filename = "../data/iris.csv"
    
    print(f"Loading dataset from: {filename}")
    features, labels = read_csv_all_but_last(filename)
    print(f"Loaded {len(features)} data points with {len(features[0])} features each.")
    print()
    
    # Split into train (first 24) and test (last 6)
    train_features = features[:24]
    train_labels = labels[:24]
    test_features = features[24:]
    test_labels = labels[24:]
    
    print(f"Split: {len(train_features)} training, {len(test_features)} test points")
    print()
    
    # Test without normalization
    print("--- Testing WITHOUT normalization ---")
    k = 3
    acc = accuracy(test_features, test_labels, train_features, train_labels, k)
    print(f"Accuracy with k={k}: {acc * 100:.1f}%")
    print()
    
    # Test with normalization
    print("--- Testing WITH normalization ---")
    
    # Calculate stats on training data
    means, std_devs = zstats(train_features)
    
    # Apply normalization (need to make copies to preserve original data)
    train_features_norm = [row[:] for row in train_features]  # Deep copy
    test_features_norm = [row[:] for row in test_features]    # Deep copy
    
    apply_norm(train_features_norm, means, std_devs)
    apply_norm(test_features_norm, means, std_devs)
    
    acc = accuracy(test_features_norm, test_labels, train_features_norm, train_labels, k)
    print(f"Accuracy with k={k} (normalized): {acc * 100:.1f}%")
    print()
    
    # Example prediction on a single point
    print("--- Example Single Prediction ---")
    print(f"Test point features: {test_features_norm[0]}")
    
    pred = predict_knn(test_features_norm[0], train_features_norm, train_labels, k)
    print(f"Predicted label: {pred}")
    print(f"Actual label: {test_labels[0]}")
    print()
    
    print("Done!")


if __name__ == "__main__":
    main()
