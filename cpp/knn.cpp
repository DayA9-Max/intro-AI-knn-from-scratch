// k-Nearest Neighbors (kNN) from scratch in C++17
// Beginner-friendly implementation using manual memory management
// No external ML libraries, just arrays, loops, and file I/O

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

// ============================================================================
// Manual Dynamic Array Implementation (no STL containers)
// ============================================================================

struct DataPoint {
    double* features;  // Array of features
    int label;         // Class label
    int num_features;  // Number of features
};

struct Dataset {
    DataPoint* points;  // Array of data points
    int size;          // Number of data points
    int num_features;  // Number of features per point
};

// Helper function to split a string by delimiter
void split_string(const std::string& line, char delim, double* features, int& label, int num_features) {
    int idx = 0;
    std::string temp = "";
    int total_cols = 0;
    
    for (size_t i = 0; i < line.length(); i++) {
        if (line[i] == delim) {
            if (idx < num_features) {
                features[idx] = std::stod(temp);
            } else if (idx == num_features) {
                // This is the label column
                label = std::stoi(temp);
            }
            // Ignore any additional columns beyond num_features + 1
            temp = "";
            idx++;
            total_cols++;
        } else {
            temp += line[i];
        }
    }
    // Last value is the label (or ignored if there are extra columns)
    if (idx == num_features) {
        label = std::stoi(temp);
    }
}

// ============================================================================
// Function: read_csv_all_but_last
// Reads CSV file, treating all columns except the last as features
// and the last column as the label.
// ============================================================================
Dataset read_csv_all_but_last(const std::string& filename, int num_features) {
    Dataset dataset;
    dataset.num_features = num_features;
    
    // First pass: count lines
    std::ifstream file1(filename);
    int line_count = 0;
    std::string line;
    
    while (std::getline(file1, line)) {
        if (!line.empty()) {
            line_count++;
        }
    }
    file1.close();
    
    dataset.size = line_count;
    dataset.points = new DataPoint[line_count];
    
    // Second pass: read data
    std::ifstream file2(filename);
    int idx = 0;
    
    while (std::getline(file2, line)) {
        if (!line.empty()) {
            dataset.points[idx].num_features = num_features;
            dataset.points[idx].features = new double[num_features];
            split_string(line, ',', dataset.points[idx].features, 
                        dataset.points[idx].label, num_features);
            idx++;
        }
    }
    file2.close();
    
    return dataset;
}

// ============================================================================
// Function: distance_sq
// Computes squared Euclidean distance between two feature vectors
// ============================================================================
double distance_sq(const double* features1, const double* features2, int num_features) {
    double sum = 0.0;
    for (int i = 0; i < num_features; i++) {
        double diff = features1[i] - features2[i];
        sum += diff * diff;
    }
    return sum;
}

// ============================================================================
// Function: predict_knn
// Predicts the label for a test point using k-nearest neighbors
// Returns the most common label among the k nearest neighbors
// ============================================================================
int predict_knn(const double* test_features, const Dataset& train_data, int k) {
    // Manual arrays for distances and indices (no STL)
    double* distances = new double[train_data.size];
    int* indices = new int[train_data.size];
    
    // Calculate distances from test point to all training points
    for (int i = 0; i < train_data.size; i++) {
        distances[i] = distance_sq(test_features, train_data.points[i].features, 
                                   train_data.num_features);
        indices[i] = i;
    }
    
    // Simple selection sort to find k smallest distances
    // (Beginner-friendly sorting algorithm)
    for (int i = 0; i < k && i < train_data.size; i++) {
        int min_idx = i;
        for (int j = i + 1; j < train_data.size; j++) {
            if (distances[j] < distances[min_idx]) {
                min_idx = j;
            }
        }
        // Swap distances
        double temp_dist = distances[i];
        distances[i] = distances[min_idx];
        distances[min_idx] = temp_dist;
        // Swap indices
        int temp_idx = indices[i];
        indices[i] = indices[min_idx];
        indices[min_idx] = temp_idx;
    }
    
    // Count votes from k nearest neighbors
    // Assuming labels are 0, 1, 2, ... (small integers)
    int max_label = 0;
    for (int i = 0; i < train_data.size; i++) {
        if (train_data.points[i].label > max_label) {
            max_label = train_data.points[i].label;
        }
    }
    
    int* votes = new int[max_label + 1];
    for (int i = 0; i <= max_label; i++) {
        votes[i] = 0;
    }
    
    // Vote based on k nearest neighbors
    for (int i = 0; i < k && i < train_data.size; i++) {
        int label = train_data.points[indices[i]].label;
        votes[label]++;
    }
    
    // Find label with most votes
    int predicted_label = 0;
    int max_votes = votes[0];
    for (int i = 1; i <= max_label; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }
    
    // Clean up
    delete[] distances;
    delete[] indices;
    delete[] votes;
    
    return predicted_label;
}

// ============================================================================
// Function: accuracy
// Calculates classification accuracy on a test dataset
// ============================================================================
double accuracy(const Dataset& test_data, const Dataset& train_data, int k) {
    int correct = 0;
    
    for (int i = 0; i < test_data.size; i++) {
        int predicted = predict_knn(test_data.points[i].features, train_data, k);
        if (predicted == test_data.points[i].label) {
            correct++;
        }
    }
    
    return (double)correct / test_data.size;
}

// ============================================================================
// Optional Normalization Functions
// ============================================================================

struct ZStats {
    double* means;      // Mean of each feature
    double* std_devs;   // Standard deviation of each feature
    int num_features;   // Number of features
};

// Calculate mean and standard deviation for normalization
ZStats zstats(const Dataset& data) {
    ZStats stats;
    stats.num_features = data.num_features;
    stats.means = new double[data.num_features];
    stats.std_devs = new double[data.num_features];
    
    // Initialize to zero
    for (int i = 0; i < data.num_features; i++) {
        stats.means[i] = 0.0;
        stats.std_devs[i] = 0.0;
    }
    
    // Calculate means
    for (int i = 0; i < data.size; i++) {
        for (int j = 0; j < data.num_features; j++) {
            stats.means[j] += data.points[i].features[j];
        }
    }
    
    for (int i = 0; i < data.num_features; i++) {
        stats.means[i] /= data.size;
    }
    
    // Calculate standard deviations
    for (int i = 0; i < data.size; i++) {
        for (int j = 0; j < data.num_features; j++) {
            double diff = data.points[i].features[j] - stats.means[j];
            stats.std_devs[j] += diff * diff;
        }
    }
    
    for (int i = 0; i < data.num_features; i++) {
        stats.std_devs[i] = std::sqrt(stats.std_devs[i] / data.size);
        // Avoid division by zero
        if (stats.std_devs[i] < 1e-10) {
            stats.std_devs[i] = 1.0;
        }
    }
    
    return stats;
}

// Apply normalization (z-score) to a dataset
void apply_norm(Dataset& data, const ZStats& stats) {
    for (int i = 0; i < data.size; i++) {
        for (int j = 0; j < data.num_features; j++) {
            data.points[i].features[j] = 
                (data.points[i].features[j] - stats.means[j]) / stats.std_devs[j];
        }
    }
}

// ============================================================================
// Cleanup Functions
// ============================================================================

void free_dataset(Dataset& data) {
    for (int i = 0; i < data.size; i++) {
        delete[] data.points[i].features;
    }
    delete[] data.points;
}

void free_zstats(ZStats& stats) {
    delete[] stats.means;
    delete[] stats.std_devs;
}

// ============================================================================
// Main Function - Example Usage
// ============================================================================

int main() {
    std::cout << "=== k-Nearest Neighbors (kNN) from Scratch in C++ ===" << std::endl;
    std::cout << std::endl;
    
    // Load dataset
    std::string filename = "../data/iris.csv";
    int num_features = 4;  // Iris has 4 features
    
    std::cout << "Loading dataset from: " << filename << std::endl;
    Dataset data = read_csv_all_but_last(filename, num_features);
    std::cout << "Loaded " << data.size << " data points with " 
              << data.num_features << " features each." << std::endl;
    std::cout << std::endl;
    
    // Split into train (first 24) and test (last 6)
    Dataset train_data;
    train_data.num_features = num_features;
    train_data.size = 24;
    train_data.points = new DataPoint[24];
    
    Dataset test_data;
    test_data.num_features = num_features;
    test_data.size = 6;
    test_data.points = new DataPoint[6];
    
    // Copy training data
    for (int i = 0; i < 24; i++) {
        train_data.points[i].num_features = num_features;
        train_data.points[i].features = new double[num_features];
        train_data.points[i].label = data.points[i].label;
        for (int j = 0; j < num_features; j++) {
            train_data.points[i].features[j] = data.points[i].features[j];
        }
    }
    
    // Copy test data
    for (int i = 0; i < 6; i++) {
        test_data.points[i].num_features = num_features;
        test_data.points[i].features = new double[num_features];
        test_data.points[i].label = data.points[24 + i].label;
        for (int j = 0; j < num_features; j++) {
            test_data.points[i].features[j] = data.points[24 + i].features[j];
        }
    }
    
    std::cout << "Split: " << train_data.size << " training, " 
              << test_data.size << " test points" << std::endl;
    std::cout << std::endl;
    
    // Test without normalization
    std::cout << "--- Testing WITHOUT normalization ---" << std::endl;
    int k = 3;
    double acc = accuracy(test_data, train_data, k);
    std::cout << "Accuracy with k=" << k << ": " << (acc * 100) << "%" << std::endl;
    std::cout << std::endl;
    
    // Test with normalization
    std::cout << "--- Testing WITH normalization ---" << std::endl;
    ZStats stats = zstats(train_data);
    apply_norm(train_data, stats);
    apply_norm(test_data, stats);
    
    acc = accuracy(test_data, train_data, k);
    std::cout << "Accuracy with k=" << k << " (normalized): " << (acc * 100) << "%" << std::endl;
    std::cout << std::endl;
    
    // Example prediction on a single point
    std::cout << "--- Example Single Prediction ---" << std::endl;
    std::cout << "Test point features: ";
    for (int i = 0; i < num_features; i++) {
        std::cout << test_data.points[0].features[i] << " ";
    }
    std::cout << std::endl;
    
    int pred = predict_knn(test_data.points[0].features, train_data, k);
    std::cout << "Predicted label: " << pred << std::endl;
    std::cout << "Actual label: " << test_data.points[0].label << std::endl;
    std::cout << std::endl;
    
    // Cleanup
    free_dataset(data);
    free_dataset(train_data);
    free_dataset(test_data);
    free_zstats(stats);
    
    std::cout << "Done!" << std::endl;
    
    return 0;
}
