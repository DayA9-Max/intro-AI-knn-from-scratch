# k-Nearest Neighbors (kNN) from Scratch

From-scratch k-Nearest Neighbors (kNN) implementation in C++17 and Python 3 for an intro programming/AI course. This project uses **no external ML libraries**—just arrays, loops, and basic file I/O.

Perfect for learning:
- **Arrays and loops** - Core programming constructs
- **Manual memory management** (C++)
- **Structs/Classes** - Data organization
- **File I/O** - Reading CSV files
- **Algorithm implementation** - Understanding ML from the ground up

## 🎯 Features

- ✅ **C++17 implementation** with manual dynamic arrays (no STL containers)
- ✅ **Python 3 implementation** using only standard library (csv module, lists)
- ✅ **Clear function boundaries** for educational purposes
- ✅ **Optional normalization** (z-score)
- ✅ **Sample Iris dataset** included
- ✅ **Beginner-friendly** with extensive comments

## 📁 Project Structure

```
intro-AI-knn-from-scratch/
├── cpp/
│   └── knn.cpp          # C++ implementation
├── python/
│   └── knn.py           # Python implementation
├── data/
│   └── iris.csv         # Sample Iris dataset
└── README.md            # This file
```

## 🔧 Core Functions

Both implementations include these key functions:

1. **`read_csv_all_but_last`** - Reads CSV file, treating all columns except the last as features and the last column as the label
2. **`distance_sq`** - Computes squared Euclidean distance between two feature vectors
3. **`predict_knn`** - Predicts the label for a test point using k-nearest neighbors
4. **`accuracy`** - Calculates classification accuracy on a test dataset
5. **`zstats`** (optional) - Calculates mean and standard deviation for normalization
6. **`apply_norm`** (optional) - Applies z-score normalization to features

## 🚀 Getting Started

### C++ Version

**Requirements:**
- C++17 compiler (g++, clang++, etc.)
- Standard headers only (iostream, fstream, cmath, string)

**Compile:**
```bash
cd cpp
g++ -std=c++17 -o knn knn.cpp
```

**Run:**
```bash
./knn
```

### Python Version

**Requirements:**
- Python 3.x
- Standard library only (csv, math modules)

**Run:**
```bash
cd python
python3 knn.py
```

## 📊 Sample Dataset

The included `iris.csv` contains a subset of the classic Iris dataset:
- **Features:** 4 numeric values (sepal length, sepal width, petal length, petal width)
- **Labels:** 0 (setosa), 1 (versicolor), 2 (virginica)
- **Size:** 30 samples (10 per class)

Format:
```
feature1,feature2,feature3,feature4,label
5.1,3.5,1.4,0.2,0
...
```

## 🎓 Educational Focus

### C++ Implementation Highlights

- **Manual memory management** using `new` and `delete`
- **No STL containers** - implements dynamic arrays from scratch
- **Struct-based data organization** (DataPoint, Dataset, ZStats)
- **Proper cleanup** with dedicated free functions
- **Selection sort** for finding k-nearest neighbors (beginner-friendly)

### Python Implementation Highlights

- **List-based operations** - no numpy arrays
- **CSV module** for file reading
- **Dictionary** for vote counting
- **List comprehensions** for data manipulation
- **Pure Python** - no external dependencies

## 📝 Algorithm Overview

**k-Nearest Neighbors (kNN) Classification:**

1. **Load data** from CSV file
2. **Split** into training and test sets
3. **For each test point:**
   - Calculate distance to all training points
   - Find k nearest neighbors
   - Vote on the label (majority wins)
4. **Evaluate** accuracy on test set

**Optional Normalization:**
- Calculate mean and standard deviation from training data
- Apply z-score normalization: `(x - mean) / std_dev`
- Helps when features have different scales

## 🔍 Example Output

```
=== k-Nearest Neighbors (kNN) from Scratch ===

Loading dataset from: ../data/iris.csv
Loaded 30 data points with 4 features each.

Split: 24 training, 6 test points

--- Testing WITHOUT normalization ---
Accuracy with k=3: 100.0%

--- Testing WITH normalization ---
Accuracy with k=3 (normalized): 100.0%

--- Example Single Prediction ---
Test point features: [0.8, 0.7, 1.2, 0.9]
Predicted label: 1
Actual label: 1

Done!
```

## 🛠️ Customization

**Using your own dataset:**

1. Create a CSV file with format: `feature1,feature2,...,featureN,label`
2. Update the filename in `main()` function
3. Set the correct number of features (C++: `num_features` variable)
4. Ensure labels are integers starting from 0

**Tuning k:**

Change the `k` value in the `main()` function to experiment with different numbers of neighbors.

## 📚 Learning Resources

**Key Concepts:**
- **Supervised Learning** - Learning from labeled examples
- **Instance-based Learning** - kNN stores all training data
- **Distance Metrics** - Euclidean distance measures similarity
- **Voting** - Classification by majority vote
- **Normalization** - Scaling features to similar ranges

**Good for:**
- Intro to AI/ML courses
- Data structures practice
- Algorithm implementation
- Understanding ML fundamentals

## 🎯 Exercises for Students

1. **Modify** the distance function to use Manhattan distance instead of Euclidean
2. **Implement** weighted voting (closer neighbors have more influence)
3. **Add** cross-validation to find the best k value
4. **Compare** accuracy with and without normalization
5. **Create** your own dataset and test the classifier
6. **Optimize** by implementing k-d trees for faster neighbor search

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs
- Suggest improvements
- Create educational materials
- Share your own datasets

Keep it beginner-friendly and library-free!

## ✨ Acknowledgments

- Classic Iris dataset by R.A. Fisher (1936)
- Designed for intro programming/AI education
- Emphasizes fundamental concepts over optimization
