🧠 MNIST Digit Classifier – Neural Network from Scratch (NumPy)

> A simple yet powerful 2-layer neural network built **completely from scratch using NumPy**, designed to classify handwritten digits from the **MNIST dataset**.

---

## 🚀 Overview

This project demonstrates the complete machine learning pipeline without using high-level libraries like TensorFlow or PyTorch. It covers:

- Loading and preprocessing MNIST image data
- Building a 2-layer feedforward neural network
- Forward & backward propagation using matrix math
- Training via gradient descent
- Predicting digits with softmax activation
- Visualizing predictions and accuracy

---

## 🧱 Model Architecture

| Layer     | Size            | Activation |
|-----------|------------------|------------|
| Input     | 784 (28×28 image)| None       |
| Hidden    | 10 neurons       | ReLU       |
| Output    | 10 neurons       | Softmax    |

---

## 📂 File Structure

```

project/
│
├── data/
│   └── dataset/
│       └── train.csv            # MNIST training data (digits 0–9)
│
├── mnist.py                  # Complete model training script
├── README.md                    # Project documentation

````

---

## 🧪 How It Works

### 1. Data Preprocessing
- Reads `train.csv` using Pandas.
- Normalizes pixel values to `[0, 1]`.
- Splits into training and validation sets.
- One-hot encodes the labels for loss calculation.

### 2. Neural Network Training
- Initializes weights and biases for two layers.
- Performs **forward propagation** using ReLU and Softmax.
- Computes **cross-entropy loss**.
- Performs **backpropagation** to compute gradients.
- Updates weights via **gradient descent**.

### 3. Model Evaluation
- Calculates prediction accuracy on both train and dev sets.
- Includes a helper to visualize and test individual digit predictions.

---

## 🖼️ Sample Prediction

```python
test_prediction(0, W1, b1, W2, b2)
````

Outputs:

```
Prediction:  [2]
Label:  2
```

And displays the digit image using `matplotlib`.

---

## 📊 Results

| Metric       | Accuracy |
| ------------ | -------- |
| Training Set | \~80–84% |
| Dev Set      | \~82–85% |

*(Results may vary slightly with different runs due to random initialization)*

---

## 🧠 Key Concepts Used

* Matrix Multiplication & Vectorization (NumPy)
* Activation Functions: ReLU, Softmax
* Loss Function: Categorical Cross-Entropy
* Backpropagation and Chain Rule
* Gradient Descent Optimization
* One-hot Encoding
* Visualization with `matplotlib`

---

## 📦 Requirements

* Python 3.x
* `numpy`
* `pandas`
* `matplotlib`

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## 🎯 Learning Objectives

This project is ideal for:

* Understanding neural networks under the hood
* Practicing vectorized coding
* Learning how gradient descent and backpropagation work
* Building a portfolio piece that proves your fundamentals

---

## 📌 Future Enhancements

* Add more hidden layers or dropout
* Visualize loss curve over epochs
* Extend to use TensorFlow/Keras version
* Integrate test set evaluation
* Add model persistence (save/load weights)

---

## 👨‍💻 Author

**Sanjay** — *Aspiring Data Scientist*
Follow my journey in building robust ML projects from scratch 🚀

---

## 📜 License

This project is open-source and available for personal learning and portfolio use.
