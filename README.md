# Text Classification with PyTorch

## Overview

This project implements a text classification model using PyTorch to categorize messages into one of three classes:

* **Enquiry**
* **Job**
* **Spam**

The model uses TF-IDF for text vectorization and a simple feedforward neural network for classification.

---

## Dataset

The dataset used is a CSV file containing the following columns:

* `message`: Raw text input
* `label`: (Manually annotated) class label - one of `enquiry`, `job`, or `spam`

Only a subset of the data is labeled (\~105 samples), making this a semi-supervised learning use case.

---

## Features

* TF-IDF vectorization of text data
* Simple neural network built with PyTorch
* Label encoding and decoding
* Single message prediction function
* Accuracy evaluation

---

## Requirements

Install the required Python libraries:

```bash
pip install torch scikit-learn pandas numpy
```

---

## Project Structure

```
text-classifier/
├── foduu_contacts_message.csv       # Input CSV with messages
├── train.py                         # Script for training the model
├── predict.py                       # Function to predict a label for a new message
├── model.pth                        # Saved PyTorch model (optional)
├── vectorizer.pkl                   # Saved TF-IDF vectorizer
├── label_encoder.pkl                # Saved LabelEncoder
└── README.md                        # Project documentation
```

---

## How to Use

### 1. Train the Model

Run the training script:

```bash
python train.py
```

This will:

* Load and preprocess the data
* Train a neural network
* Save the trained model, label encoder, and TF-IDF vectorizer

### 2. Predict a Single Message

After training, import and use the `predict_message()` function:

```python
from predict import predict_message

message = "Looking for digital marketing jobs."
print(predict_message(message))  # Output: 'job'
```

---

## Model Architecture

* **Input Layer**: TF-IDF vector (500 features)
* **Hidden Layer**: 128 neurons, ReLU activation
* **Output Layer**: 3 neurons (softmax for class probabilities)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam
* **Epochs**: 20 (adjustable)

---

## Evaluation

The model achieves \~85% accuracy on the small labeled subset.

---

## Improvements

* Add more labeled samples
* Use more advanced NLP embeddings (e.g., BERT, FastText)
* Add cross-validation and confusion matrix
* Deploy using Flask or Streamlit

---

## Author

**Vivek Nagar**
MCA Graduate | Aspiring Machine Learning Engineer

---

## License

This project is licensed under the [MIT License](LICENSE).
