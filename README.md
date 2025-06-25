# 🧠 Text Classification using Transformers (PyTorch & TensorFlow)

## 📌 Overview

This project builds a **text classification system** to categorize real-world message data into one of three categories:

* **Enquiry**
* **Job**
* **Spam**

It starts with a **TF-IDF + Neural Network** (PyTorch), then progresses to an advanced version using **Transformers (BERT)** with **both PyTorch and TensorFlow**. This covers foundational to modern NLP approaches in a semi-supervised learning setting.

---

## 📂 Dataset

* **Source**: Real-world business message dataset (`contacts_message.csv`)
* **Columns**:

  * `message`: Raw user message
  * `label`: Manually annotated label (only \~105 out of \~1700 are labeled)
* This is a **semi-supervised** use case (limited labeled data).

---

## 🚀 Features

✅ Text vectorization using **TF-IDF**
✅ Simple neural network with **PyTorch**
✅ Transformer-based classifier using **BERT (HuggingFace Transformers)**
✅ Implemented in both **PyTorch** and **TensorFlow**
✅ Label encoding & decoding
✅ Single message prediction interface
✅ Model evaluation & accuracy report

---

## 🛠️ Requirements

```bash
pip install torch transformers tensorflow scikit-learn pandas numpy
```

---

## 📁 Project Structure

```
text-classifier/
├── foduu_contacts_message.csv       # Input CSV with messages
├── train.py                         # PyTorch model training with TF-IDF
├── predict.py                       # PyTorch TF-IDF-based prediction
├── bert_pytorch.py                  # BERT-based model using PyTorch
├── bert_tensorflow.py               # BERT-based model using TensorFlow
├── model.pth                        # Saved PyTorch model
├── bert_model.h5                    # Saved TensorFlow BERT model
├── vectorizer.pkl                   # TF-IDF vectorizer (pickle)
├── label_encoder.pkl                # Label encoder (pickle)
└── README.md                        # Project documentation
```

---

## ⚙️ How to Use

### 1️⃣ Train Basic Model (TF-IDF + NN)

```bash
python train.py
```

This:

* Loads labeled data
* Vectorizes using TF-IDF
* Trains a feedforward NN
* Saves model, vectorizer, and label encoder

### 2️⃣ Predict with Basic Model

```python
from predict import predict_message
print(predict_message("Looking for freelance job"))  # ➡️ job
```

### 3️⃣ Train BERT Model (PyTorch)

```bash
python bert_pytorch.py
```

Trains a Transformer-based classifier using HuggingFace and PyTorch.

### 4️⃣ Train BERT Model (TensorFlow)

```bash
python bert_tensorflow.py
```

Runs the same logic in TensorFlow (using Keras Functional API + BERT).

---

## 🧠 Model Architectures

### TF-IDF + Feedforward NN

* **Input**: TF-IDF (500 features)
* **Hidden**: 128 neurons, ReLU
* **Output**: 3 classes (softmax)
* **Loss**: CrossEntropyLoss
* **Optimizer**: Adam
* **Framework**: PyTorch

### Transformer (BERT)

* **Input**: Tokenized message
* **Base**: `bert-base-uncased` (HuggingFace)
* **Output**: Softmax layer (3 classes)
* **Loss**: CrossEntropy
* **Frameworks**: PyTorch & TensorFlow

---

## 📊 Evaluation

* Accuracy (TF-IDF Model): \~85% (on limited 105 labeled samples)
* BERT Model: Better contextual performance (subject to labeled data size)

---

## 💡 Future Improvements

* Increase labeled data (active learning)
* Fine-tune larger transformer models
* Add Streamlit or Flask-based UI
* Add confidence scores & explainability (e.g., SHAP)

---

## 👨‍💻 Author

**Vivek Nagar**
*MCA Graduate | Aspiring Machine Learning Engineer*
🔗 [LinkedIn Profile](https://www.linkedin.com/in/vivek-nagar)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

