![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)


---

# 🧠 Deep Learning with PyTorch – Hands-On Workshop

This repository contains my implementations from a multi-day **Deep Learning with PyTorch workshop**, covering fundamental architectures like **Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs)**.

I’m documenting my journey here as I build and train models from scratch using PyTorch.

---

## 📌 Day 2: Artificial Neural Network (ANN)

### ✨ What I built

* Implemented a **fully-connected feedforward neural network** (ANN) in PyTorch.
* Trained it on the **MNIST dataset** (handwritten digits).
* Debugged and learned about:

  * Device management (`cpu` vs `cuda`)
  * Training loops (forward, loss, backward, optimizer step)
  * Inference workflow (`model.eval()`, `torch.no_grad()`)

### 📊 Results

* Achieved \~**96% accuracy** on MNIST test set with a simple ANN.
* First successful end-to-end deep learning project 🎉

### 🚀 Inference Example

```python
model.eval()
with torch.no_grad():
    X, y = test_dataset[0]
    X = X.unsqueeze(0).to(device)
    pred = model(X).argmax(dim=1).item()

print(f"True Label: {y}, Predicted: {pred}")
```

---

## 📌 Upcoming Days

* **Day 2+:** Extend to CNNs for image classification.
* **Day 3+:** Explore RNNs for sequential data.
* Final day: Combine learnings into a mini-project.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Framework:** PyTorch
* **Dataset:** MNIST (via `torchvision.datasets`)
* **Hardware:** Trained on GPU (`cuda`)

---

## 📂 Project Structure

```
├── torch-day1.ipynb    # Jupyter notebook: Basics of Pytorch and DL
├── torch-day2.ipynb    # Jupyter notebook: ANN on MNIST (Day 1)
├── ann.pt              # Saved ANN model weights
├── data/               # Directory for datasets (e.g., MNIST)
└── README.md
```
```

---

## 📈 Learning Outcomes

* How to implement, train, and evaluate deep learning models in PyTorch.
* Understanding of forward/backward passes, optimizers, and loss functions.
* Practical debugging of device mismatches and training stability issues.

---

## 🙌 Acknowledgements

Thanks to the IITM workshop mentors for guiding this journey and to the PyTorch community for excellent documentation.

---

⚡ *This repo is a work in progress — stay tuned as I add CNNs, RNNs, and more experiments!*
