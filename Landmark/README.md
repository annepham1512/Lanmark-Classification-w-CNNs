# Landmark Classification with CNNs

## Overview

This project involves building a machine learning pipeline to classify images of landmarks into predefined categories. The task is challenging due to the diverse visual features of landmarks and the presence of mundane images within the dataset. The project demonstrates both the development of a Convolutional Neural Network (CNN) from scratch and the use of transfer learning to achieve robust classification performance.

---

## Project Goals

1. Develop a CNN from scratch to classify landmark images and achieve a test accuracy of at least **50%**.
2. Apply transfer learning with a pretrained model to further improve accuracy, aiming for **>75%**.
3. Explore and implement advanced techniques for model optimization, data augmentation, and efficient training.
4. Export the trained models using TorchScript for deployment.

---

## Directory Structure

```
.
|-- cnn_from_scratch.ipynb       # Jupyter notebook for CNN training from scratch
|-- transfer_learning.ipynb      # Jupyter notebook for transfer learning
|-- src/                         # Source code directory
|   |-- data.py                  # Data preprocessing and augmentation functions
|   |-- model.py                 # Definition of the CNN architecture
|   |-- train.py                 # Training and evaluation scripts
|   |-- optimization.py          # Loss and optimizer setup
|   |-- transfer.py              # Transfer learning model setup
|   |-- predictor.py             # Model prediction and export functionality
|-- checkpoints/                 # Directory for storing model weights
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project documentation (this file)
|-- static_images/               # Sample images and icons for documentation
```

---

## Key Features

### 1. CNN from Scratch

- **Architecture**: Designed a CNN with convolutional, pooling, and dropout layers, followed by a fully connected classifier.
- **Data Augmentation**: Applied transformations such as cropping, horizontal flipping, and color jittering to increase dataset diversity.
- **Performance**: Achieved a test accuracy of **69%**.

### 2. Transfer Learning

- **Pretrained Model**: Used ResNet50 for transfer learning, fine-tuned on the landmark dataset.
- **Optimization**: Employed Adam optimizer with regularization to combat overfitting.
- **Performance**: Improved test accuracy to **79%**, demonstrating the effectiveness of transfer learning.

### 3. Model Deployment

- Exported both models using TorchScript, enabling seamless integration into production environments.
- Verified exported models for consistent performance with test datasets.

---

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository/landmark-classification.git
   cd landmark-classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   The dataset is automatically downloaded and set up when running the notebooks.

4. Run the notebooks:
   - For CNN from scratch: `cnn_from_scratch.ipynb`
   - For transfer learning: `transfer_learning.ipynb`

---

## Usage

### Training

- Modify hyperparameters in the respective notebooks or source files.
- Run training using the provided Jupyter notebooks.

### Testing

- Use the `one_epoch_test` function from `src/train.py` to evaluate model performance on the test set.

### Deployment

- Load exported models using TorchScript for inference in your applications:
  ```python
  import torch
  model = torch.jit.load("checkpoints/transfer_exported.pt")
  model.eval()
  ```

---

## Results

- **CNN from Scratch**: Test Accuracy: **69%**
- **Transfer Learning**: Test Accuracy: **79%**

### Confusion Matrix

![Confusion Matrix](static_images/confusion_matrix.png)

---

## Future Work

1. Address class imbalance by augmenting underrepresented classes or using weighted loss functions.
2. Experiment with ensemble models to improve classification accuracy.
3. Optimize the inference pipeline for real-time applications.
4. Extend the project to classify more landmarks or adapt it for other domains.

---

## Acknowledgments

- Udacity for providing the dataset and starter code.
- The PyTorch and torchvision communities for pretrained models and utilities.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or feedback, please contact [Anne PhaM](mailto:phamn@dickinson.edu).
