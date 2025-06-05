# 🧠 CIFAR-10 Image Classification with CNN (GoogLeNet) and MLP

## 📖 Table of Contents
- 🎯 Project Overview
- 📂 Dataset
- 🛠️ Requirements
- 🏗️ Model Architectures
  - MLP Architecture
  - CNN (GoogLeNet) Architecture
- 🚀 Implementation Details
  - Data Preprocessing
  - Training Process
  - Evaluation Metrics
- 📊 Results
- 📈 Visualizations
- ⚙️ Usage
- 📝 Notes
- 🤝 Contributing
- 📜 License

## 🎯 Project Overview
The CIFAR-10 dataset is used to train and evaluate two models: an MLP and a CNN (GoogLeNet). The dataset consists of 60,000 32x32 color images across 10 classes. The MLP is a fully connected neural network with multiple hidden layers, while the CNN leverages the GoogLeNet architecture with Inception modules for efficient feature extraction.

**Objective**: Compare the performance of MLP and CNN in terms of accuracy, loss, and computational efficiency for image classification on CIFAR-10.

## 📂 Dataset
The CIFAR-10 dataset includes:

- **Training Set**: 50,000 images  
- **Test Set**: 10,000 images  
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)  
- **Image Size**: 32x32 pixels (RGB)  

The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

## 🛠️ Requirements

To run the notebooks, install the following dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn torchsummary
```

**Hardware**:
- GPU (CUDA-enabled) recommended for faster training.
- CPU fallback supported.

## 🏗️ Model Architectures

### MLP Architecture

Implemented in `MLP.ipynb` with the following structure:

- **Input Layer**: Flattens 32x32x3 images (3,072 features).
- **Hidden Layers**: 5 fully connected layers (512 units each) with:
  - `LeakyReLU` activation (negative slope: 0.1)
  - `Dropout(0.3)` for regularization
- **Output Layer**: 10 units (one per class)
- **Weight Initialization**: Kaiming uniform for linear layers
- **Total Parameters**: ~1.8M

## MLP Model Construction

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
```

### CNN (GoogLeNet) Architecture

Implemented in `CNN.ipynb` using a simplified GoogLeNet architecture with Inception modules:

- **Pre-layers**: 3x3 convolution (192 filters) with BatchNorm and ReLU
- **Inception Modules**: Multiple branches with:
  - 1x1 convolutions
  - 1x1 + 3x3 convolutions
  - 1x1 + 5x5 convolutions
  - 3x3 max pooling + 1x1 convolution
- **Pooling**: MaxPooling and AveragePooling
- **Output Layer**: Fully connected layer with 10 units
- **Total Parameters**: ~6.2M (as shown in `torchsummary`)
## Inception Modules
```python
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)
```

## CNN Model Construction
```python
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```


## 🚀 Implementation Details

### Data Preprocessing

**MLP**:
- Training: Random cropping, horizontal flipping, normalization  
- Testing: Normalization only  
- Batch Size: 128

**CNN**:
- Training/Testing: Normalization only  
- Batch Size: 256  
- Data Loading: DataLoader with 2-4 workers for parallel processing

### Training Process

**MLP**:
- Optimizer: Adam (lr=0.001, weight decay=1e-4)
- Loss Function: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (factor=0.1, patience=10)
- Epochs: Up to 100 with early stopping (patience=20)
- Checkpointing: Saves best model based on test accuracy

**CNN**:
- Optimizer: Adam (details not fully specified in code)
- Loss Function: CrossEntropyLoss
- Epochs: Not explicitly defined in the provided code snippet

### Evaluation Metrics

**Metrics**:
- Training/Test Loss
- Training/Test Accuracy
- Confusion Matrix for class-wise performance

**Evaluation**:
- MLP: Evaluates after each epoch and reports final test accuracy/loss
- CNN: Similar evaluation with confusion matrix visualization

## 📊 Results

## Comparison of MLP and CNN Results

| Criterion               | MLP                              | CNN (GoogLeNet)                  |
|------------------------|----------------------------------|----------------------------------|
| **Initial Accuracy**   | ~10% (random)                   | ~66.7% (random)                   |
| **Final Accuracy**     | 55-60%                          | 90-92%                          |
| **Training Time**      | Faster (simpler architecture)   | Slower (complex Inception)      |
| **Efficiency**         | Poor with spatial data          | Strong, captures spatial features |
| **Parameters**         | ~1.8M                           | ~6.2M                           |
## 📈 Visualizations

Both notebooks include visualizations to analyze model performance:

- **Loss Curves**:
```python
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

- **Accuracy Curves**
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/CNN%20RESULTS/Loss%20-%20Accuracy%20-%20CNN.png" alt="CNN-Loss-Accuracy" width="300"/>
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/MLP%20RESULTS/Loss%20-%20Accuracy%20-%20MLP.png" alt="MLP-Loss-Accuracy" width="300"/>
</div>

  
</div>
- **Confusion Matrix**: Heatmap showing class-wise predictions vs. true labels

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/CNN%20RESULTS/Matrix%20Confusion%20-%20CNN.png?raw=true" alt="CNN-Confusion Matrix" width="300"/>
  <img src="https://github.com/Hieuvu4438/Python-Assignment-02/blob/main/RESULTS/MLP%20RESULTS/Confusion%20Matrix%20-%20MLP.png?raw=true" alt="MLP-Confusion Matrix" width="300"/>
</div>

## ⚙️ Usage

**Clone the Repository**:
```bash
git clone https://github.com/your-username/cifar10-classification.git
cd cifar10-classification
```

**Install Dependencies**:
```bash
pip install -r requirements.txt
```

**Run Notebooks**:
- Open `MLP.ipynb` or `CNN.ipynb` in Jupyter Notebook or Colab.
- Execute cells sequentially to download data, train models, and visualize results.

**Modify Parameters**:
Adjust hyperparameters (e.g., learning rate, batch size, epochs) or try other architectures/augmentations.

## 📝 Notes

- **MLP Limitations**: Struggles with spatial data due to flattening, leading to lower accuracy.
- **CNN Advantages**: GoogLeNet's Inception modules efficiently capture multi-scale features.
- **Hardware**: Use a GPU for faster training, especially for CNN.
- **Incomplete CNN Code**: CNN.ipynb lacks the full training loop and results; ensure completion for practical use.
- **Reproducibility**: Set random seeds (`torch.manual_seed(1)`) for consistent results.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Open a Pull Request  

Please follow [PEP8](https://peps.python.org/pep-0008/) and document your code.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

🌟 Happy Coding! If you find this repository useful, give it a star! 🌟
