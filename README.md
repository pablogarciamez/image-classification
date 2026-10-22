# Fashion-MNIST image classification

Classifying the images of the Fashion-MNIST dataset with neural networks, starting with a basic multilayer perceptron (MLP).

Fashion-MNIST has 60,000 training images and 10,000 test images of clothing items, split into 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Each image is 28 × 28 pixels in grayscale. The test set has exactly 1,000 images per class.

## Setup

Create a virtual environment at the root of the repository and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

requirements.txt pins the CPU-only builds of PyTorch (torch==2.14.1+cpu, torchvision==0.29.1+cpu). These builds are not on PyPI, so the first line of the file (--extra-index-url https://download.pytorch.org/whl/cpu) tells pip to also look in PyTorch's own package index.

## Repository structure

- `notebooks/01_mlp.ipynb`: MLP on flattened images.
- `data/`: not included in the repository. torchvision downloads Fashion-MNIST here the first time the notebook runs.

## Results

### MLP (`notebooks/01_mlp.ipynb`)

Each 28 × 28 image is flattened into a vector of 784 values. The network has a hidden layer of 128 units (ReLU) and an output layer of 10 units, one score per class: 784-128-10, 101,770 parameters. It was trained for 5 epochs with Adam (learning rate 0.001) and batches of 64, minimising cross-entropy, with `torch.manual_seed(42)` so that results are reproducible. More epochs could improve it.

**Test accuracy: 86.3%**, against 10% for random guessing.

The confusion matrix (rows: true class, columns: predicted class) shows where the errors come from:

- Classes with a distinctive shape are almost always right: Bag (97.7%), Trouser (97.1%), Sandal (96.3%).
- Upper-body garments (T-shirt/top, Pullover, Coat, Shirt) are confused with each other. The worst class is Pullover (63.9%): 171 pullovers are predicted as Coat and 165 as Shirt. Shirt also absorbs many errors: 196 T-shirts/tops and 117 coats are predicted as Shirt. At 28 × 28 in grayscale the details that tell them apart (collar, buttons, sleeves, texture) are mostly lost.
- Footwear is also confused among itself to a lesser degree: 43 sneakers are predicted as Sandal and 36 ankle boots as Sneaker.
- Which upper-body class comes out worst changes between runs (without a fixed seed, another run gave 86.7% accuracy with Shirt as the worst class, at 67.8%), but the pattern of confusion among the four is stable.
