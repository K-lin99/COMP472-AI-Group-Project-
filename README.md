# Brain Tumor MRI Classification

**Team:** Lucas Graham 40249532 · Dominique Proulx 40177566 · Azmi Abidi 40248132 · Kalin Milanov 40179417

## Project Description

Brain tumor classification from MRI scans using deep learning. Three models (MobileNetV2, EfficientNet-B2, ResNet-34) are trained from scratch on three brain MRI datasets containing 4 to 44 tumor classes. Hyperparameter tuning is performed on MobileNetV2. Transfer learning variants of EfficientNet-B2 and ResNet-34 are also evaluated. An ablative study is conducted on EfficientNet-B2 using Dataset 2 and Dataset 3.

## Requirements

Python 3.10+ and the following libraries:

```
torch torchvision
numpy pandas matplotlib seaborn
scikit-learn torchmetrics
Pillow tqdm
monai
pytorch-grad-cam
kaggle
```

Install all at once:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn torchmetrics Pillow tqdm monai grad-cam kaggle
```

## Datasets

All datasets are downloaded automatically by the preprocessing notebooks via the Kaggle API.

Place your `kaggle.json` credentials at `~/.kaggle/kaggle.json` (download from kaggle.com, Account, API, Create New Token).

| Dataset | Classes | Kaggle slug |
|---------|---------|-------------|
| 1 | 4 | `sartajbhuvaji/brain-tumor-classification-mri` |
| 2 | 4 | `masoudnickparvar/brain-tumor-mri-dataset` |
| 3 | 44 | `fernando2rad/brain-tumor-mri-images-44c` |

## How to Train

1. Run the preprocessing notebook to download and prepare the data:
   `Training Models/preprocessing.ipynb`

2. Run the training notebook for the desired model:

   **From scratch**
   - `Training Models/mobilenet_v2_brain_tumor.ipynb` (MobileNetV2 - Google Colab file) 
   - `Training Models/efficientnet_b2.ipynb` (EfficientNet-B2)
   - `Training Models/resnet_34.ipynb` (ResNet-34)

   **Hyperparameter tuning with Optuna**
   - `Training Models/mobilenet_v2_brain_tumor.ipynb` (MobileNetV2, includes hyperparameter tuning)

   **Transfer learning**
   - `Transfer Learning/efficientnet_b2_transfer_learning.ipynb`
   - `Transfer Learning/resnet34_transfer_learning_model.ipynb`

   **Ablative study**
   - `Ablative Study/ablativeStudy_preprocessing.ipynb` (run first)
   - `Ablative Study/ablativeStudy_efficientnet_b2_Dataset2.ipynb`
   - `Ablative Study/ablativeStudy_efficientnet_b2_Dataset3.ipynb`

Each notebook saves the best checkpoint to its `outputs/` folder.

## Running on the Sample Test Set

A sample test set of 100 images per dataset (evenly distributed across classes) is provided in `sample_test_set/`.

To regenerate it:
```bash
python sampleTestSet.py
```

To run a pre-trained model on it, load the saved `.pth` checkpoint in the corresponding training notebook and point the test loader at the `sample_test_set/` images.

## Source Code

```
Colab Files/
  preprocessing.ipynb
  mobilenet_v2_brain_tumor.ipynb
  efficientNet_B2.ipynb
  resnet34_brain_tumor.ipynb

Training Models/
  preprocessing.ipynb
  efficientnet_b2.ipynb
  resnet_34.ipynb

Transfer Learning/
  efficientnet_b2_transfer_learning.ipynb
  resnet34_transfer_learning_model.ipynb

Ablative Study/
  ablativeStudy_preprocessing.ipynb
  ablativeStudy_efficientnet_b2_Dataset2.ipynb
  ablativeStudy_efficientnet_b2_Dataset3.ipynb

sampleTestSet.py
```
