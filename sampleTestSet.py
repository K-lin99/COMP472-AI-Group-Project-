"""
sampleTestSet.py
----------------
Extracts 100 images total from each dataset, sampled equally across all classes,
and saves them as PNG files into sample_test_set/<datasetN>/<class_name>/.

Dataset 1 : 4 classes  : 25 images/class
Dataset 2 : 4 classes  → 25 images/class
Dataset 3 : 15 classes → 6 images/class  (last class gets 10 to reach 100)

Images are taken from the test split of each dataset.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Config
SEED              = 42
TOTAL_IMAGES      = 100
OUTPUT_DIR        = Path('./sample_test_set')

PREPROCESSED = {
    'dataset1': Path('./Training Models/brain_mri_preprocessed'),
    'dataset2': Path('./Training Models/brain_mri_preprocessed'),
    'dataset3': Path('./Training Models/brain_mri_preprocessed'),
}

random.seed(SEED)
torch.manual_seed(SEED)


def load_test_split(preprocessed_dir: Path, dataset_name: str) -> dict:
    path = preprocessed_dir / f'{dataset_name}_test.pt'
    data = torch.load(path, weights_only=False)
    return {
        'images':         data['images'],           # (N, 1, 224, 224) float32 [0,1]
        'labels':         data['labels'],            # (N,) int64
        'class_to_label': data['class_to_label'],   # {class_name: int}
    }


def sample_per_class(images, labels, class_to_label, total=100):
    """Return indices sampled as evenly as possible across all classes."""
    label_to_class = {v: k for k, v in class_to_label.items()}
    num_classes    = len(class_to_label)
    base           = total // num_classes
    remainder      = total % num_classes

    selected = []  # list of (idx, class_name)
    for class_name, label_int in sorted(class_to_label.items(), key=lambda x: x[1]):
        class_idx = (labels == label_int).nonzero(as_tuple=True)[0].tolist()
        n_pick    = base + (1 if label_int < remainder else 0)
        n_pick    = min(n_pick, len(class_idx))
        picked    = random.sample(class_idx, n_pick)
        for idx in picked:
            selected.append((idx, class_name))

    return selected


def save_images(selected, images, dataset_name, output_dir: Path):
    saved = 0
    for idx, class_name in selected:
        class_dir = output_dir / dataset_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        img_tensor = images[idx].squeeze(0)                     # (224, 224)
        img_np     = (img_tensor.numpy() * 255).astype(np.uint8)
        img_pil    = Image.fromarray(img_np, mode='L')
        img_pil.save(class_dir / f'{dataset_name}_{idx:05d}.png')
        saved += 1
    return saved


# -- Main --─────────────────────────────────────────────────────────────────
for dataset_name, preprocessed_dir in PREPROCESSED.items():
    print(f'\n-- {dataset_name} --')
    data           = load_test_split(preprocessed_dir, dataset_name)
    images         = data['images']
    labels         = data['labels']
    class_to_label = data['class_to_label']

    num_classes = len(class_to_label)
    print(f'  Classes   : {num_classes}')
    print(f'  Test size : {len(images):,}')

    selected = sample_per_class(images, labels, class_to_label, total=TOTAL_IMAGES)
    saved    = save_images(selected, images, dataset_name, OUTPUT_DIR)

    per_class = {}
    for _, cls in selected:
        per_class[cls] = per_class.get(cls, 0) + 1
    for cls, count in sorted(per_class.items()):
        print(f'    {cls:<30} {count} images')
    print(f'  Total saved: {saved}')

print(f'\nDone. Images saved to: {OUTPUT_DIR.resolve()}')
