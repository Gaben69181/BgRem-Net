# Background Removal with U-Net Segmentation

Final project for Deep Learning course – image background removal via pixel-wise
segmentation using a U-Net convolutional neural network implemented in PyTorch.

The system takes an RGB image (any resolution up to ~3000×3000), resizes it to
512×512 internally, predicts a foreground mask, and returns:

- A **binary foreground mask**
- A **soft (probabilistic) mask**
- A **background-removed image** (transparent PNG or composited with a solid color)
- (Optional) **timelapse videos** showing how the predicted mask improves over epochs

Everything runs **fully locally** with a **Streamlit** frontend and a modular
backend for training, evaluation, inference, and visualization.

## 1. Project structure

```text
project_root/
├── backend/
│   ├── model.py          # U-Net model definition
│   ├── dataset.py        # Dataset + dataloaders for paired images / masks
│   ├── train.py          # Training & validation loop (+ timelapse frame export)
│   ├── inference.py      # Inference utilities & CLI
│   ├── timelapse.py      # Build GIF/MP4 timelapse videos from frames
│   ├── metrics.py        # Losses (BCE+Dice) and metrics (IoU, Dice, Pixel Acc.)
│
├── frontend/
│   └── app.py            # Streamlit web UI for interactive background removal
│
├── data/
│   ├── train/
│   │   ├── images/       # Training RGB images
│   │   └── masks/        # Corresponding binary/alpha masks
│   └── val/
│       ├── images/       # Validation RGB images
│       └── masks/        # Corresponding binary/alpha masks
│
├── outputs/
│   ├── checkpoints/      # Saved model checkpoints (best_model.pth)
│   ├── images/           # Inference outputs (masks + composited foregrounds)
│   ├── timelapse/        # Per-epoch prediction frames 
│   └── videos/           # GIF/MP4 timelapse videos
│
├── requirements.txt      # Python dependencies
└── README.md             # This documentation
```

## 2. Installation

1. Create and activate a Python environment (recommended: Python 3.10+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   # or
   .venv\Scripts\activate    # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional but recommended) Verify that PyTorch detects your GPU:

   ```python
   import torch
   print(torch.cuda.is_available())
   ```

4. Download and prepare the dataset:

   ```bash
   python download_dataset.py
   ```

   This downloads the AISegment.com Matting Human Datasets from Kaggle and
   organizes them into the required `data/` folder structure.

## 3. Dataset

The code assumes a **paired image–mask dataset** with the following structure:

```text
data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

- Image file names in `images/` and `masks/` must match by stem, e.g.:
  - `images/0001.jpg` ↔ `masks/0001.png`
  - `images/dog_01.png` ↔ `masks/dog_01.png`
- Masks should be **single-channel** (binary) images where foreground pixels are
  white (255) and background pixels are black (0). Grayscale alpha mattes are
  automatically thresholded at 0.5.

### 3.1 Automatic dataset download

To use the AISegment.com Matting Human Datasets from Kaggle, run:

```bash
python download_dataset.py
```

This will:
- Download the dataset using `kagglehub`
- Organize the files into `data/train/` and `data/val/` subfolders

**Note:** The script includes placeholder paths for organizing the dataset.
After downloading, inspect the downloaded folder structure and adjust the
`organize_dataset()` function in `download_dataset.py` if needed to match
the actual dataset layout.

### 3.2 Manual dataset setup

Alternatively, you can manually prepare any human foreground matting /
segmentation dataset. Examples:

- **P3M-10k** – high-quality portrait images with alpha mattes
- **HumanSeg** – person segmentation dataset
- **Adobe Image Matting** subset – foreground/background with alpha mattes

Convert / rename files as necessary so they follow the folder structure above.

## 4. Model architecture (U-Net)

The network is a standard **U-Net** for binary segmentation, implemented in
`backend/model.py`. Key properties:

- Input: 3-channel RGB image, resized to 512×512.
- Output: 1-channel logit map (same spatial size) for foreground probability.
- Encoder: 4 levels of downsampling with `DoubleConv` blocks (Conv–BN–ReLU ×2).
- Decoder: 4 levels of upsampling with skip connections from encoder.
- Final 1×1 convolution maps decoder features to a 1-channel logit mask.
- Upsampling uses bilinear interpolation for stability.

During training we apply `torch.sigmoid` to the logits inside the Dice loss; at
inference time, we threshold the mask (default 0.5) to get a binary foreground.

## 5. Training

The training script is `backend/train.py` and can be run from
the project root:

```bash
python backend/train.py \
  --data-root data \
  --epochs 20 \
  --batch-size 4 \
  --img-size 512 \
  --lr 1e-3 \
  --timelapse
```

Important arguments:

- `--data-root` : Root folder containing `train/` and `val/` subfolders.
- `--epochs`    : Number of training epochs.
- `--batch-size`: Batch size (fits into GPU memory; adjust as needed).
- `--img-size`  : Internal square resize; 512 is a good default.
- `--lr`        : Learning rate for Adam optimizer.
- `--timelapse` : If set, saves per-epoch prediction frames for a few
                  validation samples into `outputs/timelapse/`.
- `--pretrained-checkpoint` : Optional `.pth` file for fine-tuning instead of
                              training from scratch.

Checkpoints are saved to `outputs/checkpoints/best_model.pth` based on the best
validation IoU score.

### 5.1 Loss functions

We use a combination of:

- **Binary Cross Entropy with Logits (BCE)**
- **Dice Loss** (soft version)

combined as:

\( L = \lambda_{BCE} \cdot L_{BCE} + \lambda_{Dice} \cdot L_{Dice} \)

This captures both pixel-wise accuracy (BCE) and overlap quality (Dice).

### 5.2 Evaluation metrics

For each validation batch we compute:

- **IoU (Intersection over Union)**
- **Dice Score**
- **Pixel Accuracy**

These are implemented in `backend/metrics.py` and reported
per epoch in the training logs.

## 6. Timelapse visualization

If you train with `--timelapse`, the script will:

1. Select a fixed subset of validation images.
2. For each epoch, save a horizontal strip image:
   `[ input | ground-truth mask | predicted mask ]`
   into `outputs/timelapse/sample_xx/epoch_XXX.png`.

To convert these frames into a GIF or MP4 video, run:

```bash
python backend/timelapse.py \
  --frames-root outputs/timelapse \
  --out-dir outputs/videos \
  --format gif   # or mp4
```

The resulting videos can be used in presentations to demonstrate how the model
gradually learns better segmentation masks over time.

## 7. Inference (CLI)

You can run inference on a single image directly using `backend/inference.py`:

```bash
python backend/inference.py path/to/image.jpg \
  --checkpoint outputs/checkpoints/best_model.pth \
  --img-size 512 \
  --threshold 0.5 \
  --out-dir outputs/images
```

Outputs:

- `*_mask_binary.png` – binary foreground mask (0/255).
- `*_mask_soft.png`   – soft grayscale probability mask (0–255).
- `*_foreground.png`  – composited RGBA image with transparent background.

You can also optionally specify a solid background color (e.g. green screen):

```bash
python backend/inference.py path/to/image.jpg \
  --checkpoint outputs/checkpoints/best_model.pth \
  --bg-color 0,255,0
```

## 8. Streamlit web application

The interactive frontend is implemented in `frontend/app.py`.
Launch it from the project root with:

```bash
streamlit run frontend/app.py
```

Features:

- Upload any RGB image (PNG/JPG/JPEG/BMP/TIFF) up to ~3000×3000.
- Configure internal resize size and mask threshold.
- Choose output background:
  - Transparent (PNG alpha channel).
  - Solid color (chosen via color picker).
- View side-by-side:
  - Original image
  - Segmentation mask (soft)
  - Background-removed result
- Download:
  - Binary mask (PNG)
  - Soft mask (PNG)
  - Foreground-composited PNG
- Browse and play generated timelapse GIF/MP4 files from `outputs/videos/`.

The app automatically selects GPU if available; otherwise it uses CPU.

## 9. How to run the full pipeline

High-level sequence:

1. **Install:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download and prepare dataset:**

   ```bash
   python download_dataset.py
   ```

   This automatically downloads the AISegment.com Matting Human Datasets and
   organizes them into `data/train/` and `data/val/` folders.

3. **Train (with timelapse frames):**

   ```bash
   python backend/train.py --data-root data --epochs 20 --timelapse
   ```

4. **Generate timelapse videos:**

   ```bash
   python backend/timelapse.py --frames-root outputs/timelapse --out-dir outputs/videos --format gif
   ```

5. **Run inference on a single image (optional):**

   ```bash
   python backend/inference.py path/to/image.jpg \
     --checkpoint outputs/checkpoints/best_model.pth \
     --out-dir outputs/images
   ```

6. **Launch the Streamlit app:**

   ```bash
   streamlit run frontend/app.py
   ```

## 10. Notes for academic presentation

When presenting this project, you can highlight:

1. **Problem definition**
   - Background removal as a pixel-wise binary segmentation problem.
   - Importance in virtual backgrounds, photo editing, AR, etc.

2. **Model choice (U-Net)**
   - Encoder–decoder structure with skip connections.
   - Trade-off between model capacity and real-time inference speed.

3. **Data processing**
   - Handling of arbitrary input resolutions via resize to 512×512.
   - Data augmentation (flips, slight rotations) to improve generalization.

4. **Loss and metrics**
   - Why combine BCE + Dice (stability + overlap quality).
   - Use of IoU, Dice, and pixel accuracy to evaluate segmentation.

5. **Results & visualization**
   - Qualitative examples of input / mask / background-removed outputs.
   - Timelapse videos illustrating convergence over training epochs.

6. **Limitations & future work**
   - Failure cases (e.g., very complex backgrounds, small objects).
   - Possible extensions: MODNet-style trimap, refinement network, 
     more advanced backbones (e.g., ResNet, MobileNet), or knowledge
     distillation for faster inference.

## 11. Reproducibility checklist

- [x] Fixed project structure with separate frontend and backend.
- [x] Deterministic input size (512×512) and normalization.
- [x] Clearly defined training/evaluation scripts (`backend/train.py`).
- [x] Defined metrics (IoU, Dice, Pixel Accuracy).
- [x] Local deployment with Streamlit UI (`frontend/app.py`).
- [x] Timelapse visualization pipeline (`backend/timelapse.py`).

This completes a full deep learning pipeline for **image background removal**
from dataset and training to evaluation, visualization, and local deployment.