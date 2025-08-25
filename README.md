# DreamBooth + LoRA for Subject and Style Transfer

This repository contains an implementation of **DreamBooth combined with LoRA** to fine-tune **Stable Diffusion** on custom subjects and artistic styles. The following capabilities are included:

* Subject Transfer using DreamBooth + LoRA
* Style Transfer using DreamBooth + LoRA
* Experimental implementation of **ZipLoRA**, which attempts to apply both subject and style transfer simultaneously

The code is organized within a Jupyter Notebook (`dbooth.ipynb`) that guides you through the training process in a sequential manner.

> Note: The inference code for ZipLoRA is not included, and validation images have been omitted due to suboptimal performance. Checkpoints are not pushed to the repository but can be provided upon request.

---

## Repository Structure

```
.
├── data/                        # Processed subject and style image datasets
├── raw_images/                 # Raw input images before preprocessing
├── dbooth.ipynb                # Main notebook for training
├── environment.yaml            # Conda environment file
├── README.md                   # Project description and instructions
```

Only the files and folders listed above are tracked in version control. All outputs, checkpoints, and other intermediate files are ignored via `.gitignore`.

---

## Features

### Subject Transfer

Fine-tunes a base Stable Diffusion model on a custom subject using DreamBooth and LoRA to reduce memory and training time requirements.

### Style Transfer

Fine-tunes the model to learn a specific artistic style. This can then be applied to other images or subjects.

### ZipLoRA (Experimental)

ZipLoRA is an experimental module designed to perform both subject and style transfer in a single training pipeline. The training code is included, but inference is not implemented.

---

## Installation

Create the Conda environment using the provided file:

```bash
conda env create -f environment.yaml
conda activate project_env
```

Install PyTorch and other required libraries:

```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
```

---

## Usage

### Step 1: Prepare Data

1. Choose a subject (e.g., an object or person) and collect multiple images of it.
2. Choose a style (e.g., watercolor, oil painting) and collect relevant style images.
3. Place raw images into the `raw_images/` directory.
4. Run the **Prepare Data** section in `dbooth.ipynb`. This will:

   * Organize images into the correct format under `data/`
   * Generate any necessary class images for prior preservation

Example structure after preparation:

```
data/
├── subject/
│   └── <subject_name>/
│       └── subject_instance_images/
│        └── subject_class_images/ 
├── style/
│   └── <style_name>/
│       └── style_instance_images/
│        └── style_class_images/ 
```

---

### Step 2: Train Models

Open `dbooth.ipynb` and execute cells sequentially:

1. Subject Transfer Training
2. Style Transfer Training
3. Optional: ZipLoRA Training

Training configurations are pre-defined using `sys.argv` format. Each example contains arguments such as:

```python
sys.argv = [
   "--pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5",
   "--instance_data_dir", "./data/subject/bag/subject_instance_images",
   "--style_data_dir", "./data/style/waterpaint/style_instance_images",
   "--instance_prompt", "an image of tqz bag",
   "--style_prompt", "in the style of pqrstyle",
   "--output_dir", "./output/ziplora-model/bag+waterpaint",
   "--resolution", "512",
   "--train_batch_size", "1",
   "--learning_rate", "1e-4",
   "--lr_scheduler", "constant",
   "--num_train_epochs", "100",
   "--lora_rank", "16",
   "--lora_alpha", "32",
   "--validation_prompt", "an image of tqz bag in the style of pqrstyle",
   "--use_8bit_adam",
   "--enable_xformers",
   "--checkpointing_steps", "100",
   "--gradient_accumulation_steps", "1",
   "--gradient_checkpointing",
   "--seed", "42"
]
```

These can be modified based on your subject or style.

---

## Examples Included

* Two examples for subject transfer
* Two examples for style transfer

Each example includes training prompts and configurations. Inference for subject and style transfer is also implemented in the notebook.

---

## Limitations

* Inference code for ZipLoRA is not included
* Validation images for ZipLoRA are not included due to suboptimal results
* Model checkpoints are not pushed to the repository due to size constraints but are available on request

---

