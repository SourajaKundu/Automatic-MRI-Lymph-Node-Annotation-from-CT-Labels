# Automatic MRI Lymph-Node Annotation from CT Labels

**Official Implementation**

This repository contains the official implementation of our work on **cross-modality medical image translation, deformable registration, and automatic MRI lymph-node annotation**. Across three publications, we introduce: (1) an improved Residual Vision Transformer GAN for high-fidelity CT→MRI synthesis, (2) a Transformer-based local–global correlation model for accurate cross-modality medical image registration, and (3) a complete three-stage pipeline for generating MRI lymph-node annotations using only CT labels. Together, these works advance the reliability of synthetic MRI generation, the precision of CT–MRI alignment, and the automation of lymph-node labeling in clinical imaging.

---

## 📚 Publications

### **1) Improved Residual Vision Transformer for CT→MRI Translation**

Souraja Kundu, Yuji Iwahori, Manas Kamal Bhuyan, Manish Bhatt, Akira Ouchi, Yasuhiro Shimizu
*TransAI 2023, Oral Presentation, LA, USA*
[https://ieeexplore.ieee.org/document/10387640](https://ieeexplore.ieee.org/document/10387640)

### **2) Cross-Modality Medical Image Registration with Local-Global Spatial Correlation**

Souraja Kundu, Yuji Iwahori, Manas Kamal Bhuyan, Manish Bhatt, Boonserm Kijsirikul, Aili Wang, Akira Ouchi, Yasuhiro Shimizu
*ICPR 2024, Oral (Top 6%), Kolkata, India*
[https://link.springer.com/chapter/10.1007/978-3-031-78195-7_8](https://link.springer.com/chapter/10.1007/978-3-031-78195-7_8)

### **3) Automatic MRI Lymph-Node Annotation from CT Labels**

Souraja Kundu, Yuji Iwahori, M.K. Bhuyan, Manish Bhatt, Boonserm Kijsirikul, Aili Wang, Akira Ouchi, Yasuhiro Shimizu
*IEEE Access 2025*
[https://ieeexplore.ieee.org/document/10855406](https://ieeexplore.ieee.org/document/10855406)

---

## Overview

This project demonstrates a three-step pipeline for automatic annotation of MRI lymph nodes using paired CT images. Due to restricted data permissions, the dataset is not provided. However, this repository contains all necessary scripts to train and test the image registration model and to transfer lymph node annotations from CT to MRI.

### Highlights
- Multi-modal image registration using a discriminator-free image-to-image translation approach.
- Lymph node position transfer from CT to MRI using a registered mapping.
- Adapted from [DFMIR](https://github.com/heyblackC/DFMIR): "Unsupervised Multi-Modal Medical Image Registration via Discriminator-Free Image-to-Image Translation".

## Dataset Structure

The dataset should be organized as follows:
```
Dataset/
├── trainA/  # Modality A training images
├── trainB/  # Modality B training images
├── valA/    # Modality A validation images
├── valB/    # Modality B validation images
├── testA/   # Modality A test images
├── testB/   # Modality B test images
```

Ensure the folder paths align with the argument `--dataroot` in the scripts.

## Requirements

### Dependencies
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

Key packages include:
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-image
- tqdm

### Environment
Tested on:
- Python 3.8+
- CUDA-enabled GPUs

## Instructions

### Training
To train the image registration model, run:
```bash
python3 train.py \
    --dataroot /path/to/Dataset \
    --input_nc 3 \
    --crop_size 512 \
    --load_size 512 \
    --output_nc 3 \
    --netG stylegan2
```
Additional options can be explored in `options/train_options.py`.

### Testing
To test the trained registration model, run:
```bash
python3 test.py \
    --dataroot /path/to/Dataset \
    --input_nc 3 \
    --crop_size 512 \
    --load_size 512 \
    --output_nc 3 \
    --netG stylegan2
```

### Annotation Transfer
The second step of transferring lymph node positions from CT to MRI can be found in the notebook `Loss Plots DSC Calculation.ipynb`. 

## Checkpoints
Trained model weights are saved in the `checkpoints/` directory. You can use the same weights for both registration and de-registration in steps 1 and 3, after training only once.

## Changing model to DFMIR
The few changes that need to be done are 
1. Use the PatchSampleFDFMIR function instead of PatchSampleF in models/networks.py
2. Remove loss_super_resolution from self.loss_R in class REGISTRATIONModel of models/registration_model.py
3. Use the UnetDFMIR class instead of Unet in models/voxelmorph/torchvoxelmorph/networks.py  

## Citation
If you use this code, please cite the following papers:

1. Chen, Zekang, Jia Wei, and Rui Li. "Unsupervised multi-modal medical image registration via discriminator-free image-to-image translation." *arXiv preprint arXiv:2204.13656* (2022).  
   - GitHub Repo: [DFMIR](https://github.com/heyblackC/DFMIR)
2. Kundu, Souraja, Yuji Iwahori, M. K. Bhuyan, Manish Bhatt, Boonserm Kijsirikul, Aili Wang, Akira Ouchi, and Yasuhiro Shimizu. "Cross-Modality Medical Image Registration with Local-Global Spatial Correlation." In *International Conference on Pattern Recognition*, pp. 112-126. Springer, Cham, 2025.

## License
This project is distributed under the MIT License. See `LICENSE` for more information.

## Contact
For any questions or issues, please contact:
- **Souraja Kundu**
- Email: [sourajaakundu@gmail.com]
