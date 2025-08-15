<div align="center">
<br>
<h2>InstantEdit: Text-Guided Few-Step Image Editing with Piecewise Rectified Flow</h2>

[Yiming Gong](https://github.com/nickgong1);  [Zhen Zhu](https://zzhu.vision); [Minjia Zhang](https://minjiazhang.github.io/)
<br>

 University of Illinois Urbana-Champaign&nbsp;


[![arXiv](https://img.shields.io/badge/arXiv-2412.01827-A42C25?style=flat&logo=arXiv&logoColor=A42C25)]()
<!-- [![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)]()  -->


<img src="assets/diverse_edit.png" width="100%">  

<img src="assets/main_fig.png" width="100%">

</div>

## Overview
**InstantEdit** is a few-step image editing framework that enables efficient and precise text-guided image editing. Our method builds upon the Piecewise Rectified Flow model for accurate editing in just a few steps. Key features include:

- **Training Free**: No fine-tuning required. Use the model out of the box for immediate editing.
- **Precise Control**: Achieves better editability while maintains the consistency of image comparing to counterpart few-step editing methods.
- **Versatile Applications**: Supports various editing tasks including object manipulation, style transfer, and attribute modification


## Getting Started

### Setup
1. Clone the repository
```
git clone https://github.com/Supercomputing-System-AI-Lab/InstantEdit.git
cd InstantEdit
```
2. Create and activate the Conda environment:
```
conda env create -f environment.yaml
conda activate InstantEdit
```
### Run PIE Evaluation
1. Download PIE Benchmark  
You can download the PIE benchmark from [here](https://docs.google.com/forms/d/e/1FAIpQLSftGgDwLLMwrad9pX3Odbnd4UXGvcRuXDkRp6BT1nPk8fcH_g/viewform) by filling in the form from the original authors.

2. A minimal script to run
```
python instantedit.py --dataset_path $Your PIE dataset path$
```
**Optional arguments**
-   `--num_inference_steps`  
    Number of inversion and inference steps
-   `--mask_threshold`  
    Mask threshold for attention masking
-   `--controlnet_conditioning_scale`  
    Control strength for ControlNet
-   `--dpg_weight`  
    DPG weight to control editing strength
-   `--cfg_weight`  
    CFG weight to control editing strength

### Run Per-image editing 
We prepare a gradio demo so you can upload your own images and editing prompts. Simply follow the instructions on the GUI when you open gradio url.
```
python demo.py$ 
```

## Bibtex
```
@inproceedings{gong2025instantedit,
  title        = {InstantEdit: Text-Guided Few-Step Image Editing with Piecewise Rectified Flow},
  author       = {Gong, Yiming and Zhu, Zhen and Zhang, Minjia},
  booktitle    = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year         = {2025}
}
```

## Acknowledgement
- Part of our code is adopted from the implementation of [P2P](https://github.com/google/prompt-to-prompt) and [InfEdit](https://github.com/sled-group/InfEdit). We thank the authors for releasing their code.
- We reuse the PIE evaluator from [DirectInversion](https://github.com/cure-lab/PnPInversion)



