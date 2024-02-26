## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### Building on another system
To build on a system that does not have a GPU device but provide the drivers:
```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### Example conda environment setup
```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

# Install the detectron 2 b/c we don't have a good pre-built binary
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install cudatoolkit dev for nvcc command to work
conda install -c conda-forge cudatoolkit-dev 

pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

# Install mmcv-full
pip install -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.9.0/index.html mmcv-full==1.3.17

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Data preparation
- Activate the virtual environment `conda activate mask2former`.
- Set the environment path `export DETECTRON2_DATASETS=/jupyter-users-home/tan-2enguyen/datasets/detectron2`
- Go to the following [page](https://github.com/thnguyn2/Mask2Former/blob/main/datasets/README.md) to prepare the dataset.
- For the ADE20k dataset, `cd /jupyter-users-home/tan-2enguyen/Mask2Former`. Run `prepare_ade20k_sem_seg.py`, `prepare_ade20k_pan_seg.py`, and 

### Things that worked perfectly for me
- Finished data prep the COCO dataset.
- Finished data prep for the ADE20k dataset

### Things that goes wrong
- `dtype` error in the training with amp.
- `Assertion error` from the evaluation

