## Installation
### Requirements
- Python 3.7
- Pytorch 1.2
- torchvision 0.4.0
- cuda 10.0

### Setup with Conda
```bash
# create a new environment
conda create --name oscar python=3.7
conda activate oscar

# install pytorch1.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# install oscar submodules
cd ../../
git submodule init
git submodule update
cd models/context_compat_oscar/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt
```

