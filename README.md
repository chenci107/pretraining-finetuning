**Installlation**

```
conda create -n pretrain python=3.8
conda activate pretrain
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
cd isaacgym/python && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

**Usage**

step 1: train the pre-trained model

```
cd scripts
python train.py --exptid xxx
```
play the pre-trained model
```
cd scripts
python play.py --exptid xxx
```
step 2: BO embedded with fine-tune
```
cd legged_gym_bayes
python legged_fine_tune.py --exptid yyy --resumeid xxx
```
**Acknowledgement**

<https://github.com/chengxuxin/extreme-parkour>

