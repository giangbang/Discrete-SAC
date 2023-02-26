# Discrete-SAC
Discrete action version of SAC with Pytorch

## Installation
```
git clone https://github.com/giangbang/Discrete-SAC.git
cd Discrete-SAC
pip install -r requirements.txt
```
or it can be installed with `pip`
```
pip install git+https://github.com/giangbang/Discrete-SAC.git
```

## Training
```
python train.py --env_name LunarLander-v2 --total_env_step 500000
```
If the repo is installed by `pip`, then it can be run with the command
```
sac_discrete --env_name LunarLander-v2 --total_env_step 500000
```
For `atari` environments, they need to be installed separately via command `pip install gymnasium[atari]`. Terminal arguments with the training of `atari` environments need to specify a flag `--atari True` to properly setup some preprocessing steps generally done, e.g frame skipping, grayscaling and resizing.
```
python train.py --env_name ALE/Gravitar-v5 --atari True --total_env_step 1_000_000 --target_entropy_ratio 0.3 --eval_interval 5000
```
## Results
