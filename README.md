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

## Results