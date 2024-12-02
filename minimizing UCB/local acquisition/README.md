# Minimizing UCB: a Better Local Search Strategy in Local Bayesian Optimization

This repository contains code for the paper [Minimizing UCB: a Better Local Search Strategy in Local Bayesian Optimization](https://openreview.net/forum?id=5GCgNFZSyo&referrer=%5Bthe%20profile%20of%20Zheyi%20Fan%5D(%2Fprofile%3Fid%3D~Zheyi_Fan2)).
Our code implementation extends the [GIBO](https://arxiv.org/abs/2106.11899)'s and the [MPD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/555479a201da27c97aaeed842d16ca49-Abstract-Conference.html) codebase, and more detail can be found in [https://github.com/sarmueller/gibo](https://github.com/sarmueller/gibo), [https://github.com/kayween/local-bo-mpd](https://github.com/kayween/local-bo-mpd).

Please consider citing our paper:
```
@inproceedings{nguyen2022local,
    title = {{Minimizing UCB: a Better Local Search Strategy in Local Bayesian Optimization}},
    author = {Fan, Zheyi and Wang, Wenyu and Ng, Szu Hui and Hu, Qingpei},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024}
}
```

## Installation
Our implementation relies on mujoco-py 0.5.7 with MuJoCo Pro version 1.31.
To install MuJoCo follow the instructions here: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py).

### Conda
Or you can create an anaconda environment called MinUCB using
```
conda env create -f environment.yaml
conda activate MinUCB
```

## Usage
For experiments with synthetic test functions and reinforcement learning problems (e.g. MuJoCo) a command-line interface is supplied.

### Synthetic Test Functions
First generate the needed data for the synthetic test functions.

```
python generate_data_synthetic_functions.py -c ./configs/synthetic_experiment/generate_data_default.yaml
```

Afterwards you can run for instance our method MPD on these test functions.

```
python run_synthetic_experiment.py -c ./configs/synthetic_experiment/LAminUCB_default.yaml -cd ./configs/synthetic_experiment/generate_data_default.yaml
```

### Reinforcement Learning

Run the MuJoCo swimmer environment with the proposed method MPD.

```
python run_rl_experiment.py -c ./configs/rl_experiment/swimmer/LAminUCB.yaml
```

### Custom Objective Functions

Run the Rover trajectory planning function with the proposed method MPD.

```
python run_custom_experiment.py -c ./configs/custom_experiment/LAMinUCB_default.yaml
```
