# Pred2Nav

This repository contains the codes related to our works: "From Crowd Prediction Models to Robot Navigation in Crowds" in IROS 2023, and "Winding Through: Crowd Navigation via Topological Invariance" in R-AL 2023 and ICRA 2023. 
For more details, please refer to the [arXiv](https://arxiv.org/abs/2303.01424).
For experiment demonstrations, please refer to the [youtube video](#).

The framework is based on the implmenetation of a crowd navigation simulator based on the work [here](https://github.com/Shuijing725/CrowdNav_DSRNN), as compared the propietory Gazebo based Honda ballbot simulator demonstrated in our paper.


## Setup
1. Install Python3.6 (The code may work with other versions of Python, but 3.6 is highly recommended).
2. Install the required python package using pip or conda. For pip, use the following command:  
```
pip install -r requirements.txt
```
For conda, please install each package in `requirements.txt` into your conda environment manually and 
follow the instructions on the anaconda website.  

3. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library.  


## Getting started
This repository is organized in three parts: 
- `crowd_sim/` folder contains the simulation environment. Details of the simulation framework can be found
[here](crowd_sim/README.md).
- `crowd_nav/` folder contains configurations and non-neural network policies
- `crowd_nav/policy/vecMPC` contains the code for the model predictive controller with different prediction models.  
 
Below are the instructions for training and testing policies.

### Change configurations
1. Environment configurations and training hyperparameters: modify `crowd_nav/configs/config_vecMPC.py`

### Run the code

1. Test policies.  
Please modify the test arguments in the begining of `evaluation_mpc.py`.     
We provide sample configs for different prediction models in `crowd_nav/configs/params`.

```
python run_eval.py 
```



<!-- ## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517-3524}
}
``` -->

## Credits

Part of the code is based on the following repositories:  

[1] S. Liu*, P. Chang*, W. Liang†, N. Chakraborty†, and K. Driggs-Campbell, "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning," in International Conference on Robotics and Automation (ICRA), 2021, pp. 3517-3524. (Github: https://github.com/Shuijing725/CrowdNav_DSRNN)

[2] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[3] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

[4] A. Vemula, K. Muelling, and J. Oh, “Social attention: Modeling attention in human crowds,” in IEEE international Conference on Robotics and Automation (ICRA), 2018, pp. 1–7.
(Github: https://github.com/jeanoh/big)

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.%                                                                                                             
