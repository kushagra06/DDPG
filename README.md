# DDPG
Pytorch implementation of Deep Deterministic Policy Gradient (Deep reinfor algorithm) on OpenAI gym's inverted pendulum environment. 
The goal is to swing the pendulum up so it stays upright.  

## Prerequisites:
To run the code, you need to have installed the following libraries/softwares on your system (preferably,Ubuntu or any linux distro):
* python: Required version >= 3.5. Also, install pip using `sudo apt install python3-pip`. (if your package manager is apt)
* PyTorch: Recommeded to install via pip. https://pytorch.org/
* numpy: `pip install numpy`
* jupyter: `pip install jupyter`
* matplotlib: `pip install matplotlib`
* seaborn: `pip install seaborn`
* IPython: `sudo apt install python3-ipython`
* tqdm: `pip install tqdm` 
* OpenAI gym: https://gym.openai.com/docs/

It is recommended to run the code in a virtualenv.

## Running the code:
Install the required softwares and clone this repo. To test the code or perform experiments run a new jupyter session using
```
jupyter notebook
```
on terminal which launches the jupyter notebook app in a browser. In the notebook dashboard, navigate to find your notebook and run it.
To train/test the model, execute 
```
python pendulum.py
```

## Organization:
* [gym_utils.py](https://github.com/kushagra06/SAC/blob/master/gym_utils.py): 
Some utility functions to get parameters of the gym environment used, e.g. number of states and actions.
* [model.py](https://github.com/kushagra06/SAC/blob/master/model.py): Deep learning network for the agent. 
* [replay_buffer.py](https://github.com/kushagra06/SAC/blob/master/replay_buffer.py): A replay buffer to store state-actoin transitions
and then randomly sample from it. 
* [pendulum.ipynb](https://github.com/kushagra06/SAC/blob/master/softac,ipynb): SDDPG implementation in a jupyter notebook for
testing the code and performing experiments. 
* [pendulum.py](https://github.com/kushagra06/SAC/blob/master/softac.py): Implementation of the algorithm for training and testing on the 
task of inverted pendulum (default). 

*The repo is still under construction. To report bugs or add changes, open a pull request.*
