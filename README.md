# Learning Reinforcement Learning

## Environment

1. `cd Isaac-Gym`
2. `conda env create -f ./RL_conda_env.yml`
3. `pip install 'gym[classic_control]'`
4. `pip install 'gym[all]'`
5. Download IssacGym and put it to `Isaac-Gym`, then `pip install -e ./IsaacGym`
6. Download `https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git` and put it to `Isaac-Gym`, then `pip install -e ./IsaacGymEnvs`
7. `python test_env.py`
8. If you come across fault like:
    ```
    ImportError: libpython3.7m.so.1.0: cannot open shared object file:.....
    ```
    Put `export LD_LIBRARY_PATH=/home/(Your_username)/miniconda3/envs/(Your_new_env_name)/lib/` to `.zshrc` or `.bashrc`, then `source .zshrc` or `source .bashrc`

## Run

1. OpenAI Gym
    1. Run `PPO_train.py` to train
    2. Run `PPO_visualize.py` to visualize (test)
    3. If you come across fault like:
        ```
        ModuleNotFoundError: No module named 'pygame'
        ```
        Run:
        `pip install 'gym[classic_control]'`
    4. Fault: 
        ``` AttributeError: module 'gym.envs.box2d' has no attribute 'LunarLander'```
        Do:
        `pip install Box2D`
2. Issac Gym
    1. `cd Isaac-Gym/IsaacGymEnvs/isaacgymenvs`
    2. `python train.py task=Ant headless=True`
    3. If you come across fault like:
        ```
        File "/home/*/miniconda3/envs/RL/lib/python3.7/site-packages/torch/__init__.py", line 196, in <module>
            from torch._C import *
        ImportError: /home/*/miniconda3/envs/RL/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
        ```
        Run:
        `pip install mkl==2024.0`
