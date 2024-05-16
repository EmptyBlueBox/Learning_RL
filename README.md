# Learning Reinforcement Learning

## Environment

1. `cd Isaac-Gym`
2. `conda env create -f ./RL_conda_env.yml`
3. Download IssacGym and put it to `Isaac-Gym`, then `pip install -e ./IsaacGym`
4. Download `https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git` and put it to `Isaac-Gym`, then `pip install -e ./IsaacGymEnvs`
5. `python test_env.py`
6. If you come across fault like:
    ```
    ImportError: libpython3.7m.so.1.0: cannot open shared object file:.....
    ```
    Put `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/*/miniconda3/envs/RL/lib/libpython3.7m.so.1.0` to `.zshrc` or `.bashrc`

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
