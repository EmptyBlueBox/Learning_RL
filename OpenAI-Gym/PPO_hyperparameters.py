N_episode = 1000
K_epoch = 3
T_horizon = 10
batch_size = 8

learning_rate = 1e-4
gamma = 0.99
lmbda = 0.95
eps_clip = 0.1

use_buffer = 3 # 是否使用之前的数据
max_buffer_dis = 5 # buffer 最长距离

model_name = 'PPO_model'
visualize_folder = './PPO_result'
visualize_suffix = 'no_buffer'