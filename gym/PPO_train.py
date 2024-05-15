# CartPole-PPO.py

import gym
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from PPO_model import PPO
from PPO_hyperparameters import T_horizon, N_episode, model_path


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20
    writer = SummaryWriter('runs/PPO')  # 初始化TensorBoard

    for n_epi in range(N_episode):
        s = env.reset()
        if isinstance(s, tuple):
            s = s[0]  # 如果返回的是元组，则提取第一个元素作为状态
        done = False
        episode_reward = 0
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated  # 将 terminated 和 truncated 逻辑或

                if isinstance(s_prime, tuple):
                    s_prime = s_prime[0]  # 如果返回的是元组，则提取第一个元素作为状态

                # 存储经验
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))

                s = s_prime

                episode_reward += r
                if done:
                    break

            loss = model.train_net()  # 训练网络

        score += episode_reward
        writer.add_scalar('Loss/train', loss, n_epi)  # 记录损失
        writer.add_scalar('Reward/episode', episode_reward, n_epi)  # 记录每个episode的奖励

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
            writer.add_scalar('Avg_Reward/print_interval', avg_score, n_epi)  # 记录平均奖励
            score = 0.0

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    env.close()
    writer.close()  # 关闭TensorBoard


if __name__ == '__main__':
    main()
