import gym
import torch
from torch.distributions import Categorical
import imageio
from PPO_model import PPO
from PPO_hyperparameters import model_path


def visualize_model(model_path):
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    model = PPO()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    s = env.reset()
    if isinstance(s, tuple):
        s = s[0]  # 如果返回的是元组，则提取第一个元素作为状态
    done = False
    episode_reward = 0
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        prob = model.pi(torch.from_numpy(s).float())
        m = Categorical(prob)
        a = m.sample().item()
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        episode_reward += r

    env.close()
    print(f"Episode reward: {episode_reward}")

    # 保存视频
    save_frames_as_gif(frames, 'ppo_cartpole.gif')


def save_frames_as_gif(frames, path):
    imageio.mimsave(path, frames, fps=30)
    print(f"Video saved to {path}")


if __name__ == '__main__':
    visualize_model(model_path)
