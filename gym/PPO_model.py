import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PPO_hyperparameters import learning_rate, gamma, lmbda, eps_clip, K_epoch, batch_size


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        # 定义神经网络层
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        # 策略网络，输出动作的概率分布
        x = torch.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = torch.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        # 值网络，输出状态值
        x = torch.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        # 存储经验
        self.data.append(transition)

    def make_batch(self):
        # 将存储的经验转换为训练所需的批次
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []  # 清空存储的数据
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        dataset = torch.utils.data.TensorDataset(s, a, r, s_prime, done_mask, prob_a)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(K_epoch):
            for batch in data_loader:
                s, a, r, s_prime, done_mask, prob_a = batch

                # 计算TD目标
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
                delta = delta.detach().numpy()

                # 计算优势函数
                advantages = []
                advantage = 0.0
                for idx in reversed(range(len(delta))):
                    advantage = gamma * lmbda * advantage + delta[idx][0]
                    advantages.append([advantage])
                advantages.reverse()
                advantages = torch.tensor(advantages, dtype=torch.float)

                # 计算策略和值函数的损失
                pi = self.pi(s, softmax_dim=1)
                pi_a = pi.gather(1, a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        return loss.mean().item()  # 返回损失值
