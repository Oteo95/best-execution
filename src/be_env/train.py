from report.report import plot_results
import numpy as np


class TrainTabularAgent:

    def __init__(self, agent, env, buffer, nepisodes=1000, n_log=25):
        self.buffer_size = buffer
        self.nepisodes = nepisodes
        self.n_log = n_log
        self.agent = agent
        self.env = env

    def fill_buffer(self, strategy: str) -> None:
        """ Buffer experiences to allow TD value methods
        to learn during the exploration and training process.

        Args:
            strategy: Method to fill the buffer. Can be 'random'
                or 'twap'.
        """
        s = self.env.reset()
        for exps in range(self.buffer_size):
            if strategy == "random":
                a = self.agent.act(s)
            elif strategy == "twap":
                a = self.buffer_act_twap(s)
            s1, r, done, _ = self.env.step(a)
            self.agent.experience(s, a, r, s1, done)
            s = s1

            if not exps % 10000:
                print(f'buffer exps: {exps}')
            if done:
                s = self.env.reset()

    def buffer_act_twap(self, s: np.array) -> int:
        if s[1] >= s[0]:
            return 1
        return 0

    def run_process(self, epsilon_decay, min_epsilon, learn_after):
        self.agent.set_trainable(True)
        learn_counter = 0
        history_steps = []
        history_rewards = []
        history_disc_rewards = []
        history_losses = []

        for episode in range(self.nepisodes):
            s = self.env.reset()
            step = 0
            cum_reward = 0
            dis_cum_reward = 0
            while True:
                a = self.agent.act(s)
                s1, r, done, _ = self.env.step(a)
                self.agent.experience(s, a, r, s1, done)
                learn_counter += 1
                cum_reward += r
                dis_cum_reward += self.agent.gamma ** step * r
                s = s1
                step += 1
                if not learn_counter % learn_after:
                    mse = self.agent.learn()
                if done:
                    self.agent.epsilon = max(
                        [self.agent.epsilon - epsilon_decay, min_epsilon]
                        )
                    history_rewards.append(cum_reward)
                    history_disc_rewards.append(dis_cum_reward)
                    history_losses.append(mse)
                    history_steps.append(step)

                    if not episode % self.n_log:
                        mse = self.agent.learn()
                        val = np.round(
                            np.mean(history_steps[-self.n_log:]),
                            2
                        )
                        val2 = np.round(
                            np.mean(history_rewards[-self.n_log:]),
                            2
                        )
                        print(
                            f'Episode: {episode}, '
                            f'steps: {val}, '
                            f'rew: {val2}, '
                            f'mse: {np.round(mse)}, '
                            f'eps: {np.round(self.agent.epsilon, 2)}'
                        )
                    break

    def plot_policy_results(self, data):
        self.agent.set_trainable(False)
        cum_reward = 0
        step = 0
        self.env.data = data
        s = self.env.reset()
        a = 1
        s, r, done, _ = self.env.step(a)
        step += 1
        cum_reward += self.agent.gamma ** step * r
        while True:
            a = self.agent.act(s)
            s, r, done, _ = self.env.step(a)
            step += 1
            cum_reward += self.agent.gamma ** step * r
            if done:
                break
        plot_results(self.env)
