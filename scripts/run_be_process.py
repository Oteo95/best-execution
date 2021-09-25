from agents.dqn import DDQNAgent
from be_env.twap_env import TwapEnv
from be_env.train import TrainTabularAgent
import pickle

"""
    Agent Params
"""
epsilon = 1
min_epsilon = 0.05
gamma = 1
alpha = 0.0001
buffer_size = 10000
batch_size = 256
hidden_neurons = 240

"""
    Training Params
"""
nepisodes = 100
n_log = 25
epsilon_decay = (epsilon - min_epsilon) / (nepisodes * 0.95)
learn_after = batch_size

with open("./data/rep_data.pickle", "rb") as f:
    df = pickle.load(f)
data = df["train"]

env = TwapEnv(data, 60)
agent = DDQNAgent(
    env, gamma=gamma, epsilon=epsilon, alpha=alpha,
    batch_size=batch_size, buffer_size=buffer_size,
    hidden_neurons=hidden_neurons, trainable=True
)

tba = TrainTabularAgent(
    agent=agent,
    env=env,
    buffer=buffer_size,
    nepisodes=nepisodes,
    n_log=n_log
)

tba.fill_buffer()
tba.run_process(
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon,
    learn_after=learn_after
)
