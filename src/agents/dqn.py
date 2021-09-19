import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from collections import deque


class DDQNAgent():

    def __init__(
        self, env, gamma=0.99, epsilon=0.9, alpha=0.01,
        batch_size=64, buffer_size=50000,
        hidden_neurons=32, trainable=True
    ):

        self.env = env
        self.na = env.n_actions
        self.n_features = env.n_features
        self.epsilon = epsilon
        self.gamma = gamma
        self.minus_factor = None
        self.div_factor = None

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.hidden_neurons = hidden_neurons

        self.primary_model = self.network()
        self.primary_model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=alpha)
        )
        # Double Deep Q-Learning -> Add an extra model
        self.target_model = self.network()
        self.target_model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=alpha)
        )
        self._trainable = trainable

    def network(self):
        inp = Input(shape=(self.n_features, ))
        capa = Dense(int(self.hidden_neurons), activation="relu")(inp)
        capa = Dense(int(self.hidden_neurons), activation="relu")(capa)
        out = Dense(self.na, activation="linear")(capa)

        return Model(inp, out)

    def experience(self, s, a, r, s1, done=False):
        if done:
            s1 = None
        self.buffer.append((s, a, r, s1))

    def sample(self):
        # CER
        idx = np.hstack([np.random.choice(
                            (len(self.buffer)-1),
                            size=self.batch_size - 1,
                            replace=False), (len(self.buffer)-1)])

        return [self.buffer[i] for i in idx]

    @staticmethod
    def getQs(s, model):
        return model(s).numpy()

    def act(self, s):
        if self.env.state_pos == 0:
            return 1
        if self._trainable:
            if np.random.rand() > self.epsilon:
                a = self._act_no_explore(s)
            else:
                a = self.env.action_sample()
        else:
            a = self._act_no_explore(s)
        return a

    def set_trainable(self, train=False):
        self._trainable = train

    def _act_no_explore(self, s):
        q = self.getQs(
            s=s.reshape(1, -1),
            model=self.primary_model
        )
        return np.argmax(q)

    def getQ(self, s, a):
        return self.getQs(s)[:, a]

    def switchmodels(self):
        self.primary_model, self.target_model = (
            self.target_model, self.primary_model
        )

    def learn(self):
        batch = self.sample()
        s_list = np.array([experience[0].tolist() for experience in batch])
        s1_list = np.array([
            experience[3].tolist() for experience in batch
            if experience[3] is not None
        ])
        qs = self.getQs(s_list, self.primary_model)
        qs1 = self.getQs(s1_list, self.primary_model)
        q_target = self.getQs(s1_list, self.target_model)
        targets = qs

        k = 0
        for i, (s, a, r, s1) in enumerate(batch):
            if s1 is None:
                targets[i][a] = r
            else:
                best_next_action = np.argmax(qs1[k])
                targets[i][a] = r + self.gamma * q_target[k, best_next_action]
                k += 1

        loss = self.primary_model.train_on_batch(s_list, targets)

        self.switchmodels()

        return loss
