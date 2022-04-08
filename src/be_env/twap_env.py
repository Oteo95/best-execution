from be_env.base_env import BestExecutionEnv
import numpy as np
from typing import Tuple


class TwapEnv(BestExecutionEnv):
    def __init__(self, data, look_back):
        super().__init__(data, look_back)
        self.data = data
        self.look_back = look_back

    def _compute_reward(self) -> float:
        obs = self.observation_builder()
        a = self.action_hist[-1]

        if obs[1] >= obs[0] and a == 1:
            return 2

        elif obs[1] >= obs[0] and a == 0:
            return -1

        elif obs[1] < obs[0] and a == 0:
            return 0.1
        else:
            return -1

    def _compute_done_reward(self) -> float:
        # TODO: Free style
        _, is_ord_complete = self._compute_stop_conditions()
        rwd_factor = not is_ord_complete
        done_reward = 0
        if rwd_factor:
            done_reward -= self.vol_care
        done_reward += -(self.episode_bins - self.state_pos)
        return done_reward

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        """ Evalua la acción, calcula la recompensa, devuelve el
        nuevo estado y si el episodio ha terminado.
        """

        market_vwap = self._compute_market_vwap()
        act_fn = self.actions_fn.get(action)
        if act_fn is None:
            raise ValueError(
                f"Invalid {action}. Valid actions {self.actions_fn.keys()}"
            )

        reward = act_fn()

        self.market_vwap_hist.append(market_vwap)
        self.reward_hist.append(reward)

        self.state_pos += 1

        done = self._compute_done()

        if done:
            reward += self._compute_done_reward()
            return None, reward, done, {}

        observation = self.observation_builder()
        self.obs_hist.append(observation)

        return np.array(observation), reward, done, {}
