from typing import Tuple
import numpy as np
import pandas as pd


class BestExecutionEnv:
    def __init__(self, data, look_back=60):
        """Inicialización de la clase del entorno que simula
        el libro de ordenes.
        ----------------------------------------------------
        Input:
            - data:
                Dataframe con los datos previamente
                agrupados del libro de órdenes.

            - look_back:
                Ventana para la generación de features
                roladas en el instante t=0 del episodio.
                Esta ventana representa el rango máximo para
                la construcción de features.
        ----------------------------------------------------
        Variables Internas:
            - episode_bins:
                Número de bines (steps) del episodio.
            - episode_full_len:
                Es igual a look_back + episode_bins.
            - vol_care:
                Volúmen total (en títulos) de la orden care.
            - actions_fn:
                Diccionario con las posibles acciones del agente.
                Las claves acceden a la función que evalúa  la acción
                tomada por el agente.
            - n_actions:
                Número de acciones posibles.
            - n_features:
                Número de características de los estados.
            - episode:
                Dataframe que contiene los steps y estados del episodio.
            - episode_full:
                Es el episode añadiendo el look_back antes del comienzo
                del episodio.
            - episode_vwap:
                VWAP de mercado al final del episodio.
            - market_ep_vol:
                Volumen (títulos) ejecutado por el mkt por bin del episodio.
            - state_pos:
                Número de step en el que nos encontramos.
            - exec_vol:
                Acumulado de títulos ejecutados por el algoritmo.
            - action_hist:
                Lista de acciones tomadas por el algoritmo en cada step.
            - market_vwap_hist:
                Lista de VWAP de mercado en cada step.
            - reward_hist:
                Lista de rewards obtenidas en cada step.
            - price_hist:
                Lista de precios ejecutados en cada step.
            - vol_hist:
                Lista de títulos ejecutados en cada step.
        """

        # Fixed params
        self.data = data
        self.look_back = look_back
        self.episode_bins = None
        self.episode_full_len = None
        self.vol_care = None

        self.actions_fn = {
            0: self._do_nothing,
            1: self._agg_action,
        }

        self.n_actions = len(self.actions_fn)
        self.n_features = self._detect_num_feat()

        # Data variables
        self.episode = None
        self.episode_full = None

        # Env variables
        self.episode_vwap = None
        self.market_ep_vol = None
        self.state_pos = 0
        self.exec_vol = 0
        self.actions_hist = []
        self.algo_vwap_hist = []
        self.market_vwap_hist = []
        self.reward_hist = []
        self.price_hist = []
        self.vol_hist = []
        self.action_hist = []
        self.price = None
        self.vol = None
        self.obs_hist = []

    def _detect_num_feat(self):
        """Detecta el número de variables del estado.
        Función necesaria para adaptarse automaticamente a los
        cambios en las variables de observation_builder.
        """
        self._reset_env_episode_params()
        self._generate_episode()
        s = self.observation_builder()
        return s.shape[0]

    def _reset_env_episode_params(self):
        """
        Reset del episodio e inicialización de los parámetros.
        Las variables internas vuelven a sus valores originales.
        """
        self.episode_full_len = None
        self.episode = None
        self.episode_full = None
        self.episode_vwap = None
        self.market_ep_vol = None
        self.state_pos = 0
        self.algo_vwap = 0
        self.exec_vol = 0
        self.actions_hist = []
        self.algo_vwap_hist = []
        self.market_vwap_hist = []
        self.reward_hist = []
        self.price_hist = []
        self.vol_hist = []
        self.obs_hist = []
        self.action_hist = []

    def _generate_episode_params(self):
        """Se determinan las características de la orden a ejecutar.
        La órden queda definida por:
         - episode_bins:
             Obtención de un número entero aleatorio [400, 600]
             con una distribución uniforme.
         - vol_care:
             Obtención del porcentaje de steps en el que hay que
             ejecutar una órden para cubrir la órden care.
             vol_care responde a un valor uniforme [0.075, 0.125]
             multiplicado por el número self.episode_bins.
             Lo convertimos a entero.
        """
        # TODO: Int aleatorio entre 400 y 600 como un objeto numpy
        self.episode_bins = np.random.randint(low=400, high=600)
        # TODO: Float aleatorio entre 0.075 y 0.125
        pct_bins = np.random.uniform(low=0.075, high=0.125)
        # TODO: Int multiplicacion pct_bins y episode_bins
        self.vol_care = int(pct_bins * self.episode_bins)

        self.episode_full_len = self.episode_bins + self.look_back

        assert self.episode_bins <= 600
        assert self.episode_bins >= 400
        assert self.vol_care <= int(self.episode_bins * 0.125)
        assert self.vol_care >= int(self.episode_bins * 0.075)
        assert isinstance(self.vol_care, int)

    def _generate_episode(self):
        """Obtenemos el día y hora en el que comienza el episodio.
        """
        self._generate_episode_params()

        lenght_episode = 0
        while lenght_episode != self.episode_full_len:
            # TODO: Selección de un dia entre los posibles.
            # Clue: Usa np.random.choice y los dias data.keys
            selected_day = np.random.choice(
                    list(self.data.keys())
                )

            # TODO: Extrae selected_day de data
            data_day = self.data[selected_day]

            # TODO: selecciona una hora de inicio aleatoria
            init_time = np.random.choice(data_day.index)

            hour_pos = data_day.index.get_loc(init_time)
            initial_position = hour_pos - self.look_back
            final_position = hour_pos + self.episode_bins

            if initial_position < 0:
                continue
            else:
                # TODO: Filtra data_day entre por init_pos y fin_pos
                self.episode_full = data_day.iloc[
                    initial_position:final_position, :
                ]

                # TODO: Filtra data_day entre por hour_pos y final_position
                self.episode = data_day.iloc[hour_pos:final_position, :]

                lenght_episode = self.episode_full.shape[0]

    def reset(self) -> np.array:
        """Reinicialización del episodio junto con los parámetros.
        Devuelve la primera observación del nuevo episodio.
        """
        self._reset_env_episode_params()
        self._generate_episode()
        self._compute_episode_market_feat()
        obs = self.observation_builder()
        self.obs_hist.append(obs)

        return obs

    def observation_builder(self) -> np.array:
        """ Función para la construcción de las observaciones del estado.
            ------------------------------------------------------------
            Default:
                - Primera característica es tiempo restante en porcentaje.
                - Seguna característica es el volumen restante en porcentaje.
        """
        # TODO: Construye el vect con las dos features de la descripción
        # Clue: Utiliza episode_bins, state_pos, exec_vol ,vol_care
        time_left = (self.episode_bins - self.state_pos) / self.episode_bins
        vol_left = 1 - (self.exec_vol / self.vol_care)
        obs = np.array([time_left, vol_left])
        return obs

    def _compute_episode_market_feat(self) -> Tuple[float, float]:
        """Cáculo de los valores VWAP y Market Vol del episodio.
        Como no tenemos las ejecuciones de mercado, asumimos que el
        precio es el mid price de cada step.
        """
        # TODO: Calcula el mid price utilizando ask1 y bid1 de episode
        # Opcional: Utiliza un precio más realista para el mkt VWAP
        mid = (self.episode["ask1"] + self.episode["bid1"]) / 2
        # TODO: Calcula market_ep_vol
        self.market_ep_vol = self.episode.cumvol.diff()
        self.market_ep_vol[0] = 0
        # TODO: calcula el volumen acumulado del mercado en todo el episodio
        cum_vol = self.market_ep_vol.sum()
        # TODO: calcula el episode_vwap
        self.episode_vwap = (mid[:-1] * self.market_ep_vol[1:]).sum() / cum_vol

        return self.episode_vwap, self.market_ep_vol

    def _compute_algo_vwap(self) -> float:
        """Cálculo del VWAP del algoritmo hasta el step actual.
        """
        # TODO: Calcula el algo_vwap
        # Clue: utiliza price_hist, vol_hist
        p_arr = np.array(self.price_hist)
        v_arr = np.array(self.vol_hist)
        algo_vwap = np.sum(p_arr * v_arr) / np.sum(v_arr)
        return algo_vwap

    def _compute_reward(self) -> float:
        """Función de diseño de los rewards y penalizaciónes que
        recibe el algoritmo al tomar las acciones.
        --------------------------------------------------------
        Default:
            - El reward es el ratio de la diferencia entre el episode_vwap y
              el precio de la acción tomada, dividido entre episode_vwap.
        """
        # TODO: Establece y devuelve un reward cuando vol == 0
        if self.vol == 0:
            reward = 0
            return reward
        # TODO: Calcula y devuelve el reward cuando vol > 0
        # Clue: Utiliza episode_vwap y price para la reward por defecto
        # Opcional: Utiliza el self y elimina los parámetros de la función
        reward = (self.episode_vwap - self.price) / self.episode_vwap
        return reward

    def _compute_stop_conditions(self) -> Tuple[bool, bool]:
        """Define las condiciones de parada del episodio
        Return:
            Tiempo agotado, orden completada
        """
        # TODO: Calcula uy devuelve las variables de parada en orden
        is_bins_complete = self.state_pos == self.episode_bins
        is_ord_complete = self.exec_vol == self.vol_care
        return is_bins_complete, is_ord_complete

    def _compute_done_reward(self) -> float:
        # TODO: Free style
        _, is_ord_complete = self._compute_stop_conditions()
        rwd_factor = not is_ord_complete
        done_reward = -1 * rwd_factor
        return done_reward

    def _agg_action(self) -> float:
        """Acción agresiva de compra de un título a precio de episode['ask1'].
        Devolvemos el reward asociado a esa acción.
        """
        self.action_hist.append(1)
        # TODO: obtén el precio de la accion agresiva (ask1) en el state_pos
        self.price = self.episode["ask1"].values[self.state_pos]
        # TODO: guarda price en price_hist, añade 1 a exec_vol y  a vol_hist
        self.price_hist.append(self.price)
        self.vol = 1
        self.exec_vol += self.vol
        self.vol_hist.append(self.vol)

        # TODO: utiliza la función apropiada para calcula el algo_vwap
        algo_vwap = self._compute_algo_vwap()
        # guarda el algo_vwap en algo_vwap_hist
        self.algo_vwap_hist.append(algo_vwap)
        # TODO: calcula el reward utilizando la función apropiada
        reward = self._compute_reward()
        return reward

    def _do_nothing(self) -> float:
        """No hacer nada y devolvemos el reward asociado a la acción
        """
        self.action_hist.append(0)
        # TODO: Repite el proceso de _agg_action
        # Clue: Precio y volumen ejecutado = 0
        self.price = 0
        self.vol = 0
        self.price_hist.append(self.price)
        self.vol_hist.append(self.vol)
        algo_vwap = self.algo_vwap_hist[-1]
        self.algo_vwap_hist.append(algo_vwap)
        reward = self._compute_reward()

        return reward

    def _compute_market_vwap(self) -> float:
        """Cálculo del VWAP del mercado hasta el step actual.
        """
        # TODO: Establece un para el vol ejecutado por el mkt en cada step
        # Clue: puedes fijarte en _compute_episode_market_feat
        mid_p = (self.episode["ask1"] + self.episode["bid1"]) / 2
        mkt_p = (mid_p + mid_p.shift(-1).ffill()) / 2
        # Calcula todos los vwap del mkt hasta el step actual incluido
        v = self.episode["cumvol"].diff().shift(-1)
        p_arr = mkt_p.values[:self.state_pos + 1]
        v_arr = v.values[:self.state_pos + 1]
        sum_vol = np.sum(v_arr)
        # Si mkt vol hasta el step es 0,usa el último precio hasta el step
        if sum_vol == 0:
            return p_arr[-1]
        # Calcula y devuelve el vwap acumulado hasta el step
        market_vwap = np.sum(p_arr * v_arr) / sum_vol
        return market_vwap

    def _compute_done(self) -> bool:
        """ Reglas de finalización del episodio.
        """
        # TODO: Calcula las condiciones de parada con la función adecuada
        conditions = self._compute_stop_conditions()
        is_bins_complete = conditions[0]
        is_ord_complete = conditions[1]
        # TODO: Devuelve done==True si se cumplen cualquiercondiciones
        done = is_bins_complete or is_ord_complete
        return done

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        """ Evalua la accián, calcula la recompensa, devuelve el
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

    def action_sample(self):
        """
        Devuelve una acción aleatoria. El valor ha de corresponder
        con las keys de actions_fn.
        """
        # TODO: Toma una acción aleatoria
        # Opcional: ¿Qué distribución de prob es mejor para la exploración?
        p = self.vol_care / self.episode.shape[0]
        action = np.random.choice([0, 1], p=[1-p, p])
        return action

    def stats_df(self):
        """Información para el gráfico de resultados de la ejecución
        """

        my_df = pd.DataFrame(
            {"vwap": self.algo_vwap_hist, "vol": self.vol_hist},
            index=list(self.episode.index)[:len(self.algo_vwap_hist)]
        )
        my_df = my_df.reindex(self.episode.index)
        my_df["vol"] = my_df["vol"].fillna(0)
        my_df["vwap"] = my_df["vwap"].ffill()

        p = self.episode["ask1"]
        v = self.episode["cumvol"].diff().shift(-1)
        last_v = self.episode_full["cumvol"].diff()[-1]
        v.iloc[-1] = last_v
        market_vwap = (p * v).cumsum() / v.cumsum()
        market_df = pd.DataFrame(
            {"vwap": market_vwap, "vol": v},
            index=v.index
        )

        mpx = (self.episode["ask1"] + self.episode["bid1"]) / 2

        return my_df, market_df, mpx
