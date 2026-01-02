# import numpy as np
# import os

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# import tensorflow as tf
# from tensorflow.keras import layers
# from collections import deque
# import random


# class DQNAgent:
#     def __init__(self, state_size, action_size):

#         self.state_size = state_size
#         self.action_size = action_size

#         self.gamma = 0.90
#         self.lr = 0.002

#         self.epsilon = 1.0
#         self.epsilon_min = 0.05
#         self.epsilon_decay = 0.97

#         self.memory = deque(maxlen=5000)
#         self.batch_size = 32

#         self.model = self._build_model()

#     def _build_model(self):
#         model = tf.keras.Sequential(
#             [
#                 layers.Input(shape=(self.state_size,)),
#                 layers.Dense(32, activation="relu"),
#                 layers.Dense(32, activation="relu"),
#                 layers.Dense(self.action_size, activation="linear"),
#             ]
#         )
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(self.lr),
#             loss=tf.keras.losses.MeanSquaredError(),
#         )
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.action_size)
#         q_values = self.model.predict(np.array([state]), verbose=0)
#         return np.argmax(q_values[0])

#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return

#         batch = random.sample(self.memory, self.batch_size)

#         states, targets = [], []

#         for state, action, reward, next_state, done in batch:
#             target = reward
#             if not done:
#                 target += self.gamma * np.max(
#                     self.model.predict(np.array([next_state]), verbose=0)[0]
#                 )

#             target_f = self.model.predict(np.array([state]), verbose=0)[0]
#             target_f[action] = target

#             states.append(state)
#             targets.append(target_f)

#         self.model.fit(
#             np.array(states), np.array(targets), batch_size=self.batch_size, verbose=0
#         )

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def save(self, path="rl_treatment_fast.h5"):
#         self.model.save(path)

#     def load(self, path="rl_treatment_fast.h5"):
#         # FIX: compile=False to prevent deserialization errors
#         self.model = tf.keras.models.load_model(path, compile=False)
#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(self.lr),
#             loss=tf.keras.losses.MeanSquaredError(),
#         )


# import os

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# from collections import deque
# import random
# from typing import Tuple

# DEFAULTS = {
#     "gamma": 0.95,
#     "lr": 0.003,
#     "epsilon": 1.0,
#     "epsilon_min": 0.05,
#     "epsilon_decay": 0.96,
#     "memory_size": 10000,
#     "batch_size": 64,
#     "warmup": 200,
#     "target_update_freq": 20,
# }


# class DQNAgent:
#     def __init__(self, state_size: int, action_size: int, **kwargs):
#         self.state_size = state_size
#         self.action_size = action_size

#         self.gamma = kwargs.get("gamma", DEFAULTS["gamma"])
#         self.lr = kwargs.get("lr", DEFAULTS["lr"])
#         self.epsilon = kwargs.get("epsilon", DEFAULTS["epsilon"])
#         self.epsilon_min = kwargs.get("epsilon_min", DEFAULTS["epsilon_min"])
#         self.epsilon_decay = kwargs.get("epsilon_decay", DEFAULTS["epsilon_decay"])

#         self.batch_size = kwargs.get("batch_size", DEFAULTS["batch_size"])
#         self.memory = deque(maxlen=kwargs.get("memory_size", DEFAULTS["memory_size"]))
#         self.warmup = kwargs.get("warmup", DEFAULTS["warmup"])

#         self.model = self._build_model()
#         self.target_model = self._build_model()
#         self.update_target()

#     def _build_model(self) -> tf.keras.Model:
#         inputs = layers.Input(shape=(self.state_size,))
#         x = layers.Dense(32, activation="relu")(inputs)
#         x = layers.Dense(32, activation="relu")(x)
#         outputs = layers.Dense(self.action_size, activation="linear")(x)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(self.lr),
#             loss=tf.keras.losses.MeanSquaredError(),
#         )
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append(
#             (
#                 np.array(state, dtype=np.float32),
#                 int(action),
#                 float(reward),
#                 np.array(next_state, dtype=np.float32),
#                 bool(done),
#             )
#         )

#     def act(self, state) -> int:
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.action_size)
#         state = np.asarray(state, dtype=np.float32).reshape(1, -1)
#         q = self.model.predict(state, verbose=0)[0]
#         return int(np.argmax(q))

#     def replay(self, verbose: bool = False) -> Tuple[float, float]:
#         if len(self.memory) < max(self.warmup, self.batch_size):
#             return 0.0, 0.0

#         batch = random.sample(self.memory, self.batch_size)
#         states = np.stack([e[0] for e in batch], axis=0)
#         actions = np.array([e[1] for e in batch], dtype=np.int32)
#         rewards = np.array([e[2] for e in batch], dtype=np.float32)
#         next_states = np.stack([e[3] for e in batch], axis=0)
#         dones = np.array([e[4] for e in batch], dtype=np.bool_)

#         q_current = self.model.predict(states, verbose=0)
#         q_next_target = self.target_model.predict(next_states, verbose=0)

#         targets = q_current.copy()
#         max_next = np.max(q_next_target, axis=1)
#         td_target = rewards + (1.0 - dones.astype(np.float32)) * (self.gamma * max_next)

#         targets[np.arange(self.batch_size), actions] = td_target

#         history = self.model.train_on_batch(states, targets)
#         try:
#             loss = float(history)
#         except:
#             loss = float(history[0])

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#             if self.epsilon < self.epsilon_min:
#                 self.epsilon = self.epsilon_min

#         td_errors = np.abs(td_target - q_current[np.arange(self.batch_size), actions])
#         avg_td = float(np.mean(td_errors))

#         if verbose:
#             print(
#                 f"[replay] loss={loss:.4f} avg_td={avg_td:.4f} epsilon={self.epsilon:.3f}"
#             )

#         return loss, avg_td

#     def update_target(self):
#         self.target_model.set_weights(self.model.get_weights())

#     def save(self, path: str = "rl_treatment_optimized.h5"):
#         self.model.save(path, include_optimizer=False)

#     def load(self, path: str = "rl_treatment_optimized.h5"):
#         loaded = tf.keras.models.load_model(path, compile=False)
#         if (
#             loaded.output_shape[-1] != self.action_size
#             or loaded.input_shape[-1] != self.state_size
#         ):
#             raise ValueError(
#                 f"Loaded model shapes do not match agent: "
#                 f"model_in={loaded.input_shape} model_out={loaded.output_shape} "
#                 f"agent_in={(self.state_size,)} agent_out={(self.action_size,)}"
#             )
#         self.model = loaded
#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(self.lr),
#             loss=tf.keras.losses.MeanSquaredError(),
#         )
#         self.target_model = self._build_model()
#         self.update_target()

#     def memory_size(self) -> int:
#         return len(self.memory)

#     def summary(self):
#         print("DQNAgent summary:")
#         print(f" state_size: {self.state_size}, action_size: {self.action_size}")
#         print(f" gamma: {self.gamma}, lr: {self.lr}")
#         print(
#             f" epsilon: {self.epsilon:.4f} -> {self.epsilon_min}, decay: {self.epsilon_decay}"
#         )
#         print(
#             f" buffer: {len(self.memory)} entries, batch: {self.batch_size}, warmup: {self.warmup}"
#         )
#         self.model.summary()


import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
from typing import Tuple

DEFAULTS = {
    "gamma": 0.95,
    "lr": 0.003,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.96,
    "memory_size": 10000,
    "batch_size": 64,
    "warmup": 200,
    "target_update_freq": 20,
}


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, **kwargs):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = kwargs.get("gamma", DEFAULTS["gamma"])
        self.lr = kwargs.get("lr", DEFAULTS["lr"])
        self.epsilon = kwargs.get("epsilon", DEFAULTS["epsilon"])
        self.epsilon_min = kwargs.get("epsilon_min", DEFAULTS["epsilon_min"])
        self.epsilon_decay = kwargs.get("epsilon_decay", DEFAULTS["epsilon_decay"])

        self.batch_size = kwargs.get("batch_size", DEFAULTS["batch_size"])
        self.memory = deque(maxlen=kwargs.get("memory_size", DEFAULTS["memory_size"]))
        self.warmup = kwargs.get("warmup", DEFAULTS["warmup"])

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()

    def _build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(32, activation="relu")(inputs)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(self.action_size, activation="linear")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def act(self, state, greedy: bool = False) -> int:
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.action_size)
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q = self.model.predict(state, verbose=0)[0]
        return int(np.argmax(q))

    def replay(self, verbose: bool = False) -> Tuple[float, float]:
        if len(self.memory) < max(self.warmup, self.batch_size):
            return 0.0, 0.0

        batch = random.sample(self.memory, self.batch_size)
        states = np.stack([e[0] for e in batch], axis=0)
        actions = np.array([e[1] for e in batch], dtype=np.int32)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.stack([e[3] for e in batch], axis=0)
        dones = np.array([e[4] for e in batch], dtype=np.bool_)

        q_current = self.model.predict(states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        targets = q_current.copy()
        max_next = np.max(q_next_target, axis=1)
        td_target = rewards + (1.0 - dones.astype(np.float32)) * (self.gamma * max_next)

        targets[np.arange(self.batch_size), actions] = td_target

        history = self.model.train_on_batch(states, targets)
        try:
            loss = float(history)
        except:
            loss = float(history[0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        td_errors = np.abs(td_target - q_current[np.arange(self.batch_size), actions])
        avg_td = float(np.mean(td_errors))

        if verbose:
            print(
                f"[replay] loss={loss:.4f} avg_td={avg_td:.4f} epsilon={self.epsilon:.3f}"
            )

        return loss, avg_td

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path: str = "rl_treatment_optimized.h5"):
        self.model.save(path, include_optimizer=False)

    def load(self, path: str = "rl_treatment_optimized.h5"):
        loaded = tf.keras.models.load_model(path, compile=False)
        if (
            loaded.output_shape[-1] != self.action_size
            or loaded.input_shape[-1] != self.state_size
        ):
            raise ValueError(
                f"Loaded model shapes do not match agent: "
                f"model_in={loaded.input_shape} model_out={loaded.output_shape} "
                f"agent_in={(self.state_size,)} agent_out={(self.action_size,)}"
            )
        self.model = loaded
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        self.target_model = self._build_model()
        self.update_target()

    def memory_size(self) -> int:
        return len(self.memory)

    def summary(self):
        print("DQNAgent summary:")
        print(f" state_size: {self.state_size}, action_size: {self.action_size}")
        print(f" gamma: {self.gamma}, lr: {self.lr}")
        print(
            f" epsilon: {self.epsilon:.4f} -> {self.epsilon_min}, decay: {self.epsilon_decay}"
        )
        print(
            f" buffer: {len(self.memory)} entries, batch: {self.batch_size}, warmup: {self.warmup}"
        )
        self.model.summary()

    def evaluate_episode(self, env, render=False):
        state = env.reset()
        done = False
        total_reward = 0.0
        order = []
        treated = [False] * env.num_plants
        drone_pos = np.array([0.0, 0.0], dtype=np.float32)

        while not done:
            action = self.act(state, greedy=True)
            if treated[action]:
                found = False
                for i in range(env.num_plants):
                    if not treated[i]:
                        action = i
                        found = True
                        break
                if not found:
                    break

            next_state, reward, done, _ = env.step(action)
            order.append(int(action))
            total_reward += float(reward)
            state = next_state
            treated[action] = True

        return total_reward, order
