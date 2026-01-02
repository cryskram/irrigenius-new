from env import CropTreatmentEnv
from agent import DQNAgent
import tensorflow as tf
import time
import csv
import os

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

EPISODES = 800
NUM_PLANTS = 8
TARGET_UPDATE_FREQ = 20
EVAL_FREQ = 50
SAVE_BEST_PATH = "best_rl_model.h5"
LOG_CSV = "training_log.csv"

env = CropTreatmentEnv(num_plants=NUM_PLANTS)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "total_reward",
                "loss",
                "avg_td",
                "epsilon",
                "eval_reward",
                "eval_order",
                "time_s",
            ]
        )

best_eval = -1e9
start_time = time.time()

for ep in range(1, EPISODES + 1):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    loss, avg_td = agent.replay(verbose=(ep % 100 == 0))

    if ep % TARGET_UPDATE_FREQ == 0:
        agent.update_target()

    eval_reward = ""
    eval_order = ""
    if ep % EVAL_FREQ == 0:
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
        eval_reward, eval_order = agent.evaluate_episode(env)
        agent.epsilon = old_epsilon

        if eval_reward > best_eval:
            print(
                f"New best eval {eval_reward:.2f} at ep {ep}, saving model to {SAVE_BEST_PATH}"
            )
            agent.save(SAVE_BEST_PATH)
            best_eval = eval_reward

    elapsed = time.time() - start_time

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                ep,
                float(total_reward),
                float(loss),
                float(avg_td),
                float(agent.epsilon),
                eval_reward if eval_reward != "" else "",
                ",".join(map(str, eval_order)) if eval_order != "" else "",
                round(elapsed, 2),
            ]
        )

    if ep % 20 == 0:
        print(
            f"Ep {ep:04d} | Reward {total_reward:.2f} | Loss {loss:.4f} | Eps {agent.epsilon:.3f}"
        )

agent.save("rl_treatment_optimized.h5")
print("Training completed.")
