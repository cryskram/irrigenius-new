import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")

plt.plot(df["episode"], df["total_reward"])
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Reward Curve")
plt.grid(True)
plt.tight_layout()

plt.show()
