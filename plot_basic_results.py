import os
import numpy as np
import matplotlib.pyplot as plt

from envs.discrete_microgrid_env import DiscreteMicrogridEnv
from agents.q_learning import QLearningAgent
from evaluate_basic import run_random_policy, run_greedy_policy
from rules_based_baseline import run_rule_based_policy


FIGURE_DIR = "results/figures"

# Final plotting setup
TRAIN_EPISODES = 3000
EVAL_EPISODES = 1000
TRAIN_SEED = 42
EVAL_SEED = 100

# Best alpha found from the hyperparameter sweep
ALPHA = 0.1
GAMMA = 0.95
EPSILON_DECAY = 0.995


def moving_average(values, window=100):
    """
    Simple moving average so the learning curves are easier to read.
    """
    values = np.array(values)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def train_q_learning_with_history(num_episodes=TRAIN_EPISODES, seed=TRAIN_SEED):
    """
    Train Q-learning and keep track of reward and unmet demand.

    This is mainly for plotting the final learning curves.
    """
    env = DiscreteMicrogridEnv(seed=seed)

    agent = QLearningAgent(
        state_shape=env.state_space_shape(),
        num_actions=env.num_actions(),
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=EPSILON_DECAY,
        seed=seed,
    )

    rewards = []
    unmet_demands = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        total_reward = 0.0
        total_unmet = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            total_unmet += info["unmet_demand"]

            state = next_state

        agent.decay_epsilon()

        rewards.append(total_reward)
        unmet_demands.append(total_unmet)

    return agent, rewards, unmet_demands


def plot_learning_curves(rewards, unmet_demands):
    """
    Plot reward and unmet demand across training.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

    episodes = np.arange(1, len(rewards) + 1)
    window = 100

    # Reward curve
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.2, label="Episode reward")

    reward_ma = moving_average(rewards, window=window)
    ma_episodes = np.arange(window, len(rewards) + 1)
    plt.plot(
        ma_episodes,
        reward_ma,
        linewidth=2,
        label=f"{window}-episode moving average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Total episode reward")
    plt.title("Q-learning reward over training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/q_learning_reward_curve.png", dpi=300)
    plt.close()

    # Unmet demand curve
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, unmet_demands, alpha=0.2, label="Episode unmet demand")

    unmet_ma = moving_average(unmet_demands, window=window)
    plt.plot(
        ma_episodes,
        unmet_ma,
        linewidth=2,
        label=f"{window}-episode moving average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Unmet demand per episode")
    plt.title("Unmet demand over Q-learning training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/q_learning_unmet_demand_curve.png", dpi=300)
    plt.close()


def plot_baseline_comparison(agent):
    """
    Compare random, rule-based, and Q-learning policies.

    All policies use the same evaluation seed so they face the same
    sampled demand/solar episodes.
    """
    random_results = run_random_policy(num_episodes=EVAL_EPISODES, seed=EVAL_SEED)
    rule_results = run_rule_based_policy(num_episodes=EVAL_EPISODES, seed=EVAL_SEED)
    q_results = run_greedy_policy(agent, num_episodes=EVAL_EPISODES, seed=EVAL_SEED)

    policies = ["Random", "Rule-based", "Q-learning"]

    rewards = [
        random_results["reward"],
        rule_results["reward"],
        q_results["reward"],
    ]

    grid = [
        random_results["grid_import"],
        rule_results["grid_import"],
        q_results["grid_import"],
    ]

    unmet = [
        random_results["unmet_demand"],
        rule_results["unmet_demand"],
        q_results["unmet_demand"],
    ]

    waste = [
        random_results["wasted_solar"],
        rule_results["wasted_solar"],
        q_results["wasted_solar"],
    ]

    metrics = {
        "Reward": rewards,
        "Grid import": grid,
        "Unmet demand": unmet,
        "Wasted solar": waste,
    }

    x = np.arange(len(policies))
    width = 0.2

    plt.figure(figsize=(10, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        plt.bar(x + offset, values, width, label=metric_name)

    plt.xticks(x, policies)
    plt.ylabel(f"Average value over {EVAL_EPISODES} episodes")
    plt.title("Policy comparison in the discrete microgrid environment")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/baseline_policy_comparison.png", dpi=300)
    plt.close()

    print("Baseline comparison values:")
    print("Random:", random_results)
    print("Rule-based:", rule_results)
    print("Q-learning:", q_results)


def plot_hyperparameter_results():
    """
    Plot the hyperparameter sweep results.

    These values are from the updated sweep:
    1000 training episodes per setting and 1000 evaluation episodes.
    """
    experiments = [
        "baseline",
        "gamma_low",
        "gamma_mid",
        "gamma_high",
        "alpha_low",
        "alpha_high",
        "eps_fast",
        "eps_slow",
    ]

    rewards = [
        -33.63,
        -31.37,
        -32.39,
        -35.50,
        -28.49,
        -31.77,
        -32.72,
        -32.80,
    ]

    unmet = [
        1.02,
        0.87,
        0.96,
        1.24,
        0.46,
        0.70,
        1.01,
        0.86,
    ]

    x = np.arange(len(experiments))

    plt.figure(figsize=(11, 5))
    plt.bar(x, rewards)
    plt.xticks(x, experiments, rotation=35, ha="right")
    plt.ylabel("Average reward")
    plt.title("Q-learning hyperparameter sweep: reward")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/hyperparameter_reward_bar.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(x, unmet)
    plt.xticks(x, experiments, rotation=35, ha="right")
    plt.ylabel("Average unmet demand")
    plt.title("Q-learning hyperparameter sweep: unmet demand")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/hyperparameter_unmet_bar.png", dpi=300)
    plt.close()


def collect_greedy_episode(agent, seed=123):
    """
    Run one greedy episode and save values for plotting.

    This helps show what the learned policy actually does over a day.
    """
    env = DiscreteMicrogridEnv(seed=seed)
    state = env.reset()
    done = False

    hours = []
    battery_levels = []
    demand_values = []
    solar_values = []
    actions = []

    while not done:
        action = agent.best_action(state)
        next_state, reward, done, info = env.step(action)

        hours.append(info["hour"])
        battery_levels.append(info["battery_level"])
        demand_values.append(info["demand"])
        solar_values.append(info["solar"])
        actions.append(info["action_name"])

        state = next_state

    return hours, battery_levels, demand_values, solar_values, actions


def plot_battery_trajectory(agent):
    """
    Plot battery, demand, and solar over one greedy episode.
    """
    hours, battery, demand, solar, actions = collect_greedy_episode(agent, seed=123)

    plt.figure(figsize=(10, 6))
    plt.plot(hours, battery, marker="o", label="Battery level")
    plt.plot(hours, demand, marker="o", label="Demand")
    plt.plot(hours, solar, marker="o", label="Solar generation")

    plt.xlabel("Hour")
    plt.ylabel("Value")
    plt.title("Learned Q-policy behaviour over one episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/q_policy_battery_trajectory.png", dpi=300)
    plt.close()

    print("Actions in greedy episode:")
    for hour, action in zip(hours, actions):
        print(f"Hour {hour}: {action}")


def plot_policy_heatmap(agent, time_bin=3, solar_regime=0):
    """
    Plot the learned action for different battery and demand states.

    Default:
    time_bin = 3 means evening
    solar_regime = 0 means low solar

    This is optional for the report, but useful for checking the Q-table.
    """
    battery_levels = range(0, 11)
    demand_regimes = range(0, 3)

    heatmap = np.zeros((len(demand_regimes), len(battery_levels)))

    for d in demand_regimes:
        for b in battery_levels:
            state = (b, d, solar_regime, time_bin)
            action = agent.best_action(state)
            heatmap[d, b] = action

    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap, aspect="auto")

    plt.colorbar(ticks=[0, 1, 2], label="Action: 0=charge, 1=idle, 2=discharge")
    plt.xticks(list(battery_levels))
    plt.yticks([0, 1, 2], ["low demand", "medium demand", "high demand"])

    plt.xlabel("Battery level")
    plt.ylabel("Demand regime")
    plt.title("Learned Q-policy heatmap: evening, low solar")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/q_policy_heatmap_evening_low_solar.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Training Q-learning agent and collecting history...")
    agent, rewards, unmet_demands = train_q_learning_with_history(
        num_episodes=TRAIN_EPISODES,
        seed=TRAIN_SEED,
    )

    print("Plotting learning curves...")
    plot_learning_curves(rewards, unmet_demands)

    print("Plotting baseline comparison...")
    plot_baseline_comparison(agent)

    print("Plotting hyperparameter results...")
    plot_hyperparameter_results()

    print("Plotting battery trajectory...")
    plot_battery_trajectory(agent)

    print("Plotting policy heatmap...")
    plot_policy_heatmap(agent)

    print()
    print(f"Done. Figures saved in: {FIGURE_DIR}")