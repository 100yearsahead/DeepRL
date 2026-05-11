from envs.discrete_microgrid_env import DiscreteMicrogridEnv
from agents.q_learning import QLearningAgent


def train_q_learning(num_episodes=1000, seed=42):
    env = DiscreteMicrogridEnv(seed=seed)

    agent = QLearningAgent(
        state_shape=env.state_space_shape(),
        num_actions=env.num_actions(),
        alpha=0.3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=seed,
    )

    episode_rewards = []
    episode_grid_imports = []
    episode_unmet_demands = []
    episode_wasted_solar = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        total_reward = 0.0
        total_grid_import = 0.0
        total_unmet_demand = 0.0
        total_wasted_solar = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state

            total_reward += reward
            total_grid_import += info["grid_import"]
            total_unmet_demand += info["unmet_demand"]
            total_wasted_solar += info["wasted_solar"]

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_grid_imports.append(total_grid_import)
        episode_unmet_demands.append(total_unmet_demand)
        episode_wasted_solar.append(total_wasted_solar)

        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            avg_unmet = sum(episode_unmet_demands[-100:]) / 100
            avg_grid = sum(episode_grid_imports[-100:]) / 100

            print(
                f"Episode {episode + 1} | "
                f"Avg reward: {avg_reward:.2f} | "
                f"Avg unmet demand: {avg_unmet:.2f} | "
                f"Avg grid import: {avg_grid:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    results = {
        "rewards": episode_rewards,
        "grid_imports": episode_grid_imports,
        "unmet_demands": episode_unmet_demands,
        "wasted_solar": episode_wasted_solar,
    }

    return agent, results


def evaluate_greedy_policy(agent, seed=123):
    """
    Run one episode using the learned greedy policy.
    """
    env = DiscreteMicrogridEnv(seed=seed)
    state = env.reset()
    done = False

    total_reward = 0.0
    total_grid_import = 0.0
    total_unmet_demand = 0.0
    total_wasted_solar = 0.0
    total_battery_used = 0.0

    print("\nEvaluating learned greedy policy")
    print("--------------------------------")
    print("Initial state:", state)
    print("Readable state:", env.describe_state(state))
    print()

    step = 0

    while not done:
        action = agent.best_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_grid_import += info["grid_import"]
        total_unmet_demand += info["unmet_demand"]
        total_wasted_solar += info["wasted_solar"]
        total_battery_used += abs(info["battery_used"])

        print(f"Step {step + 1}")
        print("Action:", info["action_name"])
        print("State:", env.describe_state(next_state))
        print("Reward:", round(reward, 3))
        print(
            "Energy:",
            {
                "demand": info["demand"],
                "solar": info["solar"],
                "battery": info["battery_level"],
                "grid_import": info["grid_import"],
                "wasted_solar": info["wasted_solar"],
                "battery_used": info["battery_used"],
                "unmet_demand": info["unmet_demand"],
            },
        )
        print()

        state = next_state
        step += 1

    print("Learned policy episode summary")
    print("------------------------------")
    print("Total reward:", round(total_reward, 3))
    print("Total grid import:", round(total_grid_import, 3))
    print("Total wasted solar:", round(total_wasted_solar, 3))
    print("Total battery used:", round(total_battery_used, 3))
    print("Total unmet demand:", round(total_unmet_demand, 3))


if __name__ == "__main__":
    agent, results = train_q_learning(num_episodes=1000, seed=42)

    print()
    print("Training complete.")
    print("Average reward first 100:", round(sum(results["rewards"][:100]) / 100, 3))
    print("Average reward last 100:", round(sum(results["rewards"][-100:]) / 100, 3))
    print("Average unmet demand first 100:", round(sum(results["unmet_demands"][:100]) / 100, 3))
    print("Average unmet demand last 100:", round(sum(results["unmet_demands"][-100:]) / 100, 3))

    evaluate_greedy_policy(agent, seed=123)