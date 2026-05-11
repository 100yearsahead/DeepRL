from envs.discrete_microgrid_env import DiscreteMicrogridEnv
from agents.q_learning import QLearningAgent


def run_random_policy(num_episodes=100, seed=100):
    """
    Runs a random policy for several episodes.

    The random policy just chooses charge, idle, or discharge randomly.
    This gives us a weak baseline that Q-learning should beat.
    """
    total_reward = 0.0
    total_grid_import = 0.0
    total_unmet_demand = 0.0
    total_wasted_solar = 0.0
    total_battery_used = 0.0

    for episode in range(num_episodes):
        env = DiscreteMicrogridEnv(seed=seed + episode)
        state = env.reset()
        done = False

        while not done:
            action = env.rng.choice([0, 1, 2])
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_grid_import += info["grid_import"]
            total_unmet_demand += info["unmet_demand"]
            total_wasted_solar += info["wasted_solar"]
            total_battery_used += abs(info["battery_used"])

            state = next_state

    return {
        "reward": total_reward / num_episodes,
        "grid_import": total_grid_import / num_episodes,
        "unmet_demand": total_unmet_demand / num_episodes,
        "wasted_solar": total_wasted_solar / num_episodes,
        "battery_used": total_battery_used / num_episodes,
    }


def train_q_learning(
    num_episodes=1000,
    alpha=0.3,
    gamma=0.95,
    epsilon_decay=0.995,
    seed=42,
):
    """
    Trains the Q-learning agent.

    This uses the basic settings from the first experiment.
    We can tune alpha, gamma, and epsilon decay later.
    """
    env = DiscreteMicrogridEnv(seed=seed)

    agent = QLearningAgent(
        state_shape=env.state_space_shape(),
        num_actions=env.num_actions(),
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state

        agent.decay_epsilon()

    return agent


def run_greedy_policy(agent, num_episodes=100, seed=200):
    """
    Runs the learned policy after training.

    The agent no longer explores here. It always chooses the action
    with the highest Q-value for the current state.
    """
    total_reward = 0.0
    total_grid_import = 0.0
    total_unmet_demand = 0.0
    total_wasted_solar = 0.0
    total_battery_used = 0.0

    for episode in range(num_episodes):
        env = DiscreteMicrogridEnv(seed=seed + episode)
        state = env.reset()
        done = False

        while not done:
            action = agent.best_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_grid_import += info["grid_import"]
            total_unmet_demand += info["unmet_demand"]
            total_wasted_solar += info["wasted_solar"]
            total_battery_used += abs(info["battery_used"])

            state = next_state

    return {
        "reward": total_reward / num_episodes,
        "grid_import": total_grid_import / num_episodes,
        "unmet_demand": total_unmet_demand / num_episodes,
        "wasted_solar": total_wasted_solar / num_episodes,
        "battery_used": total_battery_used / num_episodes,
    }


def print_results(random_results, q_results):
    """
    Prints the results in a simple table.
    """
    print()
    print("Average performance over 100 episodes")
    print("-------------------------------------")
    print(f"{'Policy':<20} {'Reward':>10} {'Grid':>10} {'Unmet':>10} {'Waste':>10} {'Battery':>10}")
    print("-" * 75)

    print(
        f"{'Random':<20} "
        f"{random_results['reward']:>10.2f} "
        f"{random_results['grid_import']:>10.2f} "
        f"{random_results['unmet_demand']:>10.2f} "
        f"{random_results['wasted_solar']:>10.2f} "
        f"{random_results['battery_used']:>10.2f}"
    )

    print(
        f"{'Q-learning greedy':<20} "
        f"{q_results['reward']:>10.2f} "
        f"{q_results['grid_import']:>10.2f} "
        f"{q_results['unmet_demand']:>10.2f} "
        f"{q_results['wasted_solar']:>10.2f} "
        f"{q_results['battery_used']:>10.2f}"
    )


if __name__ == "__main__":
    print("Training Q-learning agent...")
    agent = train_q_learning(num_episodes=1000, seed=42)

    print("Evaluating random policy...")
    random_results = run_random_policy(num_episodes=100, seed=100)

    print("Evaluating learned greedy policy...")
    q_results = run_greedy_policy(agent, num_episodes=100, seed=200)

    print_results(random_results, q_results)