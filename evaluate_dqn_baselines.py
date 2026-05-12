"""
Evaluate DQN against simple baselines in the continuous microgrid environment.

Why we do this:
The first DQN results looked strong, but we need to check whether DQN is
actually learning a useful policy or whether the environment is just easy.

So we compare:
1. Random policy
2. Rule-based policy
3. Trained DQN policy
"""

from envs.continuous_microgrid_env import ContinuousMicrogridEnv
from train_dqn import train_dqn, evaluate_dqn


def choose_rule_based_action(env):
    """
    Simple hand-written battery controller for the continuous environment.

    Rule:
    - If solar is greater than demand and the battery has space, charge.
    - If demand is greater than solar and the battery has energy, discharge.
    - Otherwise, idle.
    """
    if env.solar > env.demand and env.battery_energy < env.battery_capacity:
        return 0  # charge

    if env.demand > env.solar and env.battery_energy > 0:
        return 2  # discharge

    return 1  # idle


def evaluate_random_policy(num_episodes=100, seed=100):
    """
    Evaluate a random policy.

    The random policy chooses charge, idle, or discharge randomly.
    This gives us a weak baseline.
    """
    total_reward = 0.0
    total_grid = 0.0
    total_unmet = 0.0
    total_waste = 0.0
    total_battery = 0.0

    for episode in range(num_episodes):
        env = ContinuousMicrogridEnv(seed=seed + episode)
        state = env.reset()
        done = False

        while not done:
            action = env.rng.choice([0, 1, 2])
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_grid += info["grid_import"]
            total_unmet += info["unmet_demand"]
            total_waste += info["wasted_solar"]
            total_battery += abs(info["battery_used"])

            state = next_state

    return {
        "reward": total_reward / num_episodes,
        "grid_import": total_grid / num_episodes,
        "unmet_demand": total_unmet / num_episodes,
        "wasted_solar": total_waste / num_episodes,
        "battery_used": total_battery / num_episodes,
    }


def evaluate_rule_based_policy(num_episodes=100, seed=100):
    """
    Evaluate the rule-based policy.

    This is a stronger baseline than random because it uses simple
    energy-management logic.
    """
    total_reward = 0.0
    total_grid = 0.0
    total_unmet = 0.0
    total_waste = 0.0
    total_battery = 0.0

    for episode in range(num_episodes):
        env = ContinuousMicrogridEnv(seed=seed + episode)
        state = env.reset()
        done = False

        while not done:
            action = choose_rule_based_action(env)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_grid += info["grid_import"]
            total_unmet += info["unmet_demand"]
            total_waste += info["wasted_solar"]
            total_battery += abs(info["battery_used"])

            state = next_state

    return {
        "reward": total_reward / num_episodes,
        "grid_import": total_grid / num_episodes,
        "unmet_demand": total_unmet / num_episodes,
        "wasted_solar": total_waste / num_episodes,
        "battery_used": total_battery / num_episodes,
    }


def print_results(random_results, rule_results, dqn_results):
    """
    Print results as a simple table.
    """
    print()
    print("Continuous environment policy comparison")
    print("----------------------------------------")
    print(f"{'Policy':<18} {'Reward':>10} {'Grid':>10} {'Unmet':>10} {'Waste':>10} {'Battery':>10}")
    print("-" * 75)

    print(
        f"{'Random':<18} "
        f"{random_results['reward']:>10.2f} "
        f"{random_results['grid_import']:>10.2f} "
        f"{random_results['unmet_demand']:>10.2f} "
        f"{random_results['wasted_solar']:>10.2f} "
        f"{random_results['battery_used']:>10.2f}"
    )

    print(
        f"{'Rule-based':<18} "
        f"{rule_results['reward']:>10.2f} "
        f"{rule_results['grid_import']:>10.2f} "
        f"{rule_results['unmet_demand']:>10.2f} "
        f"{rule_results['wasted_solar']:>10.2f} "
        f"{rule_results['battery_used']:>10.2f}"
    )

    print(
        f"{'DQN':<18} "
        f"{dqn_results['reward']:>10.2f} "
        f"{dqn_results['grid_import']:>10.2f} "
        f"{dqn_results['unmet_demand']:>10.2f} "
        f"{dqn_results['wasted_solar']:>10.2f} "
        f"{dqn_results['battery_used']:>10.2f}"
    )


if __name__ == "__main__":
    NUM_EPISODES = 100
    EVAL_SEED = 100

    print("Training DQN agent...")
    dqn_agent, history = train_dqn(num_episodes=1000, seed=42)

    print("Evaluating random policy...")
    random_results = evaluate_random_policy(num_episodes=NUM_EPISODES, seed=EVAL_SEED)

    print("Evaluating rule-based policy...")
    rule_results = evaluate_rule_based_policy(num_episodes=NUM_EPISODES, seed=EVAL_SEED)

    print("Evaluating DQN policy...")
    dqn_results = evaluate_dqn(dqn_agent, num_episodes=NUM_EPISODES, seed=EVAL_SEED)

    print_results(random_results, rule_results, dqn_results)