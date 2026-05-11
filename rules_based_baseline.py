from envs.discrete_microgrid_env import DiscreteMicrogridEnv


def choose_rule_based_action(env):
    """
    Simple hand-written battery policy.

    Rules:
    1. If there is solar surplus and the battery is not full, charge.
    2. If demand is greater than solar and the battery has energy, discharge.
    3. Otherwise, idle.
    """
    demand = env.demand_values[env.demand_regime]
    solar = env.solar_values[env.solar_regime]

    if solar > demand and env.battery_level < env.battery_capacity:
        return 0  # charge

    if demand > solar and env.battery_level > 0:
        return 2  # discharge

    return 1  # idle


def run_rule_based_policy(num_episodes=100, seed=300):
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
            action = choose_rule_based_action(env)
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


def print_rule_based_results(results):
    print()
    print("Rule-based policy average performance over 100 episodes")
    print("-------------------------------------------------------")
    print(f"{'Policy':<20} {'Reward':>10} {'Grid':>10} {'Unmet':>10} {'Waste':>10} {'Battery':>10}")
    print("-" * 75)

    print(
        f"{'Rule-based':<20} "
        f"{results['reward']:>10.2f} "
        f"{results['grid_import']:>10.2f} "
        f"{results['unmet_demand']:>10.2f} "
        f"{results['wasted_solar']:>10.2f} "
        f"{results['battery_used']:>10.2f}"
    )


if __name__ == "__main__":
    results = run_rule_based_policy(num_episodes=100, seed=300)
    print_rule_based_results(results)