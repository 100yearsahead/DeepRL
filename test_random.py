from envs.discrete_microgrid_env import DiscreteMicrogridEnv


def run_random_episode(seed=42):
    env = DiscreteMicrogridEnv(seed=seed)
    state = env.reset()

    total_reward = 0.0
    total_grid_import = 0.0
    total_wasted_solar = 0.0
    total_battery_used = 0.0
    total_unmet_demand = 0.0

    print("Initial state:", state)
    print("Readable state:", env.describe_state(state))
    print()

    done = False
    step = 0

    while not done:
        action = env.rng.choice([0, 1, 2])
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_grid_import += info["grid_import"]
        total_wasted_solar += info["wasted_solar"]
        total_battery_used += abs(info["battery_used"])
        total_unmet_demand += info["unmet_demand"]

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

        step += 1

    print("Random policy episode summary")
    print("-----------------------------")
    print("Total reward:", round(total_reward, 3))
    print("Total grid import:", round(total_grid_import, 3))
    print("Total wasted solar:", round(total_wasted_solar, 3))
    print("Total battery used:", round(total_battery_used, 3))
    print("Total unmet demand:", round(total_unmet_demand, 3))


if __name__ == "__main__":
    run_random_episode(seed=42)