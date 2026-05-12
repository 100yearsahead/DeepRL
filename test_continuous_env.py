from envs.continuous_microgrid_env import ContinuousMicrogridEnv


env = ContinuousMicrogridEnv(seed=42)
state = env.reset()

print("Initial state:", state)

for step in range(5):
    action = env.rng.choice([0, 1, 2])
    next_state, reward, done, info = env.step(action)

    print()
    print("Step:", step + 1)
    print("Action:", info["action_name"])
    print("Reward:", round(reward, 3))
    print("Next state:", next_state)
    print(
        "Info:",
        {
            "demand": round(info["demand"], 3),
            "solar": round(info["solar"], 3),
            "battery_soc": round(info["battery_soc"], 3),
            "grid_import": round(info["grid_import"], 3),
            "unmet_demand": round(info["unmet_demand"], 3),
            "wasted_solar": round(info["wasted_solar"], 3),
        },
    )