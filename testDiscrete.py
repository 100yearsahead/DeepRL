from envs.discrete_microgrid_env import DiscreteMicrogridEnv

env = DiscreteMicrogridEnv(seed=42)

state = env.reset()
print("Initial state:", state)
print("Readable:", env.describe_state(state))

for step in range(5):
    action = env.rng.choice([0, 1, 2])
    next_state, reward, done, info = env.step(action)

    print("\nStep:", step + 1)
    print("Action:", info["action_name"])
    print("Next state:", next_state)
    print("Readable:", env.describe_state(next_state))
    print("Reward:", reward)
    print("Info:", info)