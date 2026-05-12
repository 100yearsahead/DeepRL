"""
Train a simple DQN agent on the continuous microgrid environment.

Why DQN?
In the basic task, the state was discrete, so we could use a Q-table.
Here, the state is continuous, so a Q-table is not practical.

Instead, DQN uses a neural network to estimate:

Q(state, action)

The network outputs one Q-value for each action:
0 = charge
1 = idle
2 = discharge
"""

from envs.continuous_microgrid_env import ContinuousMicrogridEnv
from agents.dqn import DQNAgent


def train_dqn(num_episodes=1000, seed=42):
    """
    Train the DQN agent.

    Main idea:
    - Reset the environment at the start of each episode.
    - Let the agent choose actions using epsilon-greedy exploration.
    - Store each transition in replay memory.
    - Train the neural network using random batches from memory.
    - Occasionally update the target network.
    """

    # Create the continuous microgrid environment.
    env = ContinuousMicrogridEnv(seed=seed)

    # Create the DQN agent.
    # The agent needs to know:
    # - how many numbers are in the state vector
    # - how many actions it can choose from
    agent = DQNAgent(
        state_size=env.state_size(),
        num_actions=env.num_actions(),
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        replay_size=10000,
        seed=seed,
    )

    # These lists store training history so we can plot/inspect learning later.
    rewards = []
    unmet_demands = []
    losses = []

    # The target network is updated every fixed number of episodes.
    # This helps stabilise learning.
    target_update_every = 25

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        total_reward = 0.0
        total_unmet = 0.0
        episode_losses = []

        while not done:
            # Choose an action using epsilon-greedy.
            # Early in training this is more random.
            # Later it becomes more greedy.
            action = agent.choose_action(state)

            # Take the action in the environment.
            next_state, reward, done, info = env.step(action)

            # Store the transition in replay memory.
            # This is one of the main DQN improvements.
            agent.remember(state, action, reward, next_state, done)

            # Train the neural network from a random batch of past transitions.
            # If the replay buffer is still too small, this returns None.
            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            # Track important episode totals.
            total_reward += reward
            total_unmet += info["unmet_demand"]

            # Move to the next state.
            state = next_state

        # Reduce exploration after each episode.
        agent.decay_epsilon()

        # Copy online network weights to the target network sometimes.
        # This avoids the target changing too quickly.
        if (episode + 1) % target_update_every == 0:
            agent.update_target_network()

        rewards.append(total_reward)
        unmet_demands.append(total_unmet)

        if len(episode_losses) > 0:
            average_loss = sum(episode_losses) / len(episode_losses)
        else:
            average_loss = 0.0

        losses.append(average_loss)

        # Print progress every 100 episodes.
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards[-100:]) / 100
            avg_unmet = sum(unmet_demands[-100:]) / 100
            avg_loss = sum(losses[-100:]) / 100

            print(
                f"Episode {episode + 1} | "
                f"Avg reward: {avg_reward:.2f} | "
                f"Avg unmet: {avg_unmet:.2f} | "
                f"Avg loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    history = {
        "rewards": rewards,
        "unmet_demands": unmet_demands,
        "losses": losses,
    }

    return agent, history


def evaluate_dqn(agent, num_episodes=100, seed=100):
    """
    Evaluate the trained DQN agent.

    During evaluation, we do not use random exploration.
    The agent simply chooses the action with the highest Q-value.
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
            # Greedy action means no exploration.
            action = agent.greedy_action(state)

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_grid += info["grid_import"]
            total_unmet += info["unmet_demand"]
            total_waste += info["wasted_solar"]
            total_battery += abs(info["battery_used"])

            state = next_state

    results = {
        "reward": total_reward / num_episodes,
        "grid_import": total_grid / num_episodes,
        "unmet_demand": total_unmet / num_episodes,
        "wasted_solar": total_waste / num_episodes,
        "battery_used": total_battery / num_episodes,
    }

    return results


if __name__ == "__main__":
    print("Training DQN...")
    agent, history = train_dqn(num_episodes=1000, seed=42)

    print()
    print("Evaluating learned DQN policy...")
    results = evaluate_dqn(agent, num_episodes=100, seed=100)

    print()
    print("DQN evaluation over 100 episodes")
    print("--------------------------------")
    print("Reward:", round(results["reward"], 2))
    print("Grid import:", round(results["grid_import"], 2))
    print("Unmet demand:", round(results["unmet_demand"], 2))
    print("Wasted solar:", round(results["wasted_solar"], 2))
    print("Battery used:", round(results["battery_used"], 2))