from evaluate_basic import train_q_learning, run_greedy_policy


def run_experiment(name, alpha=0.3, gamma=0.95, epsilon_decay=0.995):
    """
    Train one Q-learning agent with a specific set of hyperparameters,
    then evaluate the learned greedy policy over 100 episodes.
    """
    agent = train_q_learning(
        num_episodes=1000,
        alpha=alpha,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        seed=42,
    )

    results = run_greedy_policy(agent, num_episodes=1000, seed=100)

    results["name"] = name
    results["alpha"] = alpha
    results["gamma"] = gamma
    results["epsilon_decay"] = epsilon_decay

    return results


def print_table(results):
    print()
    print("Q-learning hyperparameter sweep")
    print("-------------------------------")
    print(
        f"{'Experiment':<18} "
        f"{'Alpha':>7} "
        f"{'Gamma':>7} "
        f"{'EpsDecay':>9} "
        f"{'Reward':>10} "
        f"{'Grid':>10} "
        f"{'Unmet':>10} "
        f"{'Waste':>10} "
        f"{'Battery':>10}"
    )
    print("-" * 105)

    for r in results:
        print(
            f"{r['name']:<18} "
            f"{r['alpha']:>7.2f} "
            f"{r['gamma']:>7.2f} "
            f"{r['epsilon_decay']:>9.3f} "
            f"{r['reward']:>10.2f} "
            f"{r['grid_import']:>10.2f} "
            f"{r['unmet_demand']:>10.2f} "
            f"{r['wasted_solar']:>10.2f} "
            f"{r['battery_used']:>10.2f}"
        )


if __name__ == "__main__":
    experiments = []

    # Baseline from the first Q-learning experiment.
    experiments.append(
        run_experiment(
            name="baseline",
            alpha=0.3,
            gamma=0.95,
            epsilon_decay=0.995,
        )
    )

    # Gamma sweep: future reward importance.
    experiments.append(run_experiment("gamma_low", alpha=0.3, gamma=0.50, epsilon_decay=0.995))
    experiments.append(run_experiment("gamma_mid", alpha=0.3, gamma=0.80, epsilon_decay=0.995))
    experiments.append(run_experiment("gamma_high", alpha=0.3, gamma=0.99, epsilon_decay=0.995))

    # Alpha sweep: learning rate.
    experiments.append(run_experiment("alpha_low", alpha=0.1, gamma=0.95, epsilon_decay=0.995))
    experiments.append(run_experiment("alpha_high", alpha=0.7, gamma=0.95, epsilon_decay=0.995))

    # Epsilon decay sweep: exploration schedule.
    experiments.append(run_experiment("eps_fast", alpha=0.3, gamma=0.95, epsilon_decay=0.980))
    experiments.append(run_experiment("eps_slow", alpha=0.3, gamma=0.95, epsilon_decay=0.999))

    print_table(experiments)