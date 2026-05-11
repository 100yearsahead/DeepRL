# INM707 Deep RL Project Notes

## Project idea

The project models a simplified solar microgrid with battery storage. The agent controls the battery by choosing whether to charge, idle, or discharge at each timestep.

The goal is to meet demand while reducing grid import, avoiding wasted solar generation, and limiting unnecessary battery cycling.

## Basic environment

State:
- battery_level in {0, ..., 10}
- demand_regime in {low, medium, high}
- solar_regime in {low, medium, high}
- time_bin in {night, morning, midday, evening}

Actions:
- charge
- idle
- discharge

Why this is suitable:
- finite state space
- discrete action space
- interpretable
- suitable for tabular Q-learning

## Markov assumption

The next state distribution depends on the current battery level, demand regime, solar regime, time-of-day, and action. This is an approximation of a real microgrid, but it is sufficient for a simplified MDP because the relevant variables for the next control decision are included in the state.
