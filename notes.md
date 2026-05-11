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

# INM707 Deep Reinforcement Learning Project Notes

## Project idea

The project models a simplified solar microgrid with battery storage.

At each step, the agent sees:

- battery level
- demand level
- solar generation level
- time of day

The agent can choose one of three actions:

- charge
- idle
- discharge

The goal is to learn a sensible battery policy that meets demand while reducing grid import, avoiding wasted solar, and limiting unnecessary battery cycling.

---

## Basic environment

The state is written as:

```python
(battery_level, demand_regime, solar_regime, time_bin)
```

Where:

- `battery_level` is from 0 to 10
- `demand_regime`: 0 = low, 1 = medium, 2 = high
- `solar_regime`: 0 = low, 1 = medium, 2 = high
- `time_bin`: 0 = night, 1 = morning, 2 = midday, 3 = evening

The actions are:

- `0 = charge`
- `1 = idle`
- `2 = discharge`

This setup is suitable for tabular Q-learning because the state space and action space are both finite.

---

## Environment test 1: random action sanity check

Before adding Q-learning, I wanted to check that the environment itself actually behaves in a sensible way. For this first test I just used random actions, so the goal was not performance yet. I only wanted to see whether the battery, demand, solar, grid import, and reward values changed in a way that made sense.

For example:

```python
(5, 1, 0, 0)
```

means the battery is at level 5, demand is medium, solar is low, and the time is night.

### Output from the test

```text
Initial state: (5, 1, 0, 0)
Readable: {'battery_level': 5, 'demand': 'medium', 'solar': 'low', 'time': 'night'}

Step: 1
Action: charge
Next state: (5, 2, 0, 0)
Readable: {'battery_level': 5, 'demand': 'high', 'solar': 'low', 'time': 'night'}
Reward: -2.0
Info: {'hour': 1, 'action': np.int64(0), 'action_name': 'charge', 'demand': 2.0, 'solar': 0.0, 'battery_level': 5, 'battery_used': 0.0, 'grid_import': 2.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -2.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}

Step: 2
Action: charge
Next state: (5, 1, 0, 0)
Readable: {'battery_level': 5, 'demand': 'medium', 'solar': 'low', 'time': 'night'}
Reward: -3.0
Info: {'hour': 2, 'action': np.int64(0), 'action_name': 'charge', 'demand': 3.0, 'solar': 0.0, 'battery_level': 5, 'battery_used': 0.0, 'grid_import': 3.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -3.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}

Step: 3
Action: discharge
Next state: (4, 0, 0, 0)
Readable: {'battery_level': 4, 'demand': 'low', 'solar': 'low', 'time': 'night'}
Reward: -1.1
Info: {'hour': 3, 'action': np.int64(2), 'action_name': 'discharge', 'demand': 2.0, 'solar': 0.0, 'battery_level': 4, 'battery_used': 1.0, 'grid_import': 1.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -1.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.1}

Step: 4
Action: idle
Next state: (4, 1, 0, 0)
Readable: {'battery_level': 4, 'demand': 'medium', 'solar': 'low', 'time': 'night'}
Reward: -1.0
Info: {'hour': 4, 'action': np.int64(1), 'action_name': 'idle', 'demand': 1.0, 'solar': 0.0, 'battery_level': 4, 'battery_used': 0.0, 'grid_import': 1.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -1.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}

Step: 5
Action: idle
Next state: (4, 0, 0, 0)
Readable: {'battery_level': 4, 'demand': 'low', 'solar': 'low', 'time': 'night'}
Reward: -2.0
Info: {'hour': 5, 'action': np.int64(1), 'action_name': 'idle', 'demand': 2.0, 'solar': 0.0, 'battery_level': 4, 'battery_used': 0.0, 'grid_import': 2.0, 'wasted_solar': 0.0, 'unmet_demand': 0.0, 'unmet_penalty': -0.0, 'grid_penalty': -2.0, 'solar_waste_penalty': -0.0, 'battery_penalty': -0.0}
```

### What I noticed

The environment seems to be working as expected.

The starting state was:

```python
(5, 1, 0, 0)
```

This makes sense because the battery starts half full, demand is medium, solar is low, and the episode begins at night.

In the first two steps, the random action was `charge`. Since it was night and solar generation was zero, there was no surplus solar to charge the battery with. So the battery stayed at level 5. The grid had to cover the demand, which is why the rewards were `-2.0` and `-3.0`.

In step 3, the action was `discharge`. This reduced the battery from 5 to 4. The battery supplied 1 unit of energy, so grid import dropped to 1.0. The reward was `-1.1`, which comes from the grid penalty plus a small battery-use penalty. This is good because it shows that discharging can help reduce grid dependence, but it is not completely free.

In steps 4 and 5, the agent idled. Since solar was still zero, the grid covered the demand. The reward was just the grid import penalty.

One thing I noticed is that `unmet_demand` stays at zero in this version. That is because the grid is currently allowed to cover all demand not met by solar or the battery. This is fine for the first version, but later it may be useful to add a maximum grid import limit so unmet demand can actually happen. That would make the load-balancing problem more interesting.

### Small clean-up

The action values were printing as `np.int64(0)` instead of just `0`. This is because the random actions were sampled using NumPy. It does not affect the logic, but it makes the logs look messy.

I fixed this by adding:

```python
action = int(action)
```

at the start of the `step()` method.

### Conclusion

This first test confirms that the basic environment is behaving sensibly. The battery changes when it should, grid import reacts to demand and solar, and the reward values reflect the cost of using the grid, wasting solar, and using the battery.

## Random policy baseline

After the initial step-by-step sanity check, I ran a full 24-step episode using a random policy. The agent selected between `charge`, `idle`, and `discharge` uniformly at random.

The purpose of this test was to create a weak baseline before implementing Q-learning. I want the learned agent to achieve a better total reward and, more importantly, reduce grid import by using the battery more sensibly.

### Episode summary

```text
Random policy episode summary
-----------------------------
Total reward: -32.6
Total grid import: 31.0
Total wasted solar: 2.0
Total battery used: 6.0
Total unmet demand: 0.0
```

### Interpretation

The random policy performs poorly, which is expected. It does not understand when the battery should be saved or used.

One important pattern is that the random agent discharges the battery several times during the morning and midday, even when solar generation is already enough to cover demand. For example, around steps 11 and 12, the agent discharges even though grid import is already zero. This gives a small battery penalty without improving the energy balance.

By step 16, the battery has dropped to zero. This creates a problem later in the episode because the evening period has higher demand and lower solar generation. From steps 19 to 24, the battery remains empty, so the grid has to cover most of the demand. This leads to large grid import values and more negative rewards.

This baseline is useful because it shows what the learned policy should improve. A better policy should:
- avoid wasting battery when solar already covers demand;
- charge during solar surplus periods;
- preserve battery energy for evening/high-demand periods;
- reduce total grid import;
- achieve a less negative total reward.

At this stage, unmet demand is still zero because the grid can cover all remaining demand. Later, if I add a grid import limit, unmet demand will become a more meaningful reliability metric.


## Random policy baseline after adding grid limit

After adding a maximum grid import limit, I reran the random policy baseline for one 24-step episode.

This version is more interesting than the first test because the grid can no longer cover unlimited demand. The grid can only import up to 2 units per step. Therefore, if demand is high, solar is low, and the battery cannot help, unmet demand becomes possible.

### Episode summary

```text
Random policy episode summary
-----------------------------
Total reward: -77.6
Total grid import: 26.0
Total wasted solar: 2.0
Total battery used: 6.0
Total unmet demand: 5.0
```

### What changed

In the earlier version, unmet demand was always zero because the grid could cover all remaining demand. After limiting grid import, unmet demand appears in difficult states.

For example, in step 2:

```text
demand = 3.0
solar = 0.0
battery_used = 0.0
grid_import = 2.0
unmet_demand = 1.0
reward = -12.0
```

This makes sense. Demand was high, there was no solar, and the action did not use the battery. Since the grid could only supply 2 units, 1 unit of demand was unmet. Because unmet demand has a large penalty, the reward was much worse.

The same issue appears later in the evening when the battery is empty. In steps 21 to 24, demand is high, solar is zero, and the battery cannot supply energy. The grid limit means the system cannot fully meet demand, so unmet demand becomes positive and the rewards are strongly negative.

### Interpretation

This is a better load-balancing environment because the agent now has to manage a limited resource. Random behaviour wastes the battery earlier in the day and then has no stored energy left during the evening, when solar is low and demand is often high.

A learned policy should improve on this by:

- avoiding unnecessary discharge when solar already covers demand;
- charging when solar surplus is available;
- saving battery energy for evening or high-demand periods;
- reducing unmet demand;
- reducing grid import where possible;
- achieving a less negative total reward than the random policy.

This gives a clear baseline for Q-learning to beat.



## First Q-learning run

I trained the first tabular Q-learning agent for 1000 episodes using the discrete microgrid environment.

The agent used:
- alpha = 0.3
- gamma = 0.95
- epsilon start = 1.0
- epsilon minimum = 0.05
- epsilon decay = 0.995

The goal was to check whether Q-learning could improve over the random baseline.

### Training output

```text
Episode 100 | Avg reward: -48.36 | Avg unmet demand: 2.42 | Avg grid import: 21.89 | Epsilon: 0.606
Episode 200 | Avg reward: -44.49 | Avg unmet demand: 2.09 | Avg grid import: 21.27 | Epsilon: 0.367
Episode 300 | Avg reward: -36.23 | Avg unmet demand: 1.29 | Avg grid import: 21.30 | Epsilon: 0.222
Episode 400 | Avg reward: -40.02 | Avg unmet demand: 1.66 | Avg grid import: 21.37 | Epsilon: 0.135
Episode 500 | Avg reward: -33.20 | Avg unmet demand: 0.96 | Avg grid import: 21.71 | Epsilon: 0.082
Episode 600 | Avg reward: -31.11 | Avg unmet demand: 0.77 | Avg grid import: 21.63 | Epsilon: 0.050
Episode 700 | Avg reward: -29.11 | Avg unmet demand: 0.67 | Avg grid import: 20.59 | Epsilon: 0.050
Episode 800 | Avg reward: -31.13 | Avg unmet demand: 0.90 | Avg grid import: 20.45 | Epsilon: 0.050
Episode 900 | Avg reward: -34.20 | Avg unmet demand: 1.05 | Avg grid import: 21.66 | Epsilon: 0.050
Episode 1000 | Avg reward: -32.18 | Avg unmet demand: 0.87 | Avg grid import: 21.77 | Epsilon: 0.050
```

### Main training result

```text
Average reward first 100: -48.357
Average reward last 100: -32.179

Average unmet demand first 100: 2.42
Average unmet demand last 100: 0.87
```

This shows that the Q-learning agent is learning. The average reward becomes much less negative, and the average unmet demand decreases by more than half.

### Greedy policy evaluation

After training, I evaluated the learned greedy policy on one 24-step episode.

```text
Learned policy episode summary
------------------------------
Total reward: -33.2
Total grid import: 21.5
Total wasted solar: 2.0
Total battery used: 7.0
Total unmet demand: 1.0
```

For comparison, the earlier random policy baseline was:

```text
Random policy episode summary
-----------------------------
Total reward: -77.6
Total grid import: 26.0
Total wasted solar: 2.0
Total battery used: 6.0
Total unmet demand: 5.0
```

### Interpretation

The learned policy is clearly better than random on this test episode. It reduces unmet demand from 5.0 to 1.0 and improves total reward from -77.6 to -33.2. This suggests that the agent has learned to use the battery in a more useful way.

However, the learned policy is not perfect. In the greedy episode, it still discharges early at night, which may not always be ideal because evening demand is usually harder to meet. This could be due to the reward function encouraging immediate reduction in grid import, or because one evaluation episode is noisy.

The next step is to evaluate both the random policy and the learned policy over many episodes, not just one, so that the comparison is more reliable.

## Basic evaluation over 100 episodes

I evaluated the random policy and the trained Q-learning policy over 100 episodes each. This is more reliable than comparing only one episode, because demand and solar are sampled stochastically.

The random policy chooses charge, idle, and discharge with equal probability. The Q-learning policy uses the learned Q-table greedily after training.

### Output

```text
Training Q-learning agent...
Evaluating random policy...
Evaluating learned greedy policy...

Average performance over 100 episodes
-------------------------------------
Policy                   Reward       Grid      Unmet      Waste    Battery
---------------------------------------------------------------------------
Random                   -50.86      21.61       2.67       3.83       6.32
Q-learning greedy        -32.67      20.75       0.98       2.73       7.54
```

### Interpretation

The Q-learning policy clearly improves over the random policy.

The average reward improves from `-50.86` to `-32.67`, meaning the learned policy receives fewer penalties overall. More importantly, average unmet demand falls from `2.67` to `0.98`. This is the most important result because unmet demand represents failure to meet the community load when solar, battery, and limited grid import are not enough.

Grid import also falls slightly, from `21.61` to `20.75`. This improvement is smaller than the unmet demand improvement, but it still shows that the learned policy is relying slightly less on the grid. Wasted solar also falls from `3.83` to `2.73`, suggesting that the Q-learning policy makes better use of available solar energy.

Battery use increases from `6.32` to `7.54`. This is not automatically bad. In this case, the higher battery use appears to be useful because it is associated with lower unmet demand and better total reward. The key point is that the agent is learning to use the battery more productively than random behaviour.

Overall, this confirms that tabular Q-learning is learning a meaningful control policy in the discrete microgrid environment.

## Q-learning hyperparameter sweep

I ran a small hyperparameter sweep to test how Q-learning behaviour changes with different values of gamma, alpha, and epsilon decay.

The baseline setting was:

```text
alpha = 0.30
gamma = 0.95
epsilon_decay = 0.995
```

The sweep tested:
- gamma: 0.50, 0.80, 0.95, 0.99
- alpha: 0.10, 0.30, 0.70
- epsilon decay: 0.980, 0.995, 0.999

The goal was not to search every possible combination, but to understand how the main Q-learning parameters affect performance.

### Output

```text
Q-learning hyperparameter sweep
-------------------------------
Experiment           Alpha   Gamma  EpsDecay     Reward       Grid      Unmet      Waste    Battery
---------------------------------------------------------------------------------------------------------
baseline              0.30    0.95     0.995     -32.67      20.75       0.98       2.73       7.54
gamma_low             0.30    0.50     0.995     -29.90      20.39       0.76       2.14       8.45
gamma_mid             0.30    0.80     0.995     -30.75      20.48       0.84       2.15       8.02
gamma_high            0.30    0.99     0.995     -33.83      20.61       1.12       2.48       7.92
alpha_low             0.10    0.95     0.995     -27.96      21.80       0.44       2.12       7.02
alpha_high            0.70    0.95     0.995     -31.41      22.29       0.71       2.70       6.71
eps_fast              0.30    0.95     0.980     -32.59      20.31       1.04       2.12       8.25
eps_slow              0.30    0.95     0.999     -30.59      21.39       0.69       3.24       6.82
```

### Interpretation

The best result in this sweep was `alpha_low`, with:

```text
alpha = 0.10
gamma = 0.95
epsilon_decay = 0.995
```

This achieved the best reward, `-27.96`, and the lowest unmet demand, `0.44`.

The gamma sweep gave an interesting result. I originally expected higher gamma to perform best because battery scheduling seems like a future-planning problem. However, lower gamma values performed better in this simplified environment. `gamma = 0.50` achieved a reward of `-29.90`, while `gamma = 0.99` gave a weaker reward of `-33.83`.

This suggests that the environment does not require very long-term planning over many steps. Since the state already includes time-of-day, demand regime, solar regime, and battery level, the agent can often make useful decisions from the current state without needing to heavily weight distant future rewards.

The alpha sweep showed that a smaller learning rate worked best. `alpha = 0.10` outperformed both the baseline and the high-alpha setting. This makes sense because the environment is stochastic: demand and solar are sampled from probability distributions. A lower alpha means the Q-table updates more cautiously and does not overreact too strongly to individual random transitions.

The epsilon decay sweep showed that slower exploration helped compared with the baseline. `epsilon_decay = 0.999` achieved better reward and unmet demand than the baseline. This suggests that longer exploration helps the agent discover useful actions in rare but important states, such as high-demand, low-solar states with low battery.

Overall, the sweep shows that Q-learning performance is sensitive to the main hyperparameters. The best setting found was a cautious learning rate with continued exploration.

## Rule-based baseline

I added a simple rule-based controller so that Q-learning is compared against a basic human-designed strategy, not just random actions.

The rule-based policy was:

```text
if solar > demand and battery is not full:
    charge
elif demand > solar and battery has energy:
    discharge
else:
    idle
```

The idea is simple: store solar surplus when possible, use the battery when demand is higher than solar, and otherwise do nothing.

### Output

```text
Rule-based policy average performance over 100 episodes
-------------------------------------------------------
Policy                   Reward       Grid      Unmet      Waste    Battery
---------------------------------------------------------------------------
Rule-based               -38.88      19.05       1.83       1.18       9.37
```

### Comparison so far

```text
Policy              Reward   Grid   Unmet   Waste   Battery
Random              -50.86   21.61   2.67    3.83    6.32
Rule-based          -38.88   19.05   1.83    1.18    9.37
Q-learning greedy   -32.67   20.75   0.98    2.73    7.54
```

### Interpretation

The rule-based policy performs better than random, which makes sense. It follows a reasonable hand-written strategy: charge when there is solar surplus and discharge when demand is higher than solar.

The rule-based controller gives the lowest grid import and lowest wasted solar so far. This shows that the hand-written rules are good at using local solar energy efficiently.

However, the Q-learning policy still achieves the best average reward and the lowest unmet demand. This matters because unmet demand is the most serious failure in the reward function. The rule-based policy uses the battery more aggressively, with average battery use of `9.37`, but still leaves more unmet demand than Q-learning.

So the early result suggests that Q-learning is learning a better trade-off for the reward function used in this environment. It does not simply minimise grid import. Instead, it seems to reduce the most costly failure, which is unmet demand.

This is a stronger comparison than random vs Q-learning alone because the learned policy is now being compared against a simple human-designed controller.