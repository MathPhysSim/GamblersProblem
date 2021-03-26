
# Gambler's Problem (Example 4.3 - Sutton and Barto - 2020, p.84)
Value iteration and policy iteration show some (numerical) differences during convergence (surprise):
## Value iteration
![Watch the video](https://user-images.githubusercontent.com/22523245/112661151-7e2ad500-8e56-11eb-854d-ea78095181ec.mp4)

## Policy iteration
![Watch the video](https://user-images.githubusercontent.com/22523245/112661111-723f1300-8e56-11eb-9ab5-1fa2acdd1d09.mp4)


PS.: Why to exclude the zero actions? They lead to an infinite horizon probblem (under a deterministic policy), and in combination with a discount factor \gamma = 1, there is no guarantee for an unique value function anymore. Either exclude them or set \gamma <1. 
:wink:
