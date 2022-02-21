import gym
import graphRLnx

env = gym.make('graphRL-v0')

start_state = 0
aim_state = [4]
fire = [3]
num_epoch=200

final = False

for t in range(1, num_epoch + 1):
    observation = env.reset(start_state = start_state, aim_state = aim_state, fire=fire)
    path = [start_state]
    final = False
    while not final:
        current_state, reward, done, info = env.step(action=None)
        path.append(current_state)
        if done:
            final = True

print(path)
