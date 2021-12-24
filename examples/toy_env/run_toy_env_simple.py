"""Toy environment launcher. See the docs for more details about this environment.

"""

import numpy as np

from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
from Toy_env import MyEnv as Toy_env
import deer.experiment.base_controllers as bc



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

rng = np.random.RandomState(123456)

# --- Instantiate environment ---
env = Toy_env(rng)

# epsilon = 0.5

# --- Instantiate qnetwork ---
qnetwork = MyQNetwork(
    environment=env,
    random_state=rng,
    # rms_epsilon=epsilon
)
# qnetwork.setLearningRate(0.05)
qnetwork.setDiscountFactor(0.9)

# --- Instantiate agent ---
agent = NeuralAgent(
    env,
    qnetwork,
    random_state=rng)


# print(agent.qnetwork.qValues([0]))

# --- Bind controllers to the agent ---
# Before every training epoch, we want to print a summary of the agent's epsilon, discount and 
# learning rate as well as the training epoch number.
agent.attach(bc.VerboseController())

# During training epochs, we want to train the agent after every action it takes.
# Plus, we also want to display after each training episode (!= than after every training) the average bellman
# residual and the average of the V values obtained during the last episode.
agent.attach(bc.TrainerController())

# We also want to interleave a "test epoch" between each training epoch. 
agent.attach(bc.InterleavedTestEpochController(epoch_length=1000))

start_time = time.time()

n_epochs = 10
epoch_length = 1000
# --- Run the experiment ---
agent.run(n_epochs=n_epochs, epoch_length=epoch_length)

total_time = time.time() - start_time
hrs = int(total_time / (60 * 60))
mins = int(total_time % (60*60) / 60)
sec = int(total_time % 60 % (60*60))

print(f'\nFinished training after {hrs}h:{mins}m:{sec}s\n')
print('Q-values:\n')

n_states = 10
states_transformed = [2*i/(n_states-1) - 1 for i in range(n_states)]

q_values = [ qnetwork.qValues([states_transformed[i]]) for i in range(n_states) ]

action_a = [q_values[state][0] for state in range(10)]
action_b = [q_values[state][1] for state in range(10)]

print(action_a)
print(action_b)

np.savetxt('action_a.txt', np.array(action_a))
np.savetxt('action_b.txt', np.array(action_b))

# for state in range(10):
#     print(f'state {state+1} = {qnetwork.qValues([state])}')
