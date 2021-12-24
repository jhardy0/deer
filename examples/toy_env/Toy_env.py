""" 
The environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction.
Two actions are possible for the agent:
- Action 0 corresponds to selling if the agent possesses one unit or idle if the agent possesses zero unit.
- Action 1 corresponds to buying if the agent possesses zero unit or idle if the agent already possesses one unit.
The state of the agent is made up of an history of two punctual observations:
- The price signal
- Either the agent possesses the good or not (1 or 0)
The price signal is build following the same rules for the training and the validation environment. That allows the agent to learn a strategy that exploits this successfully.

"""

import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

from deer.base_classes import Environment

class MyEnv(Environment):
    def __init__(self, rng):
        """ Initialize environment.

        Parameters
        -----------
        rng : the numpy random number generator
        """

        self._state_to_transform, self._transform_to_state = self.CreateStateDictionary(10)

        self._transitions, self._rewards = self.CreateTransitionAndRewardMatrix(10)
        self._last_ponctual_observation = [self._state_to_transform[0]] # Always start in first state
        self._counter = 1

    def reset(self, mode):
        """ Resets the environment for a new episode.

        Parameters
        -----------
        mode : int
            -1 is for the training phase, others are for validation/test.

        Returns
        -------
        list
            Initialization of the sequence of observations used for the pseudo-state; dimension must match self.inputDimensions().
            If only the current observation is used as a (pseudo-)state, then this list is equal to self._last_ponctual_observation.
        """
        # if mode == -1:
        #     self.prices = self._price_signal_train
        # else:
        #     self.prices = self._price_signal_valid

        self._last_ponctual_observation = [self._state_to_transform[0]]
        self._counter = 1
        return self._last_ponctual_observation

    def act(self, action):
        """ Performs one time-step within the environment and updates the current observation self._last_ponctual_observation

        Parameters
        -----------
        action : int
            Integer in [0, ..., N_A] where N_A is the number of actions given by self.nActions()

        Returns
        -------
        reward: float
        """

        # action_num = self._transform_to_state[action]
        state_num = self._transform_to_state[ self._last_ponctual_observation[0] ]

        state_action = state_num, action
        reward_now = self._rewards[state_action]
        self._last_ponctual_observation = [self._state_to_transform[ self._transitions[state_action] ]]
        self._counter += 1

        return reward_now

    # def summarizePerformance(self, test_data_set, *args, **kwargs):
    #     """
    #     This function is called at every PERIOD_BTW_SUMMARY_PERFS.
    #     Parameters
    #     -----------
    #         test_data_set
    #     """
    #
    #     print("Summary Perf")
    #
    #     observations = test_data_set.observations()
    #     prices = observations[0][100:200]
    #     invest = observations[1][100:200]
    #
    #     steps = np.arange(len(prices))
    #     steps_long = np.arange(len(prices) * 10) / 10.
    #
    #     # print steps,invest,prices
    #     host = host_subplot(111, axes_class=AA.Axes)
    #     plt.subplots_adjust(right=0.9, left=0.1)
    #
    #     par1 = host.twinx()
    #
    #     host.set_xlabel("Time")
    #     host.set_ylabel("Price")
    #     par1.set_ylabel("Investment")
    #
    #     p1, = host.plot(steps_long, np.repeat(prices, 10), lw=3, c='b', alpha=0.8, ls='-', label='Price')
    #     p2, = par1.plot(steps, invest, marker='o', lw=3, c='g', alpha=0.5, ls='-', label='Investment')
    #
    #     par1.set_ylim(-0.09, 1.09)
    #
    #     host.axis["left"].label.set_color(p1.get_color())
    #     par1.axis["right"].label.set_color(p2.get_color())
    #
    #     plt.savefig("plot.png")
    #     print("A plot of the policy obtained has been saved under the name plot.png")

    def inputDimensions(self):
        return [(1,)] # states are in {1, ..., 10}, so only one dimensional

    def nActions(self):
        return 2  # The environment allows two different actions to be taken at each time step

    def inTerminalState(self):
        return False

    def observe(self):
        return np.array(self._last_ponctual_observation)

    @staticmethod
    def CreateTransitionAndRewardMatrix(n_states):
        # states_t = [2*(i)/(n_states-1) - 1 for i in range(n_states)]
        transitions = np.array([[i+1, 0] for i in range(n_states-1) ] + [[n_states-1, 0]])
        rewards = np.zeros(transitions.shape)
        rewards[0][1] = 0.2
        rewards[-1][0] = 1
        return transitions, rewards

    @staticmethod
    def CreateStateDictionary(n_states):
        dict_state_to_transform = {i : 2*(i)/(n_states-1)-1 for i in range(n_states)}
        dict_transform_to_state = {2 * (i)/(n_states-1)-1 : i for i in range(n_states)}
        return dict_state_to_transform, dict_transform_to_state

def main():
    # Can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv = MyEnv(rng)

    print(myenv.observe())
    observation = myenv.observe()
    print(len(observation))


if __name__ == "__main__":
    main()
