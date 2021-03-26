# import matplotlib;
#
# matplotlib.use("TkAgg")

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from mpl_toolkits.axes_grid1 import make_axes_locatable


class Gambler(gym.Env):
    """
    The gamblers problem from Sutton and Barto 2020 Exercise 4.2
    """

    def __init__(self, target=100, head_prob=0.3, gamma=1):
        # amount to reach
        self.fig = None
        self.pi_history = []
        self.vi_history = []
        self.gamma = gamma
        self.TARGET = target

        # all states, including state 0 and state Target
        self.observation_space = gym.spaces.Discrete(self.TARGET + 1)
        # init initial state
        self.current_state = None

        # probability of head
        self.HEAD_PROB = head_prob

        # state value array (everywhere zero) - increase precision
        # self.state_value = np.zeros(self.observation_space.n, dtype=np.float64)
        # final state value

        self.max_action = None

        # we have perfect knowledge about the environment - let's define the probabilities
        def actions(state):
            return range(min(state, self.TARGET - state) + 1)

        self.P = {s: {a: [] for a in actions(s)} for s in range(self.observation_space.n)}

        for state in self.P:
            for action in self.P[state]:
                prob_next_state_and_r = self.HEAD_PROB
                probs = [(prob_next_state_and_r, state + action,
                          int(state + action >= self.TARGET),
                          state + action >= self.TARGET),
                         (1 - prob_next_state_and_r, state - action, 0, state - action <= 0)]
                # terminal states
                if state == self.TARGET:
                    probs = [(1, state, 1,
                              True)]
                elif state == 0:
                    probs = [(1, state, 0,
                              True)]
                self.P[state][action] = probs
        # print(self.P)

    def step(self, action):
        # if we want to sample from the environment

        # get possible actions for current state
        assert (action <= self.max_action) and (action >= 0), "Action not valid"
        self.current_state = (self.current_state + action) \
            if random.random() <= self.HEAD_PROB else (self.current_state - action)
        done = self.current_state >= self.TARGET or self.current_state <= 0
        reward = done
        if not done:
            self.max_action = min(self.current_state, self.TARGET - self.current_state) + 1
        return self.current_state, reward, done, {}

    def reset(self, set_state=None):
        if set_state is not None:
            self.current_state = set_state
        else:
            # init returns a random state
            self.current_state = self.observation_space.sample()
        self.max_action = min(self.current_state, self.TARGET - self.current_state) + 1
        return self.current_state

    def value_iteration(self):
        # We find the optimal state-value function for the problem
        sweeps_history = []
        n_iter_max = 10000
        # important how accurate the approximation is
        threshold = 1e-20
        # value iteration - increase slightly the accuracy
        state_value = np.zeros(self.observation_space.n, dtype=np.float64)
        for n_iter in range(n_iter_max):
            # On each iteration, copy the value table to the updated_value_table
            updated_value_table = np.copy(state_value)
            sweeps_history.append(updated_value_table)
            _, Q_array, policy_array = self.policy_improvement(state_value)
            self.vi_history.append([state_value,Q_array, policy_array] )
            # Now we calculate Q Value for each actions in the state
            # and update the value of a state with maximum Q value

            for state in self.P:
                # initialize the Q table for a state
                Q_table = np.zeros(len(self.P[state]))
                for action in self.P[state]:
                    next_states_rewards = []
                    for next_sr in self.P[state][action]:
                        trans_prob, next_state, reward_prob, done = next_sr
                        Q_table[action] += (trans_prob * (
                                reward_prob + self.gamma * (state_value[next_state] if not done else 0)))
                state_value[state] = max(Q_table)
            # important what norm is chosen! Here we take the taxi-cab norm
            if np.sum(np.fabs(updated_value_table - state_value)) <= threshold:
                sweeps_history.append(state_value)
                _, Q_array, policy_array = self.policy_improvement(state_value)
                self.vi_history.append([state_value, Q_array, policy_array])
                print(f'Value-iteration converged at iteration {(n_iter + 1)}')
                break

        return sweeps_history

    def policy_improvement(self, state_value=None):
        #
        # if state_value is None:
        #     state_value = self.state_value
        # print('start policy improvement')
        # initialize the policy with zeros
        policy = np.zeros(self.observation_space.n)
        Q_array = np.zeros((self.observation_space.n, self.observation_space.n))
        policy_array = np.zeros((self.observation_space.n, self.observation_space.n))

        for state in self.P:
            # initialize the Q table for a state
            # Q_table = np.zeros(len(self.P[state]))
            Q_table = np.zeros(len(self.P[state]))
            # compute Q value for all actions in the state
            for action in self.P[state]:
                for next_sr in self.P[state][action]:
                    trans_prob, next_state, reward_prob, done = next_sr
                    Q_table[action] += (
                            trans_prob * (reward_prob + self.gamma * state_value[next_state] * (1 - done)))
                # rounding is important to minimize numerical effects
                Q_table[action] = Q_table[action].round(10)
                Q_array[action, state] = Q_table[action]
            # dont't forget to remove the non-terminal policies - (zero action)
            if len(Q_table) > 1 and not self.gamma < 1:
                indices = np.where(Q_table[1:] == Q_table[1:].max())[0] + 1
            else:
                indices = np.where(Q_table == Q_table.max())[0]
            policy_array[indices, state] = 1

            # select the action which has maximum Q value as an optimal action of the state
            # policy[state] = random.choice(indices)
            policy[state] = indices[0]
        # print('Q:\n', Q_array)
        # print('new policy: ', policy)

        # self.plot_current_policy(policy_array, Q_array, state_value, fig=self.fig)
        return policy, Q_array, policy_array

    def policy_evaluation(self, policy, state_value=None):
        """Inplace policy evaluation. The existence and uniqueness of v are guaranteed as long as either gamma  < 1
        or eventual termination is guaranteed from all states under the policy pi. We have to exclude actions which
        lead to an infinite horizon or set gamma < 1 """
        if state_value is None:
            # initialize value table with zeros
            state_value = np.zeros(self.observation_space.n)

        # set the threshold
        threshold = 1e-20
        max_number_sweeps = 20000

        for sweep_nr in range(max_number_sweeps):
            delta = 0
            # for each state in the environment, select the action according to the policy and compute the value table
            for state in self.P:
                v = state_value[state]
                action = policy[state]
                # build the value table with the selected action
                state_value[state] = sum([trans_prob * (reward +
                                                        self.gamma * (
                                                                state_value[next_state] * (1 - done)))
                                          for trans_prob, next_state, reward, done in self.P[state][action]])
                delta = max(delta, abs(v - state_value[state]))
            if delta < threshold: break

        return state_value, sweep_nr

    def policy_iteration(self):
        # Initialize policy with zeros
        old_policy = np.zeros(self.observation_space.n)
        new_value_function = np.zeros(self.observation_space.n)
        no_of_iterations = 20000
        self.policy_improvement(state_value=np.zeros(self.observation_space.n))

        print('start v: ', new_value_function)
        print('start p: ', old_policy)
        for i in range(no_of_iterations):

            # compute the value function
            new_value_function, sweep_nr = self.policy_evaluation(old_policy)  # , new_value_function)
            print('new v: ', new_value_function, np.mean(new_value_function))
            # Extract new policy from the computed value function
            new_policy, Q_array, policy_array = self.policy_improvement(new_value_function)

            self.pi_history.append([new_value_function, Q_array, policy_array])
            # self.update_plot(len(self.history)-1)
            print('new p: ', new_policy)
            # Then we check whether we have reached convergence i.e whether we found the optimal
            # policy by comparing old_policy and new policy if it same we will break the iteration
            # else we update old_policy with new_policy

            if (np.all(old_policy == new_policy)):
                print('Policy-Iteration converged at step %d.' % (i + 1))
                break
            old_policy = new_policy
        print("final p", new_policy)
        return new_policy

    def init_plot(self, fig=None, label=None):
        """Create figure for plotting """
        if fig is None:
            if self.fig is None:
                self.fig = plt.figure()
            fig = self.fig
        else:
            fig = fig

        # dummy elements
        Q_array = np.zeros((self.observation_space.n, self.observation_space.n))
        policy_array = np.zeros((self.observation_space.n, self.observation_space.n))
        state_value = np.zeros(self.observation_space.n)

        # ax = plt.subplot(221)
        self.ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=3, fig=fig)
        ax1 = self.ax1
        ax1.set_xlabel("state")
        ax1.set_ylabel("action(state)")
        extent = (0, self.TARGET, 0, self.TARGET)
        im1 = ax1.imshow(np.flipud(policy_array), cmap=plt.cm.hot, origin='upper', extent=extent)
        ax1.set_ylim(0, int(1 + self.TARGET / 2))
        # ax = plt.subplot(222, sharey=ax)
        self.ax2 = plt.subplot2grid((4, 2), (0, 1), sharey=ax1, rowspan=3)
        ax2 = self.ax2
        im2 = ax2.imshow(np.flipud(Q_array), cmap=plt.get_cmap('turbo'), origin='upper', extent=extent)

        divider = make_axes_locatable(ax2)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)
        cax = self.cax
        cbar = plt.colorbar(im2, cax=cax)
        cbar.ax.set_ylabel('Q(s,a)')
        ax2.set_xlabel("state")
        ax2.set_ylabel("action(state)")
        # ax = plt.subplot(212)
        self.ax3 = plt.subplot2grid((4, 2), (3, 0), colspan=2)
        ax3 = self.ax3
        # ax.matshow([state_value])
        ax3.plot(state_value)
        ax3.set_xlabel('state')
        ax3.set_ylabel('state value function')
        plt.yticks([])

        if label is None:
            label = f'$P_h$={gambler.HEAD_PROB}'

        fig.suptitle(label)
        fig.tight_layout()
        return fig

    def init_func(self):
        """needed for the animation"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        # pass
        # for ax in self.fig.get_axes():
        #     ax.clear()

    def plot_current_policy(self, policy_array, Q_array, state_value, label=None):
        fig = self.fig
        # ax = plt.subplot(221)
        # ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=3, fig=fig)
        ax1 = self.ax1
        ax1.set_xlabel("state")
        ax1.set_ylabel("action(state)")
        extent = (0, self.TARGET, 0, self.TARGET)
        im1 = ax1.imshow(np.flipud(policy_array), cmap=plt.cm.hot, origin='upper', extent=extent)
        ax1.set_ylim(0, int(1 + self.TARGET / 2))
        # ax = plt.subplot(222, sharey=ax)
        ax2 = self.ax2
        im2 = ax2.imshow(np.flipud(Q_array), cmap=plt.get_cmap('turbo'), origin='upper', extent=extent)

        divider = make_axes_locatable(ax2)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        cax = self.cax
        cbar = plt.colorbar(im2, cax=cax)
        cbar.ax.set_ylabel('Q(s,a)')
        ax2.set_xlabel("state")
        ax2.set_ylabel("action(state)")
        # ax = plt.subplot(212)
        ax3 = self.ax3
        # ax.matshow([state_value])
        ax3.plot(state_value)
        ax3.set_xlabel('state')
        ax3.set_ylabel('state value function')
        plt.yticks([])

        if label is None:
            label = f'$P_h$={gambler.HEAD_PROB}'

        fig.suptitle(label)
        fig.tight_layout()

        # self.ims.append([])
        # fig.show()

    def update_plot(self, values, iteration):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.cax.clear()
        state_value, Q_array, policy_array = values
        self.plot_current_policy(policy_array=policy_array,
                                 Q_array=Q_array,
                                 state_value=state_value, label=f'$P_h$={gambler.HEAD_PROB} it: {iteration}')
        # self.fig.show()
        # plt.pause(.1)


if __name__ == '__main__':
    from matplotlib.animation import FFMpegWriter
    head_prob = 0.4
    target = 30

    # matplotlib.use("TkAgg")
    gambler = Gambler(head_prob=head_prob, target=target)

    fig = gambler.init_plot()
    gambler.policy_iteration()

    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=1, metadata=metadata)

    with writer.saving(fig, f'images/Gambler PI head_prob {head_prob} target {target}.mp4', 250):
        for i, values in enumerate(gambler.pi_history):
            gambler.update_plot(values, i)
            # fig.show()
            writer.grab_frame()

    fig = gambler.init_plot()
    gambler.value_iteration()

    with writer.saving(fig, f'images/Gambler VI head_prob {head_prob} target {target}.mp4', 250):
        for i, values in enumerate(gambler.vi_history):
            gambler.update_plot(values, i)
            # fig.show()
            writer.grab_frame()

    fig.savefig(f'images/Gambler PI head_prob {head_prob} target {target}.pdf')
    fig.savefig(f'images/Gambler PI head_prob {head_prob} target {target}.png')
    fig.show()


