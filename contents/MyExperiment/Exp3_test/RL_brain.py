import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        # if episode <= 50:
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # print(q*8, (q+1)*8-1)
            # print(state_action)
            # print(state_action[state_action == np.max(state_action)].index)
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        # else:
        #     if np.random.uniform() < 0.95:
        #         # choose best action
        #         state_action = self.q_table.loc[state, :]
        #         # some actions may have the same value, randomly choose on in these actions
        #         action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        #     else:
        #         # choose random action
        #         action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if done:
            q_target = r           # next state is terminal
        else:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()    # next state is not terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)
        print(self.q_table)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # print("This state has not appeared")
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )
        # else:
            # print("This state has appeared")