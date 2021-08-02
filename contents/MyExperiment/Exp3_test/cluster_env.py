import numpy as np
import pandas as pd
import random
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class Cluster(tk.Tk, object):
    def __init__(self, state_init, server_attribute):
        super(Cluster, self).__init__()
        self.action_space = np.array([[0,0],[0,1],[0,2],[0,3],
                                      [1,0],[1,1],[1,2],[1,3],
                                      [2,0],[2,1],[2,2],[2,3],
                                      [3,0],[3,1],[3,2],[3,3],
                                      [4,0],[4,1],[4,2],[4,3],
                                      [5,0],[5,1],[5,2],[5,3],
                                      [6,0],[6,1],[6,2],[6,3],
                                      [7,0],[7,1],[7,2],[7,3],
                                      [8,0],[8,1],[8,2],[8,3],
                                      [9,0],[9,1],[9,2],[9,3],
                                      [10,0],[10,1],[10,2],[10,3],
                                      [11,0],[11,1],[11,2],[11,3]])
        self.n_actions = len(self.action_space)
        self.cost_matrix = pd.DataFrame(np.array([[0,1,5,12],
                                                  [1,0,4,2],
                                                  [5,4,0,3],
                                                  [12,2,3,0]]),
                                            columns=[0, 1, 2, 3])
        self.server_attribute = server_attribute
        self.QSs = self.read_file()
        self.state_init = state_init

        self.cost_init = self.cost_init()

    def step(self, action, state, costs):
        s = state.copy()
        #action_real[查询，移动到的服务器]
        action_real = self.action_space[action]
        q = action_real[0]
        index_server = action_real[1]
        s.iloc[q, :] = 0
        s.iloc[q, index_server] = 1

        cost_new = self.cost_caculate(q, index_server)
        if cost_new > costs[q]:
            is_better = True
        else:
            is_better = False
            # costs[action_real[0]] = cost_new

        costs[q] = cost_new
        cost_all = self.cost_all(costs)
        reward = self.reward(cost_all, s)
        s_ = s

        return s_, costs, reward, cost_all, is_better


    #判断结束的条件 选择的action在执行之后状态仍然没有变 or 判断状态是否在处与某种情况下，例如负载不平衡
    def is_finish(self):
        # TODO
        return True

    # read the file and store in an array[query,[server1,server2,......]]
    def read_file(self):
        server_attribute = self.server_attribute
        with open("D:\SynologyDrive\Reinforcement-learning-with-tensorflow-master\contents\MyExperiment\Exp3_test\QueryAttribute_test",'r') as f:
            content = f.readlines()
            QSs = []
            for item in content:
                QS = []
                item = item.strip("\n")
                q = item.split(",")[0]
                targetAttribute = item.split(",")[1:]
                targetAttribute = list(map(int, targetAttribute))
                servers = []
                for attribute in targetAttribute:
                    server = server_attribute[server_attribute.loc[:, attribute] == 1].index[0]
                    servers.append(server)
                QS.append(int(q))
                QS.append(servers)
                QSs.append(QS)
        return QSs

    # compute the initial costs array based on the initial state matrix. every element represent the total cost of the query
    def cost_init(self):
        state_init = self.state_init
        # print(len(state_init))
        states = self.state_array(state_init)
        # print(len(states))
        costs = []
        # print(len(state_init))
        for i in range(len(state_init)):
            index_server = states[i][1]
            cost = self.cost_caculate(i,  index_server)
            costs.append(cost)
        return costs


    def cost_caculate(self,q,index_server):
        cost = 0
        for j in range(len(self.QSs[q][1])):
            target_server = self.QSs[q][1][j]
            cost += self.cost_matrix.iloc[index_server, target_server]
        return cost


    # create the initial state matrix（random）


    # compute the total reward based on the costs array
    def cost_all(self, costs):
        cost_all = 0
        for i in range(len(costs)):
            cost_all += costs[i]
        return cost_all

    def reward(self, cost_all, state):
        list = []
        for i in state.columns:
            list.append(state[i].sum())

        load_weight_var = np.var(list)
        reward = (len(state)/cost_all) * self.function(1.1, load_weight_var)
        return reward

    def function(self, a, x):
        y = 100/(a**x)
        return y

    # transform the state matrix into array
    def state_array(self, state):
        states = []
        for i in range(len(state)):
            for j in range(len(state.columns)):
                state_arr = []
                if state.iloc[i, j] == 1:
                    state_arr.append(i)
                    state_arr.append(j)
                    states.append(state_arr)

        return states



def state_init():
    init_state = pd.DataFrame(np.zeros(327*8).reshape(327, 8), columns=[0, 1, 2, 3, 4, 5, 6, 7])
    for i in range(len(init_state)):
        j = random.randint(0, 7)
        init_state.iloc[i][j] = 1
    return init_state

# if __name__ == '__main__':
#     server_attribute = pd.DataFrame(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                                               0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
#                                               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
#                                               0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#                                               0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
#                                               0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
#                                               0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).
#                                     reshape(8, 24),
#                                     columns=np.arange(24))

    # env = Cluster(state_init(), server_attribute)
    # Qss = env.QSs
    # print(Qss)
    # for i in range(len(Qss)):
    #     q = i
    #     for j in range(len(server_attribute)):
    #         index_server = j



    # print(env.cost_init)
    # print("The reward of initial state is:")
    # print(env.reward(env.cost_all(env.cost_init), env.state_init))

    # print(env.state_init)
    # actions=list(range(env.n_actions))
    # print(actions)
    # env.after(100, update)
    # env.mainloop()