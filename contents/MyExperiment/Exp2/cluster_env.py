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
    def __init__(self):
        super(Cluster, self).__init__()
        self.action_space = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],
                                      [1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],
                                      [2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],
                                      [3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],
                                      [4,0],[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],
                                      [5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],
                                      [6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],
                                      [7,0],[7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],
                                      [8,0],[8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],
                                      [9,0],[9,1],[9,2],[9,3],[9,4],[9,5],[9,6],[9,7],
                                      [10,0],[10,1],[10,2],[10,3],[10,4],[10,5],[10,6],[10,7],
                                      [11,0],[11,1],[11,2],[11,3],[11,4],[11,5],[11,6],[11,7],
                                      [12,0],[12,1],[12,2],[12,3],[12,4],[12,5],[12,6],[12,7],
                                      [13,0],[13,1],[13,2],[13,3],[13,4],[13,5],[13,6],[13,7],
                                      [14,0],[14,1],[14,2],[14,3],[14,4],[14,5],[14,6],[14,7],
                                      [15,0],[15,1],[15,2],[15,3],[15,4],[15,5],[15,6],[15,7],
                                      [16,0],[16,1],[16,2],[16,3],[16,4],[16,5],[16,6],[16,7],
                                      [17,0],[17,1],[17,2],[17,3],[17,4],[17,5],[17,6],[17,7],
                                      [18,0],[18,1],[18,2],[18,3],[18,4],[18,5],[18,6],[18,7],
                                      [19,0],[19,1],[19,2],[19,3],[19,4],[19,5],[19,6],[19,7],
                                      [20,0],[20,1],[20,2],[20,3],[20,4],[20,5],[20,6],[20,7],
                                      [21,0],[21,1],[21,2],[21,3],[21,4],[21,5],[21,6],[21,7],
                                      [22,0],[22,1],[22,2],[22,3],[22,4],[22,5],[22,6],[22,7],
                                      [23,0],[23,1],[23,2],[23,3],[23,4],[23,5],[23,6],[23,7]])
        self.n_actions = len(self.action_space)
        self.cost_matrix = pd.DataFrame(np.array([[0,1,5,12,7,10,15,9],
                                                [1,0,4,2,8,6,11,10],
                                                [5,4,0,3,11,13,8,5],
                                                [12,2,3,0,7,6,10,4],
                                                [7,8,11,7,0,12,9,5],
                                                [10,6,13,6,12,0,3,8],
                                                [15,11,8,10,9,3,0,10],
                                                [9,10,5,4,5,8,10,0]]),
                                      columns = [0, 1, 2, 3, 4, 5, 6, 7])
        self.QSs = self.read_file()
        self.state_init = pd.DataFrame(np.array([1,0,0,0,0,0,0,0,
                                               0,1,0,0,0,0,0,0,
                                               0,0,0,0,1,0,0,0,
                                               0,0,0,0,0,1,0,0,
                                               0,0,0,0,0,1,0,0,
                                               0,0,1,0,0,0,0,0,
                                               0,0,0,0,0,0,1,0,
                                               0,0,0,1,0,0,0,0,
                                               0,0,0,0,1,0,0,0,
                                               0,0,0,0,1,0,0,0,
                                               0,0,1,0,0,0,0,0,
                                               0,0,0,0,0,1,0,0,
                                               1,0,0,0,0,0,0,0,
                                               1,0,0,0,0,0,0,0,
                                               0,0,0,0,0,0,0,1,
                                               1,0,0,0,0,0,0,0,
                                               0,0,0,0,0,0,1,0,
                                               0,0,1,0,0,0,0,0,
                                               0,0,0,1,0,0,0,0,
                                               0,0,0,1,0,0,0,0,
                                               0,0,0,0,0,0,1,0,
                                               0,0,0,0,0,0,0,1,
                                               0,0,0,0,1,0,0,0,
                                               0,0,0,0,1,0,0,0]).
                                               reshape(24, 8),
                                               columns = [0, 1, 2, 3, 4, 5, 6, 7])

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
        with open("D:\SynologyDrive\Reinforcement-learning-with-tensorflow-master\contents\MyExperiment\Exp2\QueryTargetServer",'r') as f:
            content = f.readlines()
            QSs = []
            for item in content:
                QS = []
                item = item.strip("\n")
                q, targetServers = item.split(" ")
                targetServers = targetServers.split(",")
                targetServers = list(map(int, targetServers))
                QS.append(int(q))
                QS.append(targetServers)
                QSs.append(QS)
        return QSs

    # compute the initial costs array based on the initial state matrix. every element represent the total cost of the query
    def cost_init(self):
        state_init = self.state_init
        states = self.state_array(state_init)
        costs = []
        for i in range(24):
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
    def state_init(self):
        init_state = pd.DataFrame(np.zeros(24*8).reshape(24, 8), columns = [0, 1, 2, 3, 4, 5, 6, 7])
        for i in range(24):
            j = random.randint(0, 7)
            init_state.iloc[i][j] = 1
        return init_state

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
        reward = (24/cost_all) * self.function(1.4, load_weight_var)
        return reward

    def function(self, a, x):
        y = 1/(a**x)
        return y

    # transform the state matrix into array
    def state_array(self, state):
        states = []
        for i in range(24):
            for j in range(8):
                state_arr = []
                if state.iloc[i, j] == 1:
                    state_arr.append(i)
                    state_arr.append(j)
                    states.append(state_arr)
        return states





if __name__ == '__main__':
    env = Cluster()
    # print(env.cost_init)
    print("The reward of initial state is:")
    print(env.reward(env.cost_all(env.cost_init), env.state_init))

    # print(env.state_init)
    # actions=list(range(env.n_actions))
    # print(actions)
    # env.after(100, update)
    # env.mainloop()