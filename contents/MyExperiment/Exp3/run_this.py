from cluster_env import Cluster
from RL_brain import QLearningTable
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random
import pandas as pd
import time
from pandas.testing import assert_frame_equal

def state_init():
    init_state = pd.DataFrame(np.zeros(327*8).reshape(327, 8), columns=[0, 1, 2, 3, 4, 5, 6, 7])
    for i in range(len(init_state)):
        j = random.randint(0, 7)
        init_state.iloc[i][j] = 1
    return init_state

def server_attribute():
    init_state = pd.DataFrame(np.zeros(8*24).reshape(8, 24), columns=np.arange(24))
    server = []
    attribute_list = range(len(init_state.columns))
    for i in range(len(init_state)):
        attribute_list = list(set(attribute_list).difference(set(server)))
        server = random.sample(attribute_list, 3)
        for j in server:
            init_state.iloc[i][j] = 1
    return init_state

def update():
    cost_all_list = []
    reward_all_list = []
    query_number = len(env.QSs)
    # print(query_number)
    for episode in episodes:
        epoch_curr_time1 = datetime.datetime.now()
        # initial state
        state_init_arr = env.state_array(env.state_init)
        state = (env.state_init).copy()
        costs = env.cost_init
        sum = 0
        reward_list = [0]
        state_arr_for_one = state_init_arr
        reward = init_reward
        while True:
            # RL choose action based on observation
            # The action here is a number(the index of the real action)
            action = RL.choose_action(str(state))
            # print(action)

            # RL take action and get next observation and reward
            state_, costs_, reward_, cost_all, is_better = env.step(action, state, costs)

            state_arr = env.state_array(state_)
            different = [y for y in (state_arr_for_one + state_arr) if y not in state_arr_for_one]
            print("diffrent:", different)
            state_arr_for_one = state_arr
            different_init = [y for y in (state_init_arr + state_arr) if y not in state_init_arr]

            if ((reward_ < init_reward and reward_ < min(reward_list) or
                (len(different) == 0 and reward_ >= reward and reward_ > (init_reward)))):
                done = True
            else:
                done = False
            # RL learn from this transition
            print("done:", done)

            if ((reward_ < init_reward) and (reward_ < min(reward_list))):
                print("reward值小于初始值或并且该循环的最小值")

            if len(different) == 0 and reward_ >= reward and reward_ > (init_reward*1.3):
                print("reward值大于前一个循环的reward值并且采取动作后状态不改变")

            if reward_ < init_reward:
                reward_ = -1

            reward = reward_

            reward_list.append(reward)

            RL.learn(str(state), action, reward, str(state_), done)

            costs = costs_

            # q_table=RL.q_table.copy()
            state = state_

            sum += 1
            # print(sum)

            if done:
                break

        reward_all_list.append(reward)
        epoch_curr_time2 = datetime.datetime.now()
        epoch_time = epoch_curr_time2 - epoch_curr_time1
                # if (action in actions and q_table.loc[str(state), action] >= 0) and (done and q_table.loc[str(state), action] >= 0 and reward > 0):
                #     break
                # else:
                #     actions.append(action)
                # break while loop when end of this episode

                # if done and q_table.loc[str(state),action]!=0:
                #     break

        cost_all_list.append(cost_all)
        print("epoch:", episode+1)
        print("The number of cycles in this epoch：", sum)
        print("The reward list:", reward_list)
        print("The best reward in this epoch：", max(reward_list))
        print("The final reward in this epoch:", reward)
        print("The final cost in this epoch:", cost_all)
        print("当前状态与初始状态的差别", (different_init))
        print("当前状态与初始状态的差别数", len(different_init))
        print("epoch_time:", epoch_time, "\n")
    # end of game
    # print("final state:\n",state)
    print("------------------------")
    print("The final state_array:", state_arr)
    print("The final cost:", cost_all)
    # if state.equals(env.state_init):
    #     print("True")
    # else:
    #     print("False")
        # assert_frame_equal(state,env.state_init)
    # env.destroy()
    return cost_all_list, reward_all_list

if __name__ == "__main__":
    improve_list = []
    test_number = 50
    state_init = state_init()
    # for i in range(50):
    # print("第%d次测试：" % (i+1))
    episodes = np.arange(20000)
    curr_time1 = datetime.datetime.now()

    # print(len(state_init))
    server_attribute = pd.DataFrame(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                              1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                              0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                              0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).
                                    reshape(8, 24),
                                    columns=np.arange(24))

    env = Cluster(state_init, server_attribute)
    init_reward = env.reward(env.cost_all(env.cost_init), env.state_init)
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # env.after(100, update)
    # env.mainloop()
    cost_all_list, reward_all_list = update()
    curr_time2 = datetime.datetime.now()
    train_time = curr_time2-curr_time1
    print("The training time：", train_time)
    print("\n")
    improve = ((reward_all_list[-1] - init_reward)/init_reward)*100
    print("The improve percent:", improve, "%")
    improve_list.append(improve)

    y_1 = reward_all_list
    y_all_list = y_1
    x = episodes
    y = y_all_list
    y1 = [init_reward]*len(episodes)
    fig = plt.Figure(figsize=(7, 5))
    pl.plot(x, y, label=u'RL')
    pl.legend()
    pl.plot(x, y1, label=u'Init')
    pl.legend()
    pl.xlabel(u"epoch", size=14)
    pl.ylabel(u"reward", size=14)
    plt.show()
    # y = improve_list
    # x = np.arange(test_number)
    # fig = plt.Figure(figsize=(7, 5))
    # pl.plot(x, y, label=u'Improve')
    # pl.legend()
    # plt.show()