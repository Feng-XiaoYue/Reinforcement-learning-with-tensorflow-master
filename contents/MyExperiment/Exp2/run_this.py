from cluster_env import Cluster
from RL_brain import QLearningTable
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random
import time
from pandas.testing import assert_frame_equal


def update():
    cost_all_list = []
    reward_all_list = []
    query_number = len(env.QSs)
    for episode in episodes:
        epoch_curr_time1 = datetime.datetime.now()

        # initial state
        state_init_arr = env.state_array(env.state_init)
        state = (env.state_init).copy()
        costs = env.cost_init
        sum = 0

        actions = []
        reward_list = []
        state_arr_for_one = []
        while True:
            for q in range(query_number):
                # print("epoch:", episode+1)
                # print("curr_costs:",costs)

                # RL choose action based on observation
                # The action here is a number(the index of the real action)
                action = RL.choose_action(str(state), q)
                # print(action)

                # RL take action and get next observation and reward
                state_, costs_, reward, cost_all, is_better = env.step(action, state, costs)

                reward_list.append(reward)

                costs = costs_

                # q_table=RL.q_table.copy()

                # RL learn from this transition
                RL.learn(str(state), action, reward, str(state_), q)

                # swap observation
                state = state_
                state_arr = env.state_array(state)

                # different = [y for y in (state_init_arr + state_arr) if y not in state_init_arr]

                # print(sum)
                # print(different)
                # print("next_costs:",costs)
                # print("reward:",reward,"\n")
            sum += 1
            # print(sum)
            different = [y for y in (state_arr_for_one + state_arr) if y not in state_arr_for_one]
            state_arr_for_one = state_arr
            if len(different) < 20:
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
        # print("The reward list:", reward_list)
        print("The best reward in this epoch：", max(reward_list))
        print("The final reward in this epoch:", reward)
        print("The final cost in this epoch:", cost_all)
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
    episodes = np.arange(100)
    curr_time1 = datetime.datetime.now()
    env = Cluster()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # env.after(100, update)
    # env.mainloop()
    cost_all_list, reward_all_list = update()
    curr_time2 = datetime.datetime.now()
    train_time = curr_time2-curr_time1
    print("The training time：", train_time)
    y_1 = reward_all_list
    # y_2 = []
    # y_3 = []
    # for i in range(30):
    #     y_one = random.uniform(0.02, 0.045)
    #     y_2.append(y_one)
    # for i in range(40):
    #     y_one = random.uniform(0.035, 0.05)
    #     y_3.append(y_one)

    y_all_list = y_1
    x = episodes
    y = y_all_list
    y1 = [0.02400945043387207]*100
    fig = plt.Figure(figsize=(7, 5))
    pl.plot(x, y, label=u'RL')
    pl.legend()
    pl.plot(x, y1, label=u'RL')
    pl.legend()
    pl.xlabel(u"epoch", size=14)
    pl.ylabel(u"reward", size=14)
    plt.show()
