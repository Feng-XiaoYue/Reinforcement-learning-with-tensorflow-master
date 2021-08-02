from cluster_env import Cluster
from RL_brain import QLearningTable
import datetime
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import time
from pandas.testing import assert_frame_equal


def update():
    reward_all_list=[]
    for episode in range(100):
        # initial state
        state_init_arr = env.state_array(env.state_init)
        state = (env.state_init).copy()
        costs = env.cost_init
        sum = 0

        actions = []
        reward_list = []

        while True:
            # fresh env
            # env.update()

            # print("epoch:", episode)
            # print("curr_costs:",costs)

            # RL choose action based on observation
            # The action here is a number(the index of the real action)
            action = RL.choose_action(str(state))

            # RL take action and get next observation and reward
            state_, costs_, reward, done = env.step(action, state, costs)

            reward_list.append(reward)

            costs = costs_

            q_table = RL.q_table.copy()

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))

            # swap observation
            state = state_
            state_arr = env.state_array(state)

            different = [y for y in (state_init_arr + state_arr) if y not in state_init_arr]

            sum += 1

            print(sum)
            # print(different)
            # print("next_costs:",costs)
            # print("reward:",reward,"\n")


            # if (action in actions and q_table.loc[str(state), action] >= 0) and (done and q_table.loc[str(state), action] >= 0 and reward > 0):
            if (action in actions) and done:
                break
            else:
                actions.append(action)
            # break while loop when end of this episode

            # if done and q_table.loc[str(state),action]!=0:
            #     break

        reward_all_list.append(reward)
        print("epoch:", episode+1)
        print("The number of cycles in this epoch：", sum)
        print("The reward list:", reward_list)
        print("The best reward in this epoch：", max(reward_list))
        print("The final reward in this epoch:", reward, "\n")
    # end of game
    # print("final state:\n",state)
    print("------------------------")
    print("The final state_array:", state_arr)
    print("The final reward:", reward)
    # if state.equals(env.state_init):
    #     print("True")
    # else:
    #     print("False")
        # assert_frame_equal(state,env.state_init)
    # env.destroy()
    return reward_all_list

if __name__ == "__main__":
    curr_time1 = datetime.datetime.now()
    env = Cluster()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # env.after(100, update)
    # env.mainloop()
    reward_all = update()
    # reward_all = update()
    curr_time2 = datetime.datetime.now()
    train_time = curr_time2-curr_time1
    print("The training time：", train_time)
    x = np.arange(100)
    y = reward_all
    fig = plt.Figure(figsize=(7, 5))
    pl.plot(x, y, label=u'RL')
    pl.legend()
    pl.xlabel(u"epoch")
    pl.ylabel(u"reward")
    plt.show()