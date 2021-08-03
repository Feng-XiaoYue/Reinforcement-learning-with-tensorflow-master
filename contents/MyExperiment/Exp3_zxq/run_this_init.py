"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from cluster_env import Cluster
from RL_brain import QLearningTable
import datetime
import time
from pandas.testing import assert_frame_equal


def update():
    reward_all_list=[]
    for episode in range(100):
        # initial observation
        state_init_arr=env.state_array(env.state_init)
        state = (env.state_init).copy()
        costs=env.cost_init
        sum=0

        actions=[]
        reward_list=[]

        while True:
            # fresh env
            # env.update()

            # RL choose action based on observation

            # 这里的action是一个数字（真实action的index）
            action = RL.choose_action(str(state))

            # RL take action and get next observation and reward
            state_, costs_, reward, done = env.step(action,state,costs)

            reward_list.append(reward)

            costs=costs_

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))


            # swap observation
            state = state_
            state_arr=env.state_array(state)

            different=[y for y in (state_init_arr + state_arr) if y not in state_init_arr]

            sum+=1
            # print(sum)
            # print(different)
            # print("reward:",reward,"\n")



            if action in actions:
                break
            else:
                actions.append(action)
            # break while loop when end of this episode

            if done:
                break

        reward_all_list.append(reward)
        print("epoch:", episode+1)
        print("该轮循环的次数为：",sum)
        print("本轮最大的reward为：",max(reward_list))
        print("本轮最终的reward为:",reward,"\n")
    # end of game
    # print("final state:\n",state)
    print("------------------------")
    print("最终的state_array为:",state_arr)
    print("最终的reward为:",reward)
    # if state.equals(env.state_init):
    #     print("True")
    # else:
    #     print("False")
        # assert_frame_equal(state,env.state_init)
    env.destroy()

if __name__ == "__main__":
    curr_time1=datetime.datetime.now()
    env = Cluster()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # env.after(100, update)
    # env.mainloop()
    update()
    curr_time2=datetime.datetime.now()
    train_time=curr_time2-curr_time1
    print("训练时间为：",train_time)