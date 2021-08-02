import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pylab as pl

import tensorflow as tf

print(tf.__version__)

# if __name__ == '__main__':

#     将我自己设置的训练集（也就是每个查询对应需要访问的机器）从文件中读出放进数组中
#     with open("D:\SynologyDrive\Reinforcement-learning-with-tensorflow-master\contents\MyExperiment\QueryTargetServer",'r') as f:
#         content=f.readlines()
#         QSs=[]
#         for item in content:
#             QS=[]
#             item=item.strip("\n")
#             q,targetServers=item.split(" ")
#             targetServers=targetServers.split(",")
#             targetServers=list(map(int,targetServers))
#             QS.append(int(q))
#             QS.append(targetServers)
#             QSs.append(QS)
#         print(QSs)


#     # 初始化状态（每个查询对应的服务器）
#     init_state=pd.DataFrame(np.zeros(24*8).reshape(24,8),columns=[0,1,2,3,4,5,6,7])
#     for i in range(24):
#         j=random.randint(0,7)
#         init_state.iloc[i][j]=1
#     print(init_state)


    # 生成数据集（每个查询对应需要访问的服务器）
    # for i in range (24):
    #     n=random.randint(2,4)
    #     array=random.sample(range(8),n)
    #     print(i,array)

    # for i in range (192):
    #     print(str(i)+",")



    # 设置每两个服务器之间的cost值
    # cost_matrix=pd.DataFrame(np.array([[0,1,5,12,7,10,15,9],
    #                                    [1,0,4,2,8,6,11,10],
    #                                    [5,4,0,3,11,13,8,5],
    #                                    [12,2,3,0,7,6,10,4],
    #                                    [7,8,11,7,0,12,9,5],
    #                                    [10,6,13,6,12,0,3,8],
    #                                    [15,11,8,10,9,3,0,10],
    #                                    [9,10,5,4,5,8,10,0]]),
    #                          columns=[0,1,2,3,4,5,6,7])
    # print(cost_matrix.iloc[0,1])


#     从状态矩阵中获取每个查询放置的对应的服务器
#     states=[]
#     for i in range(24):
#         for j in range(8):
#             state=[]
#             if init_state.iloc[i,j]==1:
#                 state.append(i)
#                 state.append(j)
#                 states.append(state)
#     print(states)


#     计算所有的cost（按照每个查询的cost计算总cost）
#         reward=0
#         costs=[]
#         for i in range(24):
#             cost=0
#             index_server=states[i][1]
#             print("index_server:",index_server)
#             for j in range(len(QSs[i][1])):
#                 target_server=QSs[i][1][j]
#                 cost+=cost_matrix.iloc[index_server,target_server]
#
#                 #     reward+=cost
#                 print("target_server:",target_server)
#             costs.append(cost)
#         print("costs:",costs)
#         for i in range(len(costs)):
#             reward+=costs[i]
#         print("reward:",reward)

    # cost=[1,3,5,7,3]
    # if 8>cost[0]:
    #     cost[0]=8
    #     reward=1
    # else:
    #     print(cost)
    #     reward=-1
    # print(reward)
    # print(cost)
    # list = [0,1,2,3,4,5,6,7,8,9]
    # fig = plt.Figure(figsize=(7, 5))
    # np1 = np.arange(10)
    # x = np1
    # y = list
    # pl.plot(x, y, label=u'Baseline')
    # pl.legend()
    # pl.xlabel(u"epoch", size=14)
    # pl.ylabel(u"reward", size=14)
    # plt.show()
    #

    # init_state = pd.DataFrame(np.zeros(8*24).reshape(8, 24), columns = np.arange(24))
    # server = []
    # attribute_list = range(24)
    # for i in range(8):
    #     attribute_list = list(set(attribute_list).difference(set(server)))
    #     server = random.sample(attribute_list, 3)
    #     for j in server:
    #         init_state.iloc[i][j] = 1
    # print(init_state)



