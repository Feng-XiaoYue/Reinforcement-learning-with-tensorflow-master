from cluster_env import Cluster
import pandas as pd
import numpy as np
import datetime


if __name__ == '__main__':
    curr_time1=datetime.datetime.now()
    env=Cluster()
    cost_matrix=env.cost_matrix
    dataset=env.QSs
    reward=0
    init_state=pd.DataFrame(np.zeros(24*8).reshape(24,8),columns=[0,1,2,3,4,5,6,7])
    for q in range(24):
        one_q_cost_list=[]
        for index_server in range(8):
            cost=env.cost_caculate(q,index_server)
            one_q_cost_list.append(cost)
        one_q_best_index_server=one_q_cost_list.index(max(one_q_cost_list))
        init_state.iloc[q,one_q_best_index_server]=1
        one_q_best_cost=max(one_q_cost_list)
        reward+=one_q_best_cost
    state_array=env.state_array(init_state)
    
    print("state_arr:", state_array)
    print(reward)

    curr_time2=datetime.datetime.now()
    caculate_time=curr_time2-curr_time1
    print("计算时间为：",caculate_time)