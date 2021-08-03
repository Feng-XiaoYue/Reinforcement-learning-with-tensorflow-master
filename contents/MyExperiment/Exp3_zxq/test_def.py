
from cluster_env import Cluster
from RL_brain import QLearningTable

if __name__ == '__main__':
    init_state=pd.DataFrame(np.zeros(24*8).reshape(24,8),columns=[0,1,2,3,4,5,6,7])
    for i in range(24):
        j=random.randint(0,7)
        init_state.iloc[i][j]=1
    print(init_state)
    env = Cluster()
    env.reward_init(init_state)