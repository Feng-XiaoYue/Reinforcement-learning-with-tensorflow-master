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
        self.QSs=self.read_file()
        self.state_init=self.state_init()


    def read_file(self):
        with open("/contents/MyExperiment/Exp1/QueryTargetServer", 'r') as f:
            content=f.readlines()
            QSs=[]
            for item in content:
                QS=[]
                item=item.strip("\n")
                q,targetServers=item.split(" ")
                targetServers=targetServers.split(",")
                targetServers=list(map(int,targetServers))
                QS.append(int(q))
                QS.append(targetServers)
                QSs.append(QS)
        return QSs

    def test_qss(self):
        print(self.QSs)

    def state_init(self):
        init_state=pd.DataFrame(np.zeros(24*8).reshape(24,8),columns=[0,1,2,3,4,5,6,7])
        for i in range(24):
            j=random.randint(0,7)
            init_state.iloc[i][j]=1
        return init_state

    def test_state(self):
        print(self.state_init)

if __name__ == '__main__':
    env=Cluster()
    env.test_state()