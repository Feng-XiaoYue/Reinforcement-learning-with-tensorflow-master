**实验一 （run_this_init, step_init）**
actions就是所有可能的操作，一共24*8，每次都在所有的actions集合里选一个action，并更新costs
reward：计算该状态下的cost总和为reward。
结束epoch的条件：每当一个action完成后state不发生改变、或者该action已经执行过了，结束该epoch
结果：不收敛

**实验二 （run_this,step）**
actions也是所有可能的操作，一共24*8，但每次选择的action是为当前查询选择的，一个for循环为所有的查询以此选择一个action
reward：计算该状态下的 (costs总和)*1/(a**方差) = reward
结束epoch的条件：当一个for循环更新完一次所有查询的服务器时，如果此时的状态与for循环之前的状态差异值小于达到给定值、并且该操作执行完之后cost总和增加，结束该epoch
结果：不收敛