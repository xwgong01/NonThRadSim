## Class:
0.0 spatialgrid
    定义三维网格，101 * 101 * 101
    分别为x,y,z,E

0.1 1d grid
    分为energy grid(101)
    以及direction(3)


1. electrons    
   在grid上，保存electron空间、能量分布
   可以给定/通过别的计算

2. photon
   在grid上，保存photon的空间能量分布。初始化为0

3. magnetic_field
   在3d grid上定义每个空间点的磁场方向

## 函数
1. syn(electrons, magnetic_field, photonSyn)
   对电子和磁场分布，计算synchrotron的空间+能量分布，存到photon里

2. IC(electrons, photonBKG, photonIC)
   同.

3. projection(photon_total, grid_obs)
   把光子的空间分布投影到观测者系，给定观测者grid。用插值。


