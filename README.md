## 开发安装

项目采用 `src/` 布局。开发时在项目根目录使用 editable 安装一次：

```bash
python -m pip install -e .
```

之后对 `src/nonthradsim/` 下 Python 源码的修改会在下次启动 Python 进程时立即生效，无需重复执行 `pip install .`。运行示例程序也应使用同一个 Python 环境：

```bash
python main.py
```

依赖或 `pyproject.toml` 中的包配置发生变更时，再重新执行上述安装命令。

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

