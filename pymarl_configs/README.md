# PyMARL 集成说明

为了保持与论文 3.1~3.4 章节完全一致，我们将 MAPPO/IPPO/Greedy 的实现放在本仓库（基于 RLlib + 自研环境），而 QMIX/MADDPG 则按照 3.1.2.2 节的设定在 **PyMARL** 框架中运行。

请按照下列步骤复制参数，并在 PyMARL 中运行实验：

1. **克隆 PyMARL**

   ```bash
   git clone https://github.com/oxwhirl/pymarl.git
   cd pymarl
   ```

2. **复制环境/算法配置**  
   - 将 `pymarl_configs/edge_maddpg.yaml` 和 `pymarl_configs/edge_qmix.yaml` 放入 PyMARL 的 `src/config` 目录。  
   - 这两个配置文件完全复刻了 `EdgeSimGym` 的关键参数（节点拓扑、泊松到达率、任务特征、奖励权重等），只需在 PyMARL 中实现对应的 `edge_marl` 环境即可。

3. **实现 EdgeSimGym 的 PyMARL 版本**
   - 参考本仓库 `MRRL.PY` 中 `EdgeSimGym` 的逻辑：  
     - 节点数 `N=10`，CPU 能力 `[1.0, 1.2, ..., 3.0]` GHz；  
     - 任务到达率在 `[5, 15] tasks/s` 间周期变化，Episode 长度 1000 step；  
     - 奖励函数 `R_t = - (0.5 * 延迟 + 0.3 * 能耗 + 0.2 * 违约)`；  
     - QMIX 只需要离散“放置”动作；MADDPG 需要“放置 + 资源分配”混合动作。

4. **运行命令示例**

   ```bash
   python src/main.py --config=edge_qmix --env-config=edge_marl with seed=0
   python src/main.py --config=edge_maddpg --env-config=edge_marl with seed=0
   ```

   `seed` 需要取 `{0,1,2,3,4}`，每次训练 5M 步，与论文中的设置一致。

5. **导出指标**  
   - 记录平均延迟、P99 延迟、平均能耗、吞吐量、Jain 指数以及每个 seed 的最后评估值，便于与表 3.5/3.6 对应。

> 提示：如果你已经在其它仓库实现了 PyMARL 版本，只需确保以上参数保持一致即可。
