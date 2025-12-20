### 在 Compshare（Light-GPU）租算力跑 QMIX：从 0 到出结果（按你当前项目口径）

你给的平台入口：[`console.compshare.cn/light-gpu/console/images`](https://console.compshare.cn/light-gpu/console/images)

这份指南以**你当前项目的最新口径**为准（和你已经跑完的 MAPPO 对齐）：
- **早停指标**：QMIX/MAPPO 都按 **reward/return** 早停（QMIX：`early_stop_metric=return`）
- **评估**：`test_interval=50_000`，`test_nepisode=5`（与 MAPPO 一致）
- **日志降频**：`log_interval/runner_log_interval/learner_log_interval=20_000`
- **并行**：用 `python run_qmix_batch.py --shard 0/2` + `--shard 1/2` 两窗口并行

---

### 1) 你需要先决定的 3 件事（决定怎么买最划算）
- **OS**：强烈建议 **Linux（Ubuntu 20.04/22.04）**。Windows 远程租机更麻烦、IO/权限坑更多。
- **GPU**：QMIX 本身显存占用不大，但你会并行跑 2 个进程，所以建议：
  - **最小可用**：单卡 12GB（如 3060/4070 等级）也能跑
  - **更稳更快**：单卡 24GB（如 4090/A10）更适合同时跑 2 个 shard
- **CPU 核心**：你环境模拟也吃 CPU，建议 **8~16 vCPU**（越多越稳），内存 **>= 32GB**。

> 经验法则：**单卡 + 16 vCPU + 32GB RAM + 100GB 磁盘**，对你当前 QMIX batch 很友好。

---

### 2) 在平台上怎么选镜像（Images 页）
进入 Images 页：[`console.compshare.cn/light-gpu/console/images`](https://console.compshare.cn/light-gpu/console/images)

优先选这类镜像（名字可能不完全一致，选“最接近”的）：
- **Ubuntu 22.04 + CUDA（>=12.1）+ cuDNN**（最好自带）
- **Python/Conda**：有 conda 更省事（没有也没关系）

如果你不确定该选哪个镜像：在 Images 页把“镜像列表截图”发我，我会按你项目需求告诉你选哪一个。

---

### 3) 创建实例后：登录机器（SSH）
平台一般会提供：
- 公网 IP（或域名）
- 用户名（常见 `root` / `ubuntu`）
- 密码或 SSH Key

Windows 上建议用 PowerShell：
```powershell
ssh 用户名@公网IP
```

（如果要指定私钥）
```powershell
ssh -i C:\path\to\key.pem 用户名@公网IP
```

---

### 4) 把代码同步到租机：强烈建议用 GitHub 固定版本
**为什么要 GitHub**：租机最怕“跑到一半发现口径变了/代码不一致/忘了改哪”，Git commit 是最稳的回滚点。

你本地现在已经是稳定版本（reward早停+评估对齐+日志降频+shard并行），建议先把它 push 到你的 GitHub 仓库，再在租机上 clone。

租机上：
```bash
git clone <你的仓库地址>
cd MARL边缘调度实验
```

---

### 5) 装依赖（推荐 conda，和你本地 marl 环境保持一致）
如果租机有 conda：
```bash
conda create -n marl python=3.10 -y
conda activate marl
```

安装 PyMARL 依赖（按仓库文件）：
```bash
pip install -r pymarl/requirements.txt
```

检查 GPU 是否可用：
```bash
python -c "import torch; print('torch',torch.__version__); print('cuda_available',torch.cuda.is_available()); print('device_count',torch.cuda.device_count())"
```

> 注意：你本地跑出来的 sacred 依赖显示过 `torch==2.4.1+cu121`。租机上只要 `torch.cuda.is_available()==True` 就算 OK。

---

### 6) 开跑：两路 shard 并行（推荐）
进入 pymarl 目录：
```bash
cd pymarl
```

建议用 tmux（断线不怕）：
```bash
tmux new -s qmix
```

在 tmux 里开两个窗格/两个窗口分别跑：
```bash
python run_qmix_batch.py --shard 0/2
```
```bash
python run_qmix_batch.py --shard 1/2
```

查看日志/进度：
- sacred 会写到：`pymarl/results/sacred/<run_id>/`
- 转换后的对比日志会写到：`pymarl/results/edge_qmix/cfg_*_seed_*_search/`

---

### 7) 跑完怎么把结果拉回本地
最简单：用 scp 把 `pymarl/results/edge_qmix` 和 `pymarl/results/sacred` 拉回。

Windows PowerShell（在本地执行）：
```powershell
scp -r 用户名@公网IP:/path/to/MARL边缘调度实验/pymarl/results/edge_qmix  C:\Users\23732\Desktop\MARL边缘调度实验\pymarl\results\
```

（sacred 原始数据也要的话）
```powershell
scp -r 用户名@公网IP:/path/to/MARL边缘调度实验/pymarl/results/sacred  C:\Users\23732\Desktop\MARL边缘调度实验\pymarl\results\
```

---

### 8) 常见坑（提前避坑）
- **断线**：不用 tmux/Screen 就很容易断线停跑（强烈建议 tmux）。
- **磁盘不够**：sacred 写盘很多，建议磁盘 >= 100GB。
- **多开进程越多越快吗？** 不一定。你现在两路 shard 已经能吃到 GPU；再加第三路可能互相抢资源变慢。


