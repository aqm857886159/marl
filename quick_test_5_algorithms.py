"""
快速测试脚本 - 验证 5 种算法是否都能正常运行
只运行少量迭代来验证代码逻辑
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

import MRRL
import ray
import time

MRRL.USE_MOCK_DATA = False

print("=" * 60)
print("快速测试：验证 MAPPO / IPPO / Greedy 配置")
print("=" * 60)

if not ray.is_initialized():
    ray.init(num_cpus=4, num_gpus=1)

algorithms_to_test = ["MAPPO", "IPPO"]
test_seed = 0
smoke_iterations = 20
smoke_max_steps = 50_000
smoke_eval_interval = 10_000

print("\n【测试 MARL 算法】")
for alg in algorithms_to_test:
    try:
        print(f"\n✓ 测试 {alg}...")
        log = MRRL.run_experiment(
            alg,
            test_seed,
            num_iterations=smoke_iterations,
            max_timesteps=smoke_max_steps,
            eval_interval_steps=smoke_eval_interval
        )
        print(f"  成功！收集了 {len(log)} 条记录")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()
        time.sleep(2)
        ray.init(num_cpus=4, num_gpus=1)

ray.shutdown()

# 测试 Greedy 基线 (不需要 Ray)
print("\n【测试 Greedy 基线】")
try:
    print("✓ 测试 Greedy...")
    result = MRRL.run_greedy_baseline(test_seed, num_episodes=10)
    print(f"  成功！延迟: {result['avg_latency_ms_mean']:.2f}ms")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 所有算法测试完成！")
print("如果上面没有错误，说明 MAPPO / IPPO / Greedy 配置正确")
print("现在可以运行完整版本: python MRRL.PY")
print("=" * 60)

