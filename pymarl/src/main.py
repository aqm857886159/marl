import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
import logging
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

# Windows 下 sacred 的 CAPTURE_MODE="fd" 容易在长跑后触发 WinError 1（stdout/stderr 句柄异常）
# 这里强制改为更稳的 "sys"（也可用 "no" 完全关闭捕获）。
SETTINGS['CAPTURE_MODE'] = "sys"

# 减少 GitPython 在 DEBUG 级别下的刷屏/开销（sacred 会读取 git 信息）
# 说明：我们自己的 logger 设为 DEBUG（见 utils/logging.py），会把 git.* 的 DEBUG 也打印出来。
# 这里显式把 git 相关 logger 压到 WARNING，减少输出和少量启动开销。
for _name in ("git", "git.cmd", "git.util", "git.repo", "git.remote"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        # Windows 默认编码可能是 GBK，遇到 UTF-8 YAML 会 UnicodeDecodeError
        # 用 utf-8-sig 兼容带 BOM 的 UTF-8 文件
        with open(
            os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
            "r",
            encoding="utf-8-sig",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"),
        "r",
        encoding="utf-8-sig",
    ) as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

