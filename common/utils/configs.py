import os
import yaml
import json
import settings
from .loggers import get_logger


logger = get_logger()


def load_config(config_path: str, encoding="utf-8") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding=encoding) as f:
        ext = os.path.splitext(config_path)[1]
        if ext == ".yaml":
            config = yaml.load(f, yaml.SafeLoader)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError("不支持的配置文件格式：" + ext)
        return config


def find_config_path(file_name: str):
    """按照 settings.CONFIG_DIRS 列表中的目录顺序查找名为 file_name（需要带扩展名）的配置文件"""

    for config_dir in settings.CONFIG_DIRS:
        # print(f"Finding {file_name} in {config_dir}")
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path

    raise ValueError(f"找不到配置文件 {file_name}，请检查 CONFIG_DIRS 或者命令行参数")


class _Config:
    """所有配置类的基类，实际上是一个字典的包装类，具体见 _SampleConfig"""

    def __init__(self, config_path: str):
        self._config = load_config(config_path)

    def _get(self, *args):
        item = ""
        value = self._config
        for key in args:
            item += key
            if type(value) is not dict:
                raise ValueError(f"尝试从配置项 {item} 访问不存在的键 {key}")
            value = value.get(key)
            if value is None:
                logger.warning(f"配置项 {item} 为空")
                break
            item += "."

        return value


class _TrainConfig(_Config):
    @property
    def LEARNING_RATE(self):
        return self._get("learning_rate")

    @property
    def WEIGHT_DECAY(self):
        return self._get("weight_decay")

    @property
    def OPTIMIZER(self):
        return self._get("optimizer")


train_config = _TrainConfig(find_config_path("train_conf.yaml"))
