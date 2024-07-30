import os
import yaml
import json


class ConfigHandler:
    def __init__(self, config_dir="../configs"):
        self._config_dir = config_dir

    def _load_config(self):
        """加载配置文件"""
        for config_fpath in os.listdir(self._config_dir):
            with open(config_fpath, "r", encoding="utf-8") as f:
                ext = os.path.splitext(config_fpath)[1]
                if ext == ".yaml":
                    config = yaml.load(f, yaml.FullLoader)
                elif ext == ".json":
                    config = json.load(f)
                else:
                    raise ValueError("不支持的配置文件格式：" + ext)
                yield os.path.splitext(config_fpath)[0], dict(config)

    def reload(self):
        loader = self._load_config()
        for type, config in loader:
            if type == "projcet_conf":
                pass


# class ProjcetConfig:
