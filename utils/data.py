import os
import json
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        ext = os.path.splitext(path)[1]
        if ext == ".yaml":
            config = yaml.load(f, yaml.FullLoader)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError("不支持的配置文件格式：" + ext)
    return config
