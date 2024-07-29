import argparse
import os
from utils import data


def main(args):
    # train_config = load_config(args.train_config_path)
    # model_config = load_config(args.model_config_path)

    # learning_rate = train_config["learning_rate"]
    # print(learning_rate)
    # print(model_config)

    yaml_config = data.load_config("./configs/project_conf.yaml")
    print(os.path.join(yaml_config["project_dir"], yaml_config["log_path"]))


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config_path", type=str, default="./config/train_config.json"
    )
    parser.add_argument(
        "--model_config_path", type=str, default="./config/model_config.json"
    )

    args = parser.parse_args()
    main(args)
