import argparse
import os

from helpers.configs import config_handler


def main(args):
    config_handler.reload(args.config_dir)
    print(config_handler.project_conf["log_path"])


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./configs")

    args = parser.parse_args()

    main(args)
