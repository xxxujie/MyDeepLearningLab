import argparse

from common.utils.configs import train_config
import settings


def main(args):
    if args.config_dir != "":
        settings.CONFIG_DIRS.insert(0, args.config_dir)
    print(train_config.learning_rate)


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")

    args = parser.parse_args()

    main(args)
