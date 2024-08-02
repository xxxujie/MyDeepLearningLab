import argparse

from common.utils import configs
import settings


def main():
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    args = parser.parse_args()
    if args.config_dir != "":
        settings.USER_CONFIG_DIRS.insert(0, args.config_dir)

    # 测试代码
    print(configs.train_config.LEARNING_RATE)


if __name__ == "__main__":
    main()
