import argparse
import settings
import torch

from common.utils import configs, loggers
from src.models import Transformer


def main():
    src = torch.LongTensor([[1, 3, 4, 5, 6, 7, 8, 2, 0, 0]])
    tgt = torch.LongTensor([[1, 3, 4, 5, 6, 7, 8, 2, 0, 0]])
    transformer = Transformer(20, 20, 10, 10)
    output = transformer(src, tgt)
    return output


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    args = parser.parse_args()
    if args.config_dir != "":
        settings.USER_CONFIG_DIRS.insert(0, args.config_dir)
    logger = loggers.get_logger()

    output = main()
    logger.info(output.shape)
