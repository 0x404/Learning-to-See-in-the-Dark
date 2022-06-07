"""launcher"""
from asyncio.log import logger
from utils import Parser
from trainer import Trainer
import logging

logger = logging.getLogger(__name__)


def main():
    """main entry"""
    config = Parser().config
    logger.info("launching runner ...")
    logger.info(f"configs : {config}")
    runner = Trainer(config)
    if config.setup.do_train:
        runner.train()
    if config.setup.do_predict:
        runner.predict()
    logger.info("runner finished! ᕦ(･ㅂ･)ᕤ")


if __name__ == "__main__":
    main()
