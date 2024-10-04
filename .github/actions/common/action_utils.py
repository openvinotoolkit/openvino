import logging
import os


def init_logger():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')


def set_github_output(name: str, value: str, github_output_var_name: str = 'GITHUB_OUTPUT'):
    """Sets output variable for a GitHub Action"""
    logger = logging.getLogger(__name__)
    # In an environment variable "GITHUB_OUTPUT" GHA stores path to a file to write outputs to
    with open(os.environ.get(github_output_var_name), 'a+') as file:
        logger.info(f"Add {name}={value} to {github_output_var_name}")
        print(f'{name}={value}', file=file)
