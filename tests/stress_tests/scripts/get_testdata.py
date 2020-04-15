#!/usr/bin/env python3
""" Script to acquire model IRs for stress tests.
Usage: ./scrips/get_testdata.py
"""
import argparse
import multiprocessing
import os
import shutil
import subprocess
from inspect import getsourcefile

# Parameters
MODEL_NAMES = 'vgg16,mtcnn-r,mobilenet-ssd,ssd300'
OMZ_VERSION = 'efd238d02035f8a5417b7b1e25cd4c997d44351f'


def abs_path(relative_path):
    """Return absolute path given path relative to the current file.
    """
    return os.path.realpath(
        os.path.join(os.path.dirname(getsourcefile(lambda: 0)), relative_path))


def main():
    """Main entry point.
    """
    parser = argparse.ArgumentParser(
        description='Acquire test data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_dir', default=f'./_models', help='directory to put test data into')
    parser.add_argument('--cache_dir', default=f'./_cache', help='directory with test data cache')
    args = parser.parse_args()

    # Clone Open Model Zoo into temporary path
    omz_path = './_open_model_zoo'
    if os.path.exists(omz_path):
        shutil.rmtree(omz_path)
    subprocess.check_call(
        f'git clone https://github.com/opencv/open_model_zoo {omz_path}' \
        f' && cd {omz_path}'\
        f' && git checkout {OMZ_VERSION}', shell=True)
    # Acquire model IRs
    mo_tool = abs_path('../../../model-optimizer/mo.py')
    subprocess.check_call(
        f'{omz_path}/tools/downloader/downloader.py --name "{MODEL_NAMES}"' \
        f' --output_dir {args.output_dir}/{OMZ_VERSION}/models' \
        f' --cache_dir {args.cache_dir}', shell=True)
    subprocess.check_call(
        f'{omz_path}/tools/downloader/converter.py --name "{MODEL_NAMES}"' \
        f' --output_dir {args.output_dir}/{OMZ_VERSION}/IRs' \
        f' --download_dir {args.output_dir}/{OMZ_VERSION}/models' \
        f' --mo {mo_tool} --jobs {multiprocessing.cpu_count()}', shell=True)


if __name__ == "__main__":
    main()
