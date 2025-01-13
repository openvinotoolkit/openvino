# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil

def remove_xml_dir(path):
    """
    Remove doxygen xml folder
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_dir')
    args = parser.parse_args()
    remove_xml_dir(args.xml_dir)


if __name__ == '__main__':
    main()
