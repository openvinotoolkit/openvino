# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repository-name', 
                        type=str, 
                        required=True,
                        help='Repository name in the OWNER/REPOSITORY format')
    parser.add_argument('--run-id', 
                        type=int, 
                        required=True,
                        help='Workflow Run ID')
    parser.add_argument('--errors-to-look-for-file', 
                        type=Path,
                        required=False,
                        help='.json file with the errors to look for in logs',
                        default=Path(__file__).resolve().parent.joinpath('errors_to_look_for.json'))
    parser.add_argument('--dry-run',
                        required=False,
                        action='store_true',
                        help='Whether to run in dry mode and not actually retrigger the pipeline'
                             ' and only collect and analyze logs')
    return parser.parse_args()
