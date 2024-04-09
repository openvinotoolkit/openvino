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
                        type=str, 
                        required=False,
                        help='.json file with the errors to look for in logs',
                        default=Path(__file__).resolve().parent.joinpath('errors_to_look_for.json'))
    return parser.parse_args()
