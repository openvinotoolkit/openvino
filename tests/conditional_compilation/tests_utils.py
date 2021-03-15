import json

from pathlib import Path


def open_cc_json(path=Path(__file__).parent / "cc_tests.json"):
    with open(path, 'r') as json_file:
        cc_tests_ids = json.load(json_file)
    return cc_tests_ids


def get_tests_structure(workspace: Path):
    cc_tests_ids = open_cc_json()
    for test_dict in cc_tests_ids:
        used_csv = []
        for cc_csv in workspace.glob("**/*.csv"):
            if test_dict["test_id"] in str(cc_csv):
                used_csv.append(cc_csv)
        test_dict["csv_path"] = used_csv
    return cc_tests_ids
