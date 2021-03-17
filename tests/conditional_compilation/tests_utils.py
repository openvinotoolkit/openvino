import json

from pathlib import Path


def open_cc_json(path: Path = Path(__file__).parent / "cc_tests.json"):
    with open(path, 'r') as json_file:
        cc_tests_ids = json.load(json_file)
    return cc_tests_ids


def save_cc_json(path: Path = Path(__file__).parent / "cc_tests.json", data: list = None):
    if data is None:
        data = []
    with open(path, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def save_tests_structure(cc_tests_ids: list, workspace: Path):
    for test_dict in cc_tests_ids:
        used_csv = []
        for cc_csv in workspace.glob("**/*.csv"):
            if test_dict["test_id"] in str(cc_csv):
                used_csv.append(str(cc_csv))
        test_dict["csv_path"] = used_csv
    save_cc_json(data=cc_tests_ids)
