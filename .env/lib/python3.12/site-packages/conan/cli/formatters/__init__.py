import json

from conan.api.output import cli_out_write


def default_json_formatter(data):
    myjson = json.dumps(data, indent=4)
    cli_out_write(myjson)
