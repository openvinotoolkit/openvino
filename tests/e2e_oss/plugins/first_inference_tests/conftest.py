import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.e2e.env_tools import Environment


def redefine_mo_out_path(instance, pipeline):
    """
    Redefine mo_out variable for used instance.
    """
    sub_path = re.sub(r'[^\w\-_\. ]', "_", instance.test_id)  # filter all symbols not supported in a file systems
    tmpdir_sub_path = Path(TemporaryDirectory(prefix=sub_path).name).name
    mo_out = str(Path(Environment.env.get('mo_out')) / tmpdir_sub_path)
    pipeline['get_ir']['mo']['mo_out'] = mo_out

    return mo_out


def get_non_infer_config(pipeline):
    """
    Delete infer step from used instance pipeline.
    """
    ie_config = deepcopy(pipeline)
    if "infer" in ie_config:
        del ie_config["infer"]
    if "postprocess" in ie_config:
        del ie_config["postprocess"]

    return ie_config
