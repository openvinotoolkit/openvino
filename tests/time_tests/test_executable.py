"""Main entry-point to run E2E OSS tests.
TODO: update
Default run:
$ pytest test.py

Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config

[*] For more information see conftest.py
"""

from pathlib import Path
import logging

from run_executable import run_executable


def test_executable(instance, executable, niter):
    """Parameterized test.

    :param instance: test instance
    """
    # Prepare model to get model_path
    model_path = ""
    if instance["model"].get("source") == "omz":
        # TODO: call OMZ and fill model_path (it may be done by parsing MO log)
        retcode = 1
        msg = "OMZ models aren't supported yet"
        assert retcode == 0, msg
    else:
        model_path = instance["model"].get("path")

    assert model_path, "Model path is empty"

    exe_args = {
        "executable": Path(executable),
        "model": Path(model_path),
        "device": instance["device"]["name"],
        "niter": niter
    }
    retcode, aggr_stats = run_executable(exe_args, log=logging)
    assert retcode == 0, "Run of executable failed"     # TODO: update
