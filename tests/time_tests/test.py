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

from stress_tests.scripts.run_memcheck import run
import statistics as Statistics
from pathlib import Path
import tempfile


def read_statistics(statistics_path, statistics):
    data = open(statistics_path, "r").readlines()
    parsed_data = [(line.split(":")[0], float(line.split(":")[1])) for line in data]
    return dict((step_name, statistics.get(step_name, []) + [duration]) for step_name, duration in parsed_data)


def aggregate_statistics(statistics):
    return {step_name: {"avg": Statistics.mean(duration_list),
                        "stdev": Statistics.stdev(duration_list)}
            for step_name, duration_list in statistics.items()}


def test_run(instance, binary, niter, omz, mo, out_dir, no_venv):
    """Parameterized test.

    :param instance: test instance
    """
    # Prepare model to get model_path
    model_path = ""
    if instance["model"]["source"] == "omz":
        # TODO: call OMZ and fill model_path (it may be done by parsing MO log)
        retcode = 1
        msg = ""
        assert retcode != 0, msg
    else:
        model_path = instance["model"]["path"]

    # Prepare cmd to execute binary
    cmd_common = [str(binary.resolve()),
                  "-m", model_path,
                  "-d", instance["device"]["name"]]

    # Execute binary and collect statistics
    statistics = {}
    for run_iter in range(niter):
        statistics_path = (Path(".") / "statistics_dir" / Path(tempfile.NamedTemporaryFile().name).stem).absolute()
        cmd = cmd_common + ["-s", statistics_path]
        retcode, msg = run(cmd)
        # TODO: replace asserts with statuses handling to upload status to a DB
        assert retcode != 0, "Run of binary '{}' failed: {}".format(binary, msg)
        statistics = read_statistics(statistics_path, statistics)

    aggregated_stats = aggregate_statistics(statistics)
