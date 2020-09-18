"""Main entry-point to run timetests tests.
TODO: update
Default run:
$ pytest test_executable.py

Options[*]:
--test_conf     Path to test config
--exe           Path to binary to execute
--niter         Number of times to run executable

[*] For more information see conftest.py
"""

from pathlib import Path
import logging

from run_executable import run_executable


def test_executable(instance, executable, niter):
    """Parameterized test.

    :param instance: test instance
    :param executable: binary executable to run
    :param niter: number of times to run executable
    """
    # Prepare model to get model_path
    model_path = ""
    if instance["model"].get("source") == "omz":
        # TODO: add OMZ support
        # call OMZ and fill model_path (it may be done by parsing MO log)
        retcode = 1
        msg = "OMZ models aren't supported yet"
        assert retcode == 0, msg
    else:
        model_path = instance["model"].get("path")

    assert model_path, "Model path is empty"

    # Run executable
    exe_args = {
        "executable": Path(executable),
        "model": Path(model_path),
        "device": instance["device"]["name"],
        "niter": niter
    }
    retcode, aggr_stats = run_executable(exe_args, log=logging)
    assert retcode == 0, "Run of executable failed"
    instance["statistics"] = aggr_stats     # Save statistics to dump to DB

    # Compare with references
    comparison_status = 0
    for step_name, references in instance["references"].items():
        for metric, reference_val in references.items():
            if aggr_stats[step_name][metric] > reference_val:
                logging.error("Comparison failed for '{}' step for '{}' metric. Reference: {}. Current values: {}"
                              .format(step_name, metric, reference_val, aggr_stats[step_name][metric]))
                comparison_status = 1
            else:
                logging.info("Comparison passed for '{}' step for '{}' metric. Reference: {}. Current values: {}"
                             .format(step_name, metric, reference_val, aggr_stats[step_name][metric]))

    assert comparison_status == 0, "Comparison with references failed"
