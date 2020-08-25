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

# from stress_tests.scripts.run_memcheck import run

def test_run(instance, omz, mo, out_dir, no_venv):
    """Parameterized test.

    :param instance: test instance TODO: update
    """
    statuses = []
    model_path = ""
    if instance["model"]["source"] == "omz":
        # TODO: call OMZ and fill model_path (it may be done by parsing MO log)
        retcode = 1
        msg = ""
        statuses.append({"omz": (retcode, msg)})
    else:
        model_path = instance["model"]["path"]

    binary = ""
    cmd = [binary,
           "-m", model_path,
           "-d", instance["device"]["name"]]
    cmd = [str(item) for item in cmd]
    retcode, msg = run(cmd)
    statuses.append({"test": (retcode, msg)})

    instance["model"]["name"]
    instance["model"]["precision"]
    instance["model"]["source"]
    instance["device"]