import os

def pytest_addoption(parser):
    parser.addoption("--do_not_cleanup", action="store_true", default=False, help="Do not cleanup temporary files after test completion")
    parser.addoption("--gpu_suffix", action="store", default="0", help="Suffix to append to GPU device name (e.g., '1' for GPU.1)")
    parser.addoption("--samples", action="store", default=15, type=int, help="Number of samples for evaluation")

def pytest_configure(config):
    """Set environment variables based on pytest options"""
    # Set cleanup option
    if config.getoption("--do_not_cleanup"):
        os.environ["PYTEST_DO_NOT_CLEANUP"] = "true"
    else:
        os.environ["PYTEST_DO_NOT_CLEANUP"] = "false"
    
    # Set other options
    os.environ["PYTEST_GPU_SUFFIX"] = str(config.getoption("--gpu_suffix"))
    os.environ["PYTEST_SAMPLES"] = str(config.getoption("--samples"))

