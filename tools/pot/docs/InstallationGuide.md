# Post-Training Optimization Tool Installation Guide {#pot_InstallationGuide}

## Prerequisites

* Python* 3.6 or higher
* [OpenVINO&trade;](https://docs.openvino.ai/latest/index.html)

The minimum and the recommended requirements to run the Post-training Optimization Tool (POT) are the same as in [OpenVINO&trade;](https://docs.openvino.ai/latest/index.html).

There are two ways how to install the POT on your system:
- Installation from PyPI repository
- Installation from Intel&reg; Distribution of OpenVINO&trade; toolkit package

## Install POT from PyPI
The simplest way to get the Post-training Optimization Tool and OpenVINO&trade; installed is to use PyPI. Follow the steps below to do that:
1. Create a separate [Python* environment](https://docs.python.org/3/tutorial/venv.html) and activate it
2. To install OpenVINO&trade;, run `pip install openvino`.
3. To install POT and other OpenVINO&trade; developer tools, run `pip install openvino-dev`.

Now the Post-training Optimization Tool is available in the command line by the `pot` alias. To verify it, run `pot -h`.
