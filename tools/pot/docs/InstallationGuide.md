# Post-Training Optimization Tool Installation Guide {#pot_InstallationGuide}

## Install POT from PyPI
POT is distributed as a part of OpenVINO&trade; Development Tools package. For installation instruction please refer to this [document](@ref openvino_docs_install_guides_install_dev_tools).

## Install POT from GitHub
The latest version of the Post-training Optimization Tool is available on [GutHub](https://github.com/openvinotoolkit/openvino/tree/master/tools/pot) and can be installed from source. As prerequisites, you should install [OpenVINO&trade; Runtime](@ref openvino_docs_install_guides_install_runtime) and other dependencies such as [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) and [Accuracy Checker](@ref omz_tools_accuracy_checker).

To install POT from source:
- Clone OpenVINO repository
   ```sh
   git clone --recusive https://github.com/openvinotoolkit/openvino.git
   ```
- Navigate to `openvino/tools/pot/` folder
- Install POT package:
   ```sh
   python3 setup.py install
   ```

After installation POT is available as a Python* library under `openvino.tools.pot.*` and in the command line by the `pot` alias. To verify it, run `pot -h`. 
