from pathlib import Path

notebooks_path = "notebooks"
repo_directory = "notebooks"
repo_owner = "openvinotoolkit"
repo_name = "openvino_notebooks"
repo_branch = "tree/main"
artifacts_link = "http://repository.toolbox.iotg.sclab.intel.com/projects/ov-notebook/0.1.0-latest/20231206220809/dist/rst_files/"
blacklisted_extensions = ['.xml', '.bin']
notebooks_repo = "https://github.com/openvinotoolkit/openvino_notebooks/blob/main/"
notebooks_binder = "https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath="
notebooks_colab = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/"
file_with_binder_notebooks = Path('../../docs/notebooks/notebooks_with_binder_buttons.txt').resolve(strict=True)
file_with_colab_notebooks = Path('../../docs/notebooks/notebooks_with_colab_buttons.txt').resolve(strict=True)
openvino_notebooks_ipynb_list = Path('../../docs/notebooks/all_notebooks_paths.txt').resolve(strict=True)
binder_image_source = "https://mybinder.org/badge_logo.svg"
colab_image_source = "https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"
github_image_source = "https://badgen.net/badge/icon/github?icon=github&label"

# Templates

binder_colab_template = """
This Jupyter notebook can be launched on-line, opening an interactive environment in a browser window.
You can also make a |installation_link|. Choose one of the following options:

{{ link_binder }}{{ link_colab }}{{ link_git }}

{{ installation_link }}

\n
"""
no_binder_template = """
This Jupyter notebook can be launched after a |installation_link| only.

{{ link_git }}

{{ installation_link }}

\n
"""
