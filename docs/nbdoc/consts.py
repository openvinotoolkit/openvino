from pathlib import Path

notebooks_path = "notebooks"
repo_directory = "notebooks"
repo_owner = "openvinotoolkit"
repo_name = "openvino_notebooks"
repo_branch = "tree/main"
artifacts_link = "http://repository.toolbox.iotg.sclab.intel.com/projects/ov-notebook/0.1.0-latest/20240209220807/dist/rst_files/"
blacklisted_extensions = ['.xml', '.bin']
notebooks_repo = "https://github.com/openvinotoolkit/openvino_notebooks/blob/main/"
notebooks_binder = "https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath="
notebooks_colab = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/"
file_with_binder_notebooks = Path('../../docs/notebooks/notebooks_with_binder_buttons.txt').resolve(strict=True)
file_with_colab_notebooks = Path('../../docs/notebooks/notebooks_with_colab_buttons.txt').resolve(strict=True)
openvino_notebooks_ipynb_list = Path('../../docs/notebooks/all_notebooks_paths.txt').resolve(strict=True)
binder_image_source = "https://mybinder.org/badge_logo.svg"
colab_image_source = "https://colab.research.google.com/assets/colab-badge.svg"
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
