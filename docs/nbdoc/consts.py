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


# Templates

binder_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub.
See the |installation_link| for instructions to run this tutorial locally on Windows, Linux or macOS.
To run without installing anything, click the "launch binder" button.

|binder_link| |github_link|

.. |installation_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}#-installation-guide" target="_blank">installation guide</a>

.. |binder_link| raw:: html

   <a href="{{ link_binder }}" target="_blank"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>

.. |github_link| raw:: html

   <a href="{{ link_git }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""
colab_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub.
See the |installation_link| for instructions to run this tutorial locally on Windows, Linux or macOS.
To run without installing anything, click the "Open in Colab" button.

|colab_link| |github_link|

.. |installation_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}#-installation-guide" target="_blank">installation guide</a>

.. |colab_link| raw:: html

   <a href="{{ link_colab }}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab"></a>

.. |github_link| raw:: html

   <a href="{{ link_git }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""
binder_colab_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub.
See the |installation_link| for instructions to run this tutorial locally on Windows, Linux or macOS.
To run without installing anything, click the "launch binder" or "Open in Colab" button.

|binder_link| |colab_link| |github_link|

.. |installation_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}#-installation-guide" target="_blank">installation guide</a>

.. |binder_link| raw:: html

   <a href="{{ link_binder }}" target="_blank"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>

.. |colab_link| raw:: html

   <a href="{{ link_colab }}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab"></a>

.. |github_link| raw:: html

   <a href="{{ link_git }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""
no_binder_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub.
See the |installation_link| for instructions to run this tutorial locally on Windows, Linux or macOS.

|github_link|

.. |installation_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}#-installation-guide" target="_blank">installation guide</a>

.. |github_link| raw:: html

   <a href="{{ link_git }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""
