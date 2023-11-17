from pathlib import Path

notebooks_path = "notebooks"
repo_directory = "notebooks"
repo_owner = "openvinotoolkit"
repo_name = "openvino_notebooks"
repo_branch = "tree/main"
artifacts_link = "http://repository.toolbox.iotg.sclab.intel.com/projects/ov-notebook/0.1.0-latest/20231114220808/dist/rst_files/"
blacklisted_extensions = ['.xml', '.bin']
notebooks_repo = "https://github.com/openvinotoolkit/openvino_notebooks/blob/main/"
notebooks_binder = "https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath="
notebooks_colab = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/"
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

   <a href="{{ link_colab }}" target="_blank"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Google Colab"></a>

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

   <a href="{{ link_colab }}" target="_blank"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Google Colab"></a>

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
