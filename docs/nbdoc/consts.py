notebooks_docs = "notebooks.rst"

notebooks_path = "notebooks"

repo_directory = "notebooks"

repo_owner = "openvinotoolkit"

repo_name = "openvino_notebooks"

artifacts_link = "http://repository.toolbox.iotg.sclab.intel.com/projects/ov-notebook/0.1.0-latest/20230302220806/dist/rst_files/"

blacklisted_extensions = ['.xml', '.bin']

section_names = ["Getting Started", "Convert & Optimize",
                 "Model Demos", "Model Training", "Live Demos"]

# Templates

binder_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub.
See the |installation_link| for instructions to run this tutorial locally on Windows, Linux or macOS.
To run without installing anything, click the launch binder button.

|binder_link| |github_link|

.. |installation_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}#-installation-guide" target="_blank">installation guide</a>

.. |binder_link| raw:: html

   <a href="https://mybinder.org/v2/gh/{{ owner }}/{{ repo }}/HEAD?filepath={{ folder }}%2F{{ notebook }}%2F{{ notebook }}.ipynb" target="_blank"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>

.. |github_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""
no_binder_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub.
See the |installation_link| for instructions to run this tutorial locally on Windows, Linux or macOS.

|github_link|

.. |installation_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}#-installation-guide" target="_blank">installation guide</a>

.. |github_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""

rst_template = """
OpenVINO notebooks documentation
================================

{% for section in sections %}
{{section.name}}
--------------------------------

.. toctree::
   :maxdepth: 1

{% for notebook in section.notebooks %}   {{notebook.path}}\n{% endfor %}
{% endfor %}

"""
