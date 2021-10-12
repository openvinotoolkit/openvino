notebooks_docs = "notebooks.rst"

notebooks_path = "notebooks"

repo_owner = "openvinotoolkit"

repo_name = "openvino_notebooks"

# Templates

binder_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub. 
See the installation guide for instructions to run this tutorial locally on Windows, Linux or macOS. 
To run without installing anything, click the launch binder button.

|binder_link| |github_link|

.. |binder_link| raw:: html 

   <a href="https://mybinder.org/v2/gh/{{ owner }}/{{ repo }}/HEAD?filepath={{ folder }}%2F{{ notebook }}%2F{{ notebook }}.ipynb" target="_blank"><img src="https://mybinder.org/badge_logo.svg" alt="Binder"></a>

.. |github_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""

no_binder_template = """
This tutorial is also available as a Jupyter notebook that can be cloned directly from GitHub. 
See the installation guide for instructions to run this tutorial locally on Windows, Linux or macOS.

|github_link|

.. |github_link| raw:: html

   <a href="https://github.com/{{ owner }}/{{ repo }}" target="_blank"><img src="https://badgen.net/badge/icon/github?icon=github&label" alt="Github"></a>

\n
"""

rst_template = """
.. toctree::
   :maxdepth: 1
   :caption: Table of Contents

{% for notebook in notebooks %}   {{notebook.path}}\n{% endfor %}

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
