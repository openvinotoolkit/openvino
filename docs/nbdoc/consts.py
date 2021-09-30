notebooks_docs = "notebooks.rst"

notebooks_path = "notebooks"

repo_owner = "openvinotoolkit"

repo_name = "openvino_notebooks"

# Templates

binder_template = """
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/{{ owner }}/{{ repo }}/HEAD?filepath={{ folder }}%2F{{ notebook }}%2F{{ notebook }}.ipynb
   :alt: Binder


"""

rst_template = """
.. toctree::
   :maxdepth: 1
   :caption: Table of Contents

{% for notebook in notebooks %}   {{notebook.name}}<{{notebook.path}}>\n{% endfor %}

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
