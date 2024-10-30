from pathlib import Path
import base64

notebooks_path = "notebooks"
repo_directory = "notebooks"
repo_owner = "openvinotoolkit"
repo_name = "openvino_notebooks"
repo_branch = "tree/main"
artifacts_link = "http://repository.toolbox.iotg.sclab.intel.com/projects/ov-notebook/0.1.0-latest/20241022220806/dist/rst_files/"
blacklisted_extensions = ['.xml', '.bin']
notebooks_repo = "https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/"
notebooks_binder = "https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath="
notebooks_colab = "https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/latest/"
file_with_binder_notebooks = Path('../../docs/notebooks/notebooks_with_binder_buttons.txt').resolve(strict=True)
file_with_colab_notebooks = Path('../../docs/notebooks/notebooks_with_colab_buttons.txt').resolve(strict=True)
openvino_notebooks_ipynb_list = Path('../../docs/notebooks/all_notebooks_paths.txt').resolve(strict=True)
binder_image_source = Path('../../docs/articles_en/assets/images/launch_in_binder.svg').resolve(strict=True)
binder_image_source_data = open(binder_image_source, 'rb').read()
binder_image_source_data_base64 = base64.b64encode(binder_image_source_data)
binder_image_base64 = binder_image_source_data_base64.decode()
colab_image_source = Path('../../docs/articles_en/assets/images/open_in_colab.svg').resolve(strict=True)
colab_image_source_data = open(colab_image_source, 'rb').read()
colab_image_source_data_base64 = base64.b64encode(colab_image_source_data)
colab_image_base64 = colab_image_source_data_base64.decode()
github_image_source = Path('../../docs/articles_en/assets/images/view_on_github.svg').resolve(strict=True)
github_image_source_data = open(github_image_source, 'rb').read()
github_image_source_data_base64 = base64.b64encode(github_image_source_data)
github_image_base64 = github_image_source_data_base64.decode()

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
