import os
import argparse
import shutil
from pathlib import Path
from utils import (
    create_content,
    add_content_below,
    verify_notebook_name,
)
from consts import (
    binder_colab_template,
    notebooks_path,
    no_binder_template,
    repo_directory,
    repo_name,
    openvino_notebooks_ipynb_list,
    file_with_binder_notebooks,
    file_with_colab_notebooks,
    repo_owner,
    notebooks_repo,
    notebooks_binder,
    notebooks_colab,
    binder_image_base64,
    colab_image_base64,
    github_image_base64,
)


def fetch_binder_list(binder_list_file) -> list:
    """Function that fetches list of notebooks with binder buttons

    :return: List of notebooks containing binder buttons
    :rtype: list
    """
    if binder_list_file:
        with open(binder_list_file) as file:
            list_of_buttons = file.read().splitlines()
    return list_of_buttons


def fetch_colab_list(colab_list_file) -> list:
    """Function that fetches list of notebooks with colab buttons

    :return: List of notebooks containing colab buttons
    :rtype: list
    """
    if colab_list_file:
        with open(colab_list_file) as file:
            list_of_cbuttons = file.read().splitlines()
    return list_of_cbuttons


def add_glob_directive(tutorials_file):
    """This function modifies toctrees of the five node articles in tutorials
       section. It adds the notebooks found in docs/notebooks directory to the menu.
    """
    with open(tutorials_file, 'r+', encoding='cp437') as mainfile:
        readfile = mainfile.read()
        if ':glob:' not in readfile:
            add_glob = readfile\
                .replace(":hidden:\n", ":hidden:\n   :glob:\n")\
                .replace("   interactive-tutorials-python/notebooks-installation\n", "   interactive-tutorials-python/notebooks-installation\n   ../../notebooks/*\n")
            mainfile.seek(0)
            mainfile.write(add_glob)
            mainfile.truncate()


class NbProcessor:
    def __init__(self, nb_path: str = notebooks_path):
        self.nb_path = nb_path

    def add_binder(self, buttons_list: list, cbuttons_list: list, template_with_colab_and_binder: str = binder_colab_template, template_without_binder: str = no_binder_template):
        """A function working as an example of how to add Binder or Google Colab buttons to existing RST files.

        :param buttons_list: A list of notebooks that work on Binder.
        :type buttons_list: list
        :param cbuttons_list: A list of notebooks that work on Google Colab.
        :type cbuttons_list: list
        :param template_with_colab_and_binder: A template with buttons added to an RST file if Binder and/or Google Colab are available. Defaults to template_with_colab_and_binder.
        :type template_with_colab_and_binder: str
        :param template_without_binder: A template with buttons added to an RST file if neither Binder nor Google Colab are available. Defaults to no_binder_template.
        :type template_without_binder: str
        :raises FileNotFoundError: In case of a failure in adding the content, an error will appear.

        """

        if not os.path.exists(openvino_notebooks_ipynb_list):
            raise FileNotFoundError("all_notebooks_paths.txt is not found")
        else:
            with open(openvino_notebooks_ipynb_list, 'r+', encoding='cp437') as ipynb_file:
                openvino_notebooks_paths_list = ipynb_file.read()

        for notebook_file in [nb for nb in os.listdir(self.nb_path) if verify_notebook_name(nb)]:

            notebook_ipynb_ext = notebook_file[:-16] + ".ipynb"
            nb_path_match = [line for line in openvino_notebooks_paths_list.split('\n') if notebook_ipynb_ext in line]
            nb_repo_path = ''.join(nb_path_match)
            notebook_item = '-'.join(notebook_file.split('-')[:-2])

            local_install = ".. |installation_link| raw:: html\n\n   <a href='https://github.com/" + \
                repo_owner + "/" + repo_name + "#-installation-guide' target='_blank' title='Install " + \
                notebook_item + " locally'>local installation</a> \n\n"
            binder_badge = ".. raw:: html\n\n   <a href='" + notebooks_binder + \
                nb_repo_path + "' target='_blank' title='Launch " + notebook_item + \
                " in Binder'><img src='data:image/svg+xml;base64," + binder_image_base64 + "' class='notebook-badge' alt='Binder'></a>\n\n"
            colab_badge = ".. raw:: html\n\n   <a href='" + notebooks_colab + \
                nb_repo_path + "' target='_blank' title='Open " + notebook_item + \
                " in Google Colab'><img src='data:image/svg+xml;base64," + colab_image_base64 + "' class='notebook-badge'alt='Google Colab'></a>\n\n"
            github_badge = ".. raw:: html\n\n   <a href='" + notebooks_repo + \
                nb_repo_path + "' target='_blank' title='View " + notebook_item + \
                " on Github'><img src='data:image/svg+xml;base64," + github_image_base64 + "' class='notebook-badge' alt='Github'></a><br><br>\n\n"

            binder_data = {
                "owner": repo_owner,
                "repo": repo_name,
                "folder": repo_directory,
                "link_git": github_badge,
                "link_binder": binder_badge if notebook_item in buttons_list else "",
                "link_colab": colab_badge if notebook_item in cbuttons_list else "",
                "installation_link": local_install
            }

            if notebook_item in buttons_list or notebook_item in cbuttons_list:
                template = template_with_colab_and_binder
            else:
                template = template_without_binder

            button_text = create_content(template, binder_data, notebook_file)
            if not add_content_below(button_text, f"{self.nb_path}/{notebook_file}"):
                raise FileNotFoundError("Unable to modify file")


def main():
    buttons_list = fetch_binder_list(file_with_binder_notebooks)
    cbuttons_list = fetch_colab_list(file_with_colab_notebooks)
    parser = argparse.ArgumentParser()
    parser.add_argument('sourcedir', type=Path)
    parser.add_argument('outdir', type=Path)
    args = parser.parse_args()
    sourcedir = args.sourcedir
    outdir = args.outdir

    main_tutorials_file = Path('../../docs/articles_en/learn-openvino/interactive-tutorials-python.rst').resolve(strict=True)
    add_glob_directive(main_tutorials_file)
    shutil.copytree(sourcedir, outdir)
    # Run processing on downloaded files in notebooks directory
    nbp = NbProcessor(outdir)
    # Add Binder, Google Colab, GitHub badges to notebooks
    nbp.add_binder(buttons_list, cbuttons_list)


if __name__ == '__main__':
    main()

