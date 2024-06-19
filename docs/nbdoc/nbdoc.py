import argparse
import shutil
from pathlib import Path
from utils import (
    create_content,
    add_content_below,
    verify_notebook_name,
)
from consts import (
    artifacts_link,
    binder_colab_template,
    blacklisted_extensions,
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
    binder_image_source,
    colab_image_source,
    github_image_source,
)

from notebook import Notebook
from section import Section
from glob import glob
from lxml import html
from jinja2 import Template
from urllib.request import urlretrieve
import requests
import os
import re
import sys

matching_notebooks_paths = []


def fetch_binder_list(binder_list_file) -> list:
    """Function that fetches list of notebooks with binder buttons

    :param file_format: Format of file containing list of notebooks with button. Defaults to 'txt'
    :type file_format: str
    :return: List of notebooks containing binder buttons
    :rtype: list
    """
    if binder_list_file:
        with open(binder_list_file) as file:
            list_of_buttons = file.read().splitlines()
    return list_of_buttons

def fetch_colab_list(colab_list_file) -> list:
    """Function that fetches list of notebooks with colab buttons

    :param file_format: Format of file containing list of notebooks with button. Defaults to 'lst'
    :type file_format: str
    :return: List of notebooks containing colab buttons
    :rtype: list
    """
    if colab_list_file:
        with open(colab_list_file) as file:
            list_of_cbuttons = file.read().splitlines()
    return list_of_cbuttons


def add_glob_directive():
    """This function modifies toctrees of the five node articles in tutorials 
       section. It adds the notebooks found in docs/notebooks directory to the menu.
    """
    tutorials_path = Path('../../docs/articles_en/learn_openvino/tutorials').resolve(strict=True)
    tutorials_files = [x for x in os.listdir(tutorials_path) if re.match("notebooks_section_[0-9]{1}\.", x)]
    for tutorials_file in tutorials_files:
        file_name = os.path.join(tutorials_path, tutorials_file)
        with open(file_name, 'r+', encoding='cp437') as section_file:
            section_number = ''.join(c for c in str(tutorials_file) if c.isdigit())
            read_file = section_file.read()
            if ':glob:' not in read_file:
                add_glob = read_file\
                    .replace(":hidden:\n", ":hidden:\n   :glob:\n   :reversed:\n\n   notebooks/" + section_number +"*\n")
                section_file.seek(0)
                section_file.write(add_glob)
                section_file.truncate()

class NbTravisDownloader:
    @staticmethod
    def download_from_jenkins(path: str = notebooks_path, artifact_link: str = artifacts_link):
        """Function for downloading files from jenkins artifacts

        :param path: path where notebooks files will be placed, defaults to notebooks_path
        :type path: str, optional
        :param artifact_link: link of notebooks artifacts rst files, defaults to artifacts_link
        :type artifact_link: str, optional
        """
        def is_directory(path: str) -> bool:
            """Helper fuction for checking whether path leads to subdirectory

            :param path: Path to traversed file or directory
            :type path: str
            :return: Returns True if path leads to directory, otherwise False
            :rtype: bool
            """
            return path[-1] == '/' and path != '../'

        def traverse(path: Path, link: str, blacklisted_extensions: list = blacklisted_extensions):
            """Traverse recursively to download all directories with their subfolders, within given link.

            :param path: Path to directory that file will be saved to.
            :type path: Path
            :param link: Link to hosted resources
            :type link: str
            """
            path.mkdir(exist_ok=True)
            page = requests.get(link, verify=False).content
            tree = html.fromstring(page)
            # retrieve all links on page returning their content
            tree = tree.xpath('//a[@*]/@href')
            files = map(str, tree)
            for file in files:
                if is_directory(file):
                    traverse(path.joinpath(file), link + file)
                elif len(Path(file).suffix) > 0 and Path(file).suffix not in blacklisted_extensions:
                    urlretrieve(link + file, path.joinpath(file))

        traverse(Path(path), artifact_link)


class NbProcessor:
    def __init__(self, nb_path: str = notebooks_path):
        self.nb_path = nb_path

        with open(openvino_notebooks_ipynb_list, 'r+', encoding='cp437') as ipynb_file:
            openvino_notebooks_paths_list = ipynb_file.readlines()

        for notebook_name in [
            nb for nb in os.listdir(self.nb_path) if
            verify_notebook_name(nb)
        ]:

            if not os.path.exists(openvino_notebooks_ipynb_list):
                raise FileNotFoundError("all_notebooks_paths.txt is not found")
            else:
                ipynb_list = [x for x in openvino_notebooks_paths_list if re.match("notebooks/[0-9]{3}.*\.ipynb$", x)]
                notebook_with_ext = notebook_name[:-16] + ".ipynb"
                matching_notebooks = [re.sub('[\n]', '', match) for match in ipynb_list if notebook_with_ext in match]

            if matching_notebooks is not None:
                for n in matching_notebooks:
                    matching_notebooks_paths.append(n)

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

        for notebook_file, nb_path in zip([
            nb for nb in os.listdir(self.nb_path) if verify_notebook_name(nb)
        ], matching_notebooks_paths):

            notebook_item = '-'.join(notebook_file.split('-')[:-2])
            local_install = ".. |installation_link| raw:: html\n\n   <a href='https://github.com/" + \
                repo_owner + "/" + repo_name + "#-installation-guide' target='_blank' title='Install " + \
                notebook_item + " locally'>local installation</a> \n\n"
            binder_badge = ".. raw:: html\n\n   <a href='" + notebooks_binder + \
                nb_path + "' target='_blank' title='Run " + notebook_item + \
                " on Binder'><img src='" + binder_image_source + "' class='notebook_badge' alt='Binder'></a>\n\n"
            colab_badge = ".. raw:: html\n\n   <a href='" + notebooks_colab + \
                nb_path + "' target='_blank' title='Run " + notebook_item + \
                " on Google Colab'><img src='" + colab_image_source + "' class='notebook_badge'alt='Google Colab'></a>\n\n"
            github_badge = ".. raw:: html\n\n   <a href='" + notebooks_repo + \
                nb_path + "' target='_blank' title='View " + notebook_item + \
                " on Github'><img src='" + github_image_source + "' class='notebook_badge' alt='Github'></a><br><br>\n\n"

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
    parser.add_argument('-d', '--download', action='store_true')
    args = parser.parse_args()
    sourcedir = args.sourcedir
    outdir = args.outdir

    add_glob_directive()

    if args.download:
        outdir.mkdir(parents=True, exist_ok=True)
        # Step 2. Run default pipeline for downloading
        NbTravisDownloader.download_from_jenkins(outdir)
    else:
        shutil.copytree(sourcedir, outdir)
    # Step 3. Run processing on downloaded file
    nbp = NbProcessor(outdir)
    nbp.add_binder(buttons_list, cbuttons_list)


if __name__ == '__main__':
    main()

