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
    binder_template,
    colab_template,
    binder_colab_template,
    blacklisted_extensions,
    notebooks_path,
    no_binder_template,
    repo_directory,
    repo_name,
    openvino_notebooks_json,
    repo_owner,
    notebooks_repo,
    notebooks_binder,
    notebooks_colab,

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
import json

matching_notebooks_paths = []
openvino_notebooks_paths_list = json.loads(open(openvino_notebooks_json, 'r').read())

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

        for notebook_name in [
            nb for nb in os.listdir(self.nb_path) if
            verify_notebook_name(nb)
        ]:

            if "tree" in openvino_notebooks_paths_list:
                notebooks_paths_listing = [p.get('path') for p in openvino_notebooks_paths_list['tree'] if
                                           p.get('path')]
                notebooks_listing = [x for x in notebooks_paths_listing if re.match("notebooks/[0-9]{3}.*\.ipynb$", x)]
                ipynb_notebook = notebook_name[:-16] + ".ipynb"
                matching_notebooks = [match for match in notebooks_listing if ipynb_notebook in match]
            else:
                raise Exception('Key "tree" is not present in the JSON file')
            if matching_notebooks is not None:
                for n in matching_notebooks:
                    matching_notebooks_paths.append(n)

    def fetch_binder_list(self, file) -> list:
        """Function that fetches list of notebooks with binder buttons

        :param file_format: Format of file containing list of notebooks with button. Defaults to 'txt'
        :type file_format: str
        :return: List of notebooks containing binder buttons
        :rtype: list
        """
        list_of_buttons = glob(f"{self.nb_path}/{file}")
        if list_of_buttons:
            with open(list_of_buttons[0]) as file:
                list_of_buttons = file.read().splitlines()
            return list_of_buttons
        return []

    def fetch_colab_list(self, file) -> list:
        """Function that fetches list of notebooks with colab buttons

        :param file_format: Format of file containing list of notebooks with button. Defaults to 'lst'
        :type file_format: str
        :return: List of notebooks containing colab buttons
        :rtype: list
        """
        list_of_cbuttons = glob(f"{self.nb_path}/{file}")
        if list_of_cbuttons:
            with open(list_of_cbuttons[0]) as file:
                list_of_cbuttons = file.read().splitlines()
            return list_of_cbuttons
        return []


    def add_binder(self, buttons_list: list,  cbuttons_list: list, template_with_colab_and_binder: str = binder_colab_template, template_with_binder: str = binder_template, template_with_colab: str = colab_template, template_without_binder: str = no_binder_template):
        """Function working as an example how to add binder button to existing rst files

        :param buttons_list: List of notebooks that work on Binder.
        :type buttons_list: list
        :param template_with_binder: Template of button added to rst file if Binder is available. Defaults to binder_template.
        :type template_with_binder: str
        :param template_without_binder: Template of button added to rst file if Binder isn't available. Defaults to no_binder_template.
        :type template_without_binder: str
        :raises FileNotFoundError: In case of failure of adding content, error will appear

        """
        for notebook_file, nb_path in zip([
            nb for nb in os.listdir(self.nb_path) if verify_notebook_name(nb)
        ], matching_notebooks_paths):

            notebook_item = '-'.join(notebook_file.split('-')[:-2])

            binder_data = {
                "owner": repo_owner,
                "repo": repo_name,
                "folder": repo_directory,
                "link_git": notebooks_repo + nb_path,
                "link_binder": notebooks_binder + nb_path,
                "link_colab ": notebooks_colab + nb_path,
            }

            if notebook_item in buttons_list:
                template = template_with_colab_and_binder if notebook_item in cbuttons_list else template_with_binder
            else:
                template = template_with_colab if notebook_item in cbuttons_list else template_without_binder

            button_text = create_content(template, binder_data, notebook_file)
            if not add_content_below(button_text, f"{self.nb_path}/{notebook_file}"):
                raise FileNotFoundError("Unable to modify file")


def add_glob_directive():
    """This function modifies toctrees of the five node articles in tutorials 
       section. It adds the notebooks found in docs/notebooks directory to the menu.
    """
    tutorials_path = Path('../../docs/articles_en/learn_openvino/tutorials').resolve(strict=True)
    tutorials_files = [x for x in os.listdir(tutorials_path) if re.match("notebooks_section_[0-9]{1}\.md$", x)]
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

def main():
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
    buttons_list = nbp.fetch_binder_list('notebooks_with_binder_buttons.txt')
    cbuttons_list = nbp.fetch_colab_list('notebooks_with_colab_buttons.txt')
    nbp.add_binder(buttons_list, cbuttons_list)


if __name__ == '__main__':
    main()

