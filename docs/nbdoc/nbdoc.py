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
    repo_branch,
    repo_owner,
)
from notebook import Notebook
from section import Section
from glob import glob
from lxml import html
from jinja2 import Template
from urllib.request import urlretrieve
from requests import get
import os
import sys


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
            page = get(link, verify=False).content
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
        self.binder_data = {
            "owner": repo_owner,
            "repo": repo_name,
            "folder": repo_directory,
            "branch": repo_branch,
        }
        self.colab_data = {
            "owner": repo_owner,
            "repo": repo_name,
            "folder": repo_directory,
        }

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

        for notebook in [
            nb for nb in os.listdir(self.nb_path) if verify_notebook_name(nb)
        ]:
            notebook_item = '-'.join(notebook.split('-')[:-2])

            if notebook_item in buttons_list:
                template = template_with_colab_and_binder if notebook_item in cbuttons_list else template_with_binder
            else:
                template = template_with_colab if notebook_item in cbuttons_list else template_without_binder

            button_text = create_content(template, self.binder_data, notebook)
            if not add_content_below(button_text, f"{self.nb_path}/{notebook}"):
                raise FileNotFoundError("Unable to modify file")

def add_glob_directive(tutorials_file):
        with open(tutorials_file, 'r+', encoding='cp437') as mainfile:
            readfile = mainfile.read()
            if ':glob:' not in readfile:
                add_glob = readfile\
                    .replace(":hidden:\n", ":hidden:\n   :glob:\n")\
                    .replace("notebooks_installation\n", "notebooks_installation\n   notebooks/*\n")
                mainfile.seek(0)
                mainfile.write(add_glob)
                mainfile.truncate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sourcedir', type=Path)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('-d', '--download', action='store_true')
    args = parser.parse_args()
    sourcedir = args.sourcedir
    outdir = args.outdir

    main_tutorials_file = Path('../../docs/tutorials.md').resolve(strict=True)
    add_glob_directive(main_tutorials_file)

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
