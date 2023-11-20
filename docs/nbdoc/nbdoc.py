import argparse
import shutil
from pathlib import Path
from utils import (
    create_content,
    add_content_below,
    process_notebook_name,
    verify_notebook_name,
    split_notebooks_into_sections,
)
from consts import (
    artifacts_link,
    binder_template,
    colab_template,
    binder_colab_template,
    blacklisted_extensions,
    notebooks_docs,
    notebooks_path,
    no_binder_template,
    repo_directory,
    repo_name,
    openvino_notebooks_ipynb_list,
    repo_owner,
    notebooks_repo,
    notebooks_binder,
    notebooks_colab,
    rst_template,
    section_names,
)
from notebook import Notebook
from section import Section
from glob import glob
from lxml import html
from jinja2 import Template
from urllib.request import urlretrieve
from requests import get
import os
import re

matching_notebooks_paths = []

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
        notebooks = [
            Notebook(
                name=process_notebook_name(notebook),
                path=notebook,
            )
            for notebook in os.listdir(self.nb_path)
            if verify_notebook_name(notebook)
        ]
        notebooks = split_notebooks_into_sections(notebooks)
        self.rst_data = {
            "sections": [
                Section(name=section_name, notebooks=section_notebooks)
                for section_name, section_notebooks in zip(section_names, notebooks)
            ]

        }

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

    def render_rst(self, path: str = notebooks_docs, template: str = rst_template):
        """Rendering rst file for all notebooks

        :param path: Path to notebook main rst file. Defaults to notebooks_docs.
        :type path: str
        :param template: Template for default rst page. Defaults to rst_template.
        :type template: str

        """
        with open(path, "w+") as nb_file:
            nb_file.writelines(Template(template).render(self.rst_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sourcedir', type=Path)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('-d', '--download', action='store_true')
    args = parser.parse_args()
    sourcedir = args.sourcedir
    outdir = args.outdir
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
    nbp.render_rst(outdir.joinpath(notebooks_docs))


if __name__ == '__main__':
    main()