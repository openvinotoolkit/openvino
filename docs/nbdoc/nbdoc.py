import argparse
from pathlib import Path
from utils import (
    create_content,
    add_content_below,
    load_secret,
    process_notebook_name,
    find_latest_artifact,
    verify_notebook_name,
    generate_artifact_link,
    remove_existing,
    split_notebooks_into_sections,
)
from consts import (
    binder_template,
    no_binder_template,
    rst_template,
    notebooks_path,
    repo_owner,
    repo_name,
    repo_directory,
    notebooks_docs,
    section_names
)
from notebook import Notebook
from section import Section
from io import BytesIO
from glob import glob
from jinja2 import Template
from requests import get
from zipfile import ZipFile
import os


class NbDownloader:
    """Class responsible for downloading and extracting notebooks"""

    def __init__(self, secret_path: str) -> None:
        self.secret = load_secret(secret_path)
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {self.secret}",
        }
        self.artifact_link = generate_artifact_link(repo_owner, repo_name)

    def default_pipeline(self, path: str = notebooks_path) -> bool:
        """Default pipeline for fetching, downloading and extracting rst files

        :param path: Path to folder that will contain notebooks. Defaults to notebooks_path.
        :type path: str
        :returns: Returns if status is sucessful
        :rtype: bool

        """
        artifacts = self.fetch_artifacts()
        latest_artifact = find_latest_artifact(artifacts)
        download_link = self.generate_artifact_download_link(latest_artifact)
        zipfile = self.download_rst_files(download_link)
        if zipfile.testzip() is None:
            remove_existing(path)
        return self.extract_artifacts(zipfile, path=path)

    def fetch_artifacts(self) -> dict:
        """Fetching artifcats from github actions

        :returns: Artifacts in repo
        :rtype: dict

        """
        return get(self.artifact_link, headers=self.headers).json()

    def generate_artifact_download_link(self, artifact_id: int) -> str:
        """Generate link based on link and latest artifact id containing rst files

        :param artifact_id: Latest artifact id containing rst files
        :type artifact_id: int
        :returns: Link to download rst files
        :rtype: str

        """
        return f"{self.artifact_link}/{artifact_id}/zip"

    def download_rst_files(self, artifact_download_link: str) -> ZipFile:
        """Downloading rst files

        :param artifact_download_link: Generated link for downloading rst
        :type artifact_download_link: str
        :returns: Zipped archive of rst files
        :rtype: ZipFile

        """
        artifact = get(artifact_download_link, headers=self.headers)
        return ZipFile(BytesIO(artifact.content))

    def extract_artifacts(self, zipfile: ZipFile, path: str) -> bool:
        """Extracting all artifacts from zipped archive

        :param zipfile: zipped rst files
        :type zipfile: ZipFile
        :param path: path to extract files to
        :type path: str
        :returns: Returns if status is sucessful
        :rtype: bool

        """
        try:
            zipfile.extractall(path=path)
            return True
        except ValueError:
            return False


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
        self.binder_data = {
            "owner": repo_owner,
            "repo": repo_name,
            "folder": repo_directory,
        }

    def fetch_binder_list(self, file_format: str = 'txt') -> list:
        """Funtion that fetches list of notebooks with binder buttons

        :param file_format: Format of file containing list of notebooks with button. Defaults to 'txt'
        :type file_format: str
        :return: List of notebooks conaining binder buttons
        :rtype: list
        """
        list_of_buttons = glob(f"{self.nb_path}/*.{file_format}")
        if list_of_buttons:
            with open(list_of_buttons[0]) as file:
                list_of_buttons = file.read().splitlines()
            return list_of_buttons
        else:
            return []

    def add_binder(self, buttons_list: list,  template_with_binder: str = binder_template, template_without_binder: str = no_binder_template):
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
            if '-'.join(notebook.split('-')[:-2]) in buttons_list:
                button_text = create_content(template_with_binder, self.binder_data, notebook)
                if not add_content_below(button_text, f"{self.nb_path}/{notebook}"):
                    raise FileNotFoundError("Unable to modify file")
            else:
                button_text = create_content(template_without_binder, self.binder_data, notebook)
                if not add_content_below(button_text, f"{self.nb_path}/{notebook}"):
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
    parser.add_argument('secret', type=Path)
    parser.add_argument('outdir', type=Path)
    args = parser.parse_args()
    secret = args.secret
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    # Step 1. Create secret file
    # link: https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token
    # For this notebooks purpose only repo -> public_repo box is required
    nbd = NbDownloader(secret)
    # Step 2. Run default pipeline for downloading
    if not nbd.default_pipeline(outdir):
        raise FileExistsError("Files not downloaded")
    # Step 3. Run processing on downloaded file
    nbp = NbProcessor(outdir)
    buttons_list = nbp.fetch_binder_list('txt')
    nbp.add_binder(buttons_list)
    nbp.render_rst(outdir.joinpath(notebooks_docs))


if __name__ == '__main__':
    main()
