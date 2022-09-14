from git import Repo
from argparse import ArgumentParser
from utils import utils

import os
import logging

logger = utils.get_logger('RunConformance')

OMZ_REPO_URL = "https://github.com/openvinotoolkit/open_model_zoo.git"

def parse_arguments():
    parser = ArgumentParser()

    ov_version_help = "OpenVINO version - commit hash, branch and so on"
    omz_version_help = "OMZ version - commit hash, branch and so on"
    out_dir_help = "Specify output directory"
    working_dir_help = "Specify working directory"
    remove_help = "Remove conformance artifacts"

    parser.add_argument("-ov", "--ov_version", help=ov_version_help, type=str, default='master', required=True)
    parser.add_argument("-omz", "--omz_version", help=omz_version_help, type=str, default='master', required=True)
    parser.add_argument("-o", "--output_dir", help=out_dir_help, type=str, default='.', required=True)
    parser.add_argument("-w", "--working_dir", help=working_dir_help, type=str, default='.', required=True)
    parser.add_argument("-r", "--remove_artifacts", help=remove_help, type=bool, default=True, required=False)

    return parser.parse_args()

class Conformance:
    def __init__(self, ov_version = "master", omz_version = "master", working_dir = ".", output_dir = "."):
        self._ov_version = ov_version
        self._omz_version = omz_version
        self._working_dir = working_dir
        self._output_dir = output_dir

    def __download_repo(self, https_url: str, version:str):
        repo_name = https_url[https_url.rfind('/') + 1:len(https_url) - 4]
        repo_path = os.path.join(self._working_dir, repo_name)
        logger.info(f'Started to clone repo: {https_url} to {repo_path}')
        repo = Repo.clone_from(https_url, repo_path)
        repo.submodule_update(recursive=True)
        logger.info(f'Repo {https_url} was cloned sucessful')
        remote_version = "origin/" + version
        branches = repo.git.branch('-r').replace(' ', '').split('\n')
        if remote_version in branches:
            branch = repo.git.checkout(version)
            repo.submodule_update(recursive=True)
            logger.info(f'Repo {https_url} is on {version}')
        return repo.working_dir

    def download_models(self):
        self._omz_path = self.__download_repo(OMZ_REPO_URL, self._omz_version)
        downloader_path = os.path.join(self._omz_path, "tools", "model_tools")

    def _convert_models(self):
        pass

    def _dump_subgraph(self):
        pass

    def _run_conformance(self):
        pass

    def _summarize(self):
        pass

    def _remove_artifacts(self):
        pass

    def start_pipeline(self):
        self.download_models()

if __name__ == "__main__":
    args = parse_arguments()
    conformance = Conformance(args.ov_version, args.omz_version, args.working_dir, args.output_dir)
    conformance.start_pipeline()
    # save_to_file(args.skip_config_folders, get_conformance_hung_test(args.input_logs), args.extend_file)