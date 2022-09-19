from asyncio import subprocess
from git import Repo
from argparse import ArgumentParser
from utils import utils
from glob import glob
from subprocess import Popen
from shutil import copytree
# from summarize import create_summary
from merge_xmls import merge_xml

import xml.etree.ElementTree as ET

import os

logger = utils.get_logger('ConformanceRunner')

OMZ_REPO_URL = "https://github.com/openvinotoolkit/open_model_zoo.git"
OMZ_REPO_BRANCH = "master"

GTEST_PARALLEL_URL = "https://github.com/asomsiko/gtest-parallel.git"
GTEST_PARALLEL_BRANCH = "limit_log_name"

API_CONFORMANCE_BIN_NAME = "apiConformanceTests"
OP_CONFORMANCE_BIN_NAME = "conformanceTests"
SUBGRAPH_DUMPER_BIN_NAME = "subgraphsDumper"


def parse_arguments():
    parser = ArgumentParser()

    models_path_help = "Path to directory/ies contains models to dump subgraph (default way is download OMZ). If `--d=False` specify Conformance IRs directory"
    device_help = "Specify target device"
    ov_help = "OV binary files path"
    working_dir_help = "Specify working directory"
    type_help = "Specify conformance type: OP or API"
    dump_conformance_help = "Set 'True' if you want to create Conformance IRs from custom models. Default value is 'False'"

    parser.add_argument("-m", "--models_path", help=models_path_help, type=str, required=True)
    parser.add_argument("-d", "--device", help= device_help, type=str, required=True)
    parser.add_argument("-ov", "--ov_binaries", help=ov_help, type=str, required=True)
    parser.add_argument("-w", "--working_dir", help=working_dir_help, type=str, required=False, default='.')
    parser.add_argument("-t", "--type", help=type_help, type=str, required=True, default="OP")
    parser.add_argument("-s", "--dump_conformance", help=dump_conformance_help, type=bool, required=False, default=False)

    return parser.parse_args()

class Conformance:
    def __init__(self, device:str, model_path:str, ov_binaries:str, type:str, working_dir:str):
        self._device = device
        self._model_path = model_path
        self._ov_binaries = ov_binaries
        self._working_dir = working_dir
        if not (type == "OP" or type == "API"):
            raise Exception("Incorrect conformance type")
        self._type = type

    def __download_repo(self, https_url: str, version: str):
        repo_name = https_url[https_url.rfind('/') + 1:len(https_url) - 4]
        repo_path = os.path.join(self._working_dir, repo_name)
        logger.info(f'Started to clone repo: {https_url} to {repo_path}')
        repo = Repo.clone_from(https_url, repo_path)
        repo.submodule_update(recursive=True)
        logger.info(f'Repo {https_url} was cloned sucessful')
        remote_version = "origin/" + version
        if remote_version in repo.git.branch('-r').replace(' ', '').split('\n'):
            branch = repo.git.checkout(version)
            repo.submodule_update(recursive=True)
            logger.info(f'Repo {https_url} is on {version}')
        return repo.working_dir

    def _convert_models(self):
        omz_tools_path = os.path.join(self._omz_path, "tools", "model_tools")
        original_model_path = os.path.join(self._working_dir, "original_models")
        converted_model_path = os.path.join(self._working_dir, "converted_models")
        converter_path = os.path.join(self._omz_path, "tools", "model_tools", 'converter.py')
        mo_path = os.path.join("openvino", "tools", "mo", ".")
        logger.info(f"Model conversion from {original_model_path} to {converted_model_path} is started")
        command = f'python3 -m venv .env3;'\
            f'source .env3/bin/activate;'\
            f'pip3 install -e "{mo_path}/.[caffe,kaldi,mxnet,onnx,pytorch,tensorflow2]";'\
            f'pip3 install "{omz_tools_path}/."'\
            f'omz_converter --all --download_dir={original_model_path} --output_dir={converted_model_path};'\
            f'deactivate;'
        process = Popen(command, stdout=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        if err is None:
            for line in str(out).split('\n'):
                logger.info(line)
        else:
            logger.error(err)
            raise Exception("Process failed on step: 'Model conversion'")
        logger.info(f"Model conversion is successful. Converted models are saved to {converted_model_path}")
        self._model_path = converted_model_path

    def download_and_convert_models(self):
        logger.info("Starting model downloading and conversion")
        self._omz_path = self.__download_repo(OMZ_REPO_URL, OMZ_REPO_BRANCH)
        self._convert_models()
        logger.info("Model downloading and conversion is finished successful")

    def dump_subgraph(self):
        subgraph_dumper_path = os.path.join(self._ov_binaries, SUBGRAPH_DUMPER_BIN_NAME)
        conformance_ir_path = os.path.join(self._working_dir, "conformance_ir")
        if not os.path.exists(conformance_ir_path):
            os.mkdir(conformance_ir_path)
        logger.info(f"Stating model dumping from {self._model_path}")
        cmd = f'{subgraph_dumper_path} --input_folders={self._model_path} --output_folder={conformance_ir_path}'
        process = Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        if err is None:
            for line in str(out).split('\n'):
                logger.info(line)
            logger.info(f"Conformance IRs were saved to {conformance_ir_path}")
        else:
            logger.error(err)
            raise Exception("Process failed on step: 'Subgraph dumping'")
        self._model_path = conformance_ir_path

    def run_conformance(self):
        gtest_parallel_path = os.path.join(self.__download_repo(GTEST_PARALLEL_URL, GTEST_PARALLEL_BRANCH), "gtest_parallel.py")
        worker_num = os.cpu_count()
        conformance_path = None
        if self._type == "OP":
            conformance_path = os.path.join(self._ov_binaries, OP_CONFORMANCE_BIN_NAME)
        else:
            conformance_path = os.path.join(self._ov_binaries, API_CONFORMANCE_BIN_NAME)

        logs_dir = os.path.join(self._working_dir, f'{self._device}_logs')
        report_dir = os.path.join(self._working_dir, 'report')
        parallel_report_dir = os.path.join(report_dir, 'parallel')
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        cmd = f'python3 {gtest_parallel_path}  {conformance_path} -w {worker_num} -d {logs_dir} '\
            f'-- --input_folders={self._model_path} --device={self._device} --report_unique_name --output_folder={parallel_report_dir}'
        logger.info(f"Stating conformance: {cmd}")
        process = Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        if err is None:
            for line in str(out).split('\n'):
                logger.info(line)
        else:
            logger.error(err)
            raise Exception("Process failed on step: 'Run conformance'")
        final_report_name = f'report_{self._type}'
        merge_xml([parallel_report_dir], report_dir, final_report_name, self._type)
        logger.info(f"Conformance is successful. XML reportwas saved to {report_dir}")
        return (os.path.join(report_dir, final_report_name), report_dir)

    # def summarize(self, xml_report_path:os.path, report_dir: os.path):
    #     summary_root = ET.parse(xml_report_path)
    #     create_summary(summary_root, report_dir)
    #     copytree("./template/", report_dir)
    #     logger.info(f"Report was saved to {report_dir}/report.html")

    def start_pipeline(self, dump_models: bool):
        if dump_models:
            if not os.path.exists(self._model_path):
                self._model_path = self.download_and_convert_models()
            self.dump_subgraph()
        if not os.path.isdir(self._model_path):
            raise Exception(f"Directory {self._model_path} does not exist")
        xml_report, report_dir = self.run_conformance()
        # self.summarize(xml_report, report_dir)
        
if __name__ == "__main__":
    args = parse_arguments()
    conformance = Conformance(args.device, args.models_path, args.ov_binaries, args.type, args.working_dir)
    conformance.start_pipeline(args.dump_conformance)