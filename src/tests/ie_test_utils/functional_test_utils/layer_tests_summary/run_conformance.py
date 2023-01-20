from asyncio import subprocess
from queue import Empty
from git import Repo
from argparse import ArgumentParser
from utils import utils
from glob import glob
from subprocess import Popen
from shutil import copytree, rmtree
from summarize import create_summary
from merge_xmls import merge_xml
from run_parallel import TestParallelRunner
from pathlib import Path, PurePath
from sys import version, platform

import defusedxml.ElementTree as ET

import os

logger = utils.get_logger('ConformanceRunner')

OPENVINO_NAME = 'openvino'

OMZ_REPO_URL = "https://github.com/openvinotoolkit/open_model_zoo.git"
OMZ_REPO_BRANCH = "master"

GTEST_PARALLEL_URL = "https://github.com/intel-innersource/frameworks.ai.openvino.ci.infrastructure.git"
GTEST_PARALLEL_BRANCH = "master"

API_CONFORMANCE_BIN_NAME = "apiConformanceTests"
OP_CONFORMANCE_BIN_NAME = "conformanceTests"
SUBGRAPH_DUMPER_BIN_NAME = "subgraphsDumper"

DEBUG_DIR = "Debug"
RELEASE_DIR = "Release"

IS_WIN = "windows" in platform or "win32" in platform

OS_SCRIPT_EXT = ".bat" if IS_WIN else ""
OS_BIN_FILE_EXT = ".exe" if IS_WIN else ""

NO_MODEL_CONSTANT = "NO_MODEL"

ENV_SEPARATOR = ";" if IS_WIN else ":"

PYTHON_NAME = "python" if IS_WIN else "python3"
PIP_NAME = "pip" if IS_WIN else "pip3"

SCRIPT_DIR_PATH, SCRIPT_NAME = os.path.split(os.path.abspath(__file__))

def find_latest_dir(in_dir: Path, pattern_list = list()):
    get_latest_dir = lambda path: sorted(Path(path).iterdir(), key=os.path.getmtime)
    entities = get_latest_dir(in_dir)
    entities.reverse()

    for entity in entities:
        if entity.is_dir():
            if not pattern_list:
                return entity
            else:
                for pattern in pattern_list: 
                    if pattern in str(os.fspath(PurePath(entity))):
                        return entity
    logger.error(f"{in_dir} does not contain applicable directories to patterns: {pattern_list}")
    exit(-1)

def get_ov_path(ov_dir=None, is_bin=False):
    if ov_dir is None or not os.path.isdir(ov_dir):
        if 'INTEL_OPENVINO_DIR' in os.environ:
            ov_dir = os.environ['INTEL_OPENVINO_DIR']
        else:
            ov_dir = os.path.abspath(SCRIPT_DIR_PATH)[:os.path.abspath(SCRIPT_DIR_PATH).find(OPENVINO_NAME) + len(OPENVINO_NAME)]
    if is_bin:
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir, ['bin']))
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir))
        ov_dir = os.path.join(ov_dir, find_latest_dir(ov_dir, [DEBUG_DIR, RELEASE_DIR]))
    return ov_dir

def get_default_working_dir():
    path = Path(__file__).parent.resolve()
    return os.path.join(path, "temp")

def parse_arguments():
    parser = ArgumentParser()

    models_path_help = "Path to the directory/ies containing models to dump subgraph (the default way is to download OMZ). If --s=0, specify the Conformance IRs directory"
    device_help = " Specify the target device. The default value is CPU"
    ov_help = "OV binary files path. The default way is try to find installed OV by INTEL_OPENVINO_DIR in environmet variables or to find the absolute path of OV repo (by using script path)"
    working_dir_help = "Specify a working directory to save all artifacts, such as reports, models, conformance_irs, etc."
    type_help = "Specify conformance type: `OP` or `API`. The default value is `OP`"
    workers_help = "Specify number of workers to run in parallel. The default value is CPU count - 1"
    gtest_filter_helper = "Specify gtest filter to apply when running test. E.g. *Add*:*BinaryConv*. The default value is None"
    dump_conformance_help = "Set '1' if you want to create Conformance IRs from custom/downloaded models. In other cases, set 0. The default value is '1'"

    parser.add_argument("-m", "--models_path", help=models_path_help, type=str, required=False, default=NO_MODEL_CONSTANT)
    parser.add_argument("-d", "--device", help= device_help, type=str, required=False, default="CPU")
    parser.add_argument("-ov", "--ov_path", help=ov_help, type=str, required=False, default=get_ov_path())
    parser.add_argument("-w", "--working_dir", help=working_dir_help, type=str, required=False, default=get_default_working_dir())
    parser.add_argument("-t", "--type", help=type_help, type=str, required=False, default="OP")
    parser.add_argument("-j", "--workers", help=workers_help, type=int, required=False, default=os.cpu_count()-1)
    parser.add_argument("--gtest_filter", help=gtest_filter_helper, type=str, required=False, default=None)
    parser.add_argument("-s", "--dump_conformance", help=dump_conformance_help, type=int, required=False, default=1)

    return parser.parse_args()

def set_env_variable(env: os.environ, var_name: str, var_value: str):
    if var_name in env:
        env[var_name] = var_value + ENV_SEPARATOR + env[var_name]
    else:
        env[var_name] = var_value
    return env

class Conformance:
    def __init__(self, device:str, model_path:os.path, ov_path:os.path, type:str, workers:int, gtest_filter:str, working_dir:os.path):
        self._device = device
        self._model_path = model_path
        self._ov_path = ov_path
        self._ov_bin_path = get_ov_path(self._ov_path, True)
        self._working_dir = working_dir
        if not (type == "OP" or type == "API"):
            logger.error(f"Incorrect conformance type: {type}. Please use 'OP' or 'API'")
            exit(-1)
        self._type = type
        self._workers = workers
        if not gtest_filter:
            gtest_filter = "*"
        self._gtest_filter = gtest_filter

    def __download_repo(self, https_url: str, version: str):
        repo_name = https_url[https_url.rfind('/') + 1:len(https_url) - 4]
        repo_path = os.path.join(self._working_dir, repo_name)
        if os.path.isdir(repo_path):
            logger.info(f'Repo: {repo_name} exists in {self._working_dir}. Skip the repo download.')
            repo = Repo(repo_path)
        else:
            logger.info(f'Started to clone repo: {https_url} to {repo_path}')
            repo = Repo.clone_from(https_url, repo_path)
            repo.submodule_update(recursive=True)
            logger.info(f'Repo {https_url} was cloned sucessful')
        remote_version = "origin/" + version
        if remote_version in repo.git.branch('-r').replace(' ', '').split('\n'):
            repo.git.checkout(version)
            repo.git.pull()
            repo.submodule_update(recursive=True)
            logger.info(f'Repo {https_url} is on {version}')
        return repo_path

    def _convert_models(self):
        omz_tools_path = os.path.join(self._omz_path, "tools", "model_tools")
        original_model_path = os.path.join(self._working_dir, "original_models")
        converted_model_path = os.path.join(self._working_dir, "converted_models")
        if os.path.isdir(original_model_path):
            logger.info(f"Original model path: {original_model_path} is removed")
            rmtree(original_model_path)
        if os.path.isdir(converted_model_path):
            logger.info(f"Converted model path: {converted_model_path} is removed")
            rmtree(converted_model_path)
        mo_path = os.path.join(self._ov_path, "tools", "mo")
        ov_python_path = os.path.join(self._ov_bin_path, "python_api", f"python{version[0:3]}")

        convert_model_env = os.environ.copy()
        ld_lib_path_name = ""
        # Windows or MacOS
        if IS_WIN or platform == "darwin":
            ld_lib_path_name = "PATH"
        # Linux
        elif "lin" in platform:
            ld_lib_path_name = "LD_LIBRARY_PATH"
        convert_model_env = set_env_variable(convert_model_env, ld_lib_path_name, self._ov_bin_path)
        convert_model_env = set_env_variable(convert_model_env, "PYTHONPATH", f"{ov_python_path}{ENV_SEPARATOR}{mo_path}")
        convert_model_env = set_env_variable(convert_model_env, "OMZ_ROOT", self._omz_path)

        logger.info(f"Model conversion from {original_model_path} to {converted_model_path} is started")
        activate_path = os.path.join(".env3", "bin", "activate")

        command = f'cd "{self._working_dir}"; ' \
            f'{"" if os.path.isdir(".env3") else f"{PYTHON_NAME} -m venv .env3; "} '\
            f'{"" if IS_WIN else "source"} {activate_path}{OS_SCRIPT_EXT}; '\
            f'{PIP_NAME} install -e "{mo_path}/.[caffe,kaldi,mxnet,onnx,pytorch,tensorflow2]"; ' \
            f'{PIP_NAME} install "{omz_tools_path}/.[paddle,pytorch,tensorflow]"; ' \
            f'omz_downloader --all --output_dir="{original_model_path}"; '\
            f'omz_converter --all --download_dir="{original_model_path}" --output_dir="{converted_model_path}"; '\
            f'deactivate'
        try:
            process = Popen(command, shell=True, env=convert_model_env)
            out, err = process.communicate()
            if err is None:
                for line in str(out).split('\n'):
                    logger.info(line)
            else:
                logger.error(err)
                exit(-1)
            logger.info(f"Model conversion is successful. Converted models are saved to {converted_model_path}")
        except:
            logger.error(f"Something is wrong with the model conversion! Abort the process")
            exit(-1)
        return converted_model_path

    def download_and_convert_models(self):
        logger.info("Starting model downloading and conversion")
        self._omz_path = self.__download_repo(OMZ_REPO_URL, OMZ_REPO_BRANCH)
        self._model_path = self._convert_models()
        logger.info("Model downloading and conversion is finished successful")

    def dump_subgraph(self):
        subgraph_dumper_path = os.path.join(self._ov_bin_path, SUBGRAPH_DUMPER_BIN_NAME)
        if not os.path.isfile(subgraph_dumper_path):
            logger.error(f"{subgraph_dumper_path} is not exist!")
            exit(-1)
        conformance_ir_path = os.path.join(self._working_dir, "conformance_ir")
        if os.path.isdir(conformance_ir_path):
            logger.info(f"Remove directory {conformance_ir_path}")
            rmtree(conformance_ir_path)
        os.mkdir(conformance_ir_path)
        logger.info(f"Stating model dumping from {self._model_path}")
        cmd = f'{subgraph_dumper_path}{OS_BIN_FILE_EXT} --input_folders="{self._model_path}" --output_folder="{conformance_ir_path}"'
        process = Popen(cmd, shell=True)
        out, err = process.communicate()
        if err is None:
            for line in str(out).split('\n'):
                logger.info(line)
            logger.info(f"Conformance IRs were saved to {conformance_ir_path}")
        else:
            logger.error(err)
            logger.error("Process failed on step: 'Subgraph dumping'")
            exit(-1)
        self._model_path = conformance_ir_path

    def run_conformance(self):
        conformance_path = None
        if self._type == "OP":
            conformance_path = os.path.join(self._ov_bin_path, OP_CONFORMANCE_BIN_NAME)
        else:
            conformance_path = os.path.join(self._ov_bin_path, API_CONFORMANCE_BIN_NAME)

        if not os.path.isfile(conformance_path):
            logger.error(f"{conformance_path} is not exist!")
            exit(-1)

        logs_dir = os.path.join(self._working_dir, f'{self._device}_logs')
        report_dir = os.path.join(self._working_dir, 'report')
        if os.path.isdir(report_dir):
            logger.info(f"Report dir {report_dir} is cleaned up")
            rmtree(report_dir)
        parallel_report_dir = os.path.join(report_dir, 'parallel')
        if not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        if not os.path.isdir(logs_dir):
            os.mkdir(logs_dir)
        
        try:
            command_line_args = [f"--device={self._device}", f'--input_folders="{self._model_path}"', f"--report_unique_name", f'--output_folder="{parallel_report_dir}"', f'--gtest_filter={self._gtest_filter}']
            conformance = TestParallelRunner(f"{conformance_path}{OS_BIN_FILE_EXT}", command_line_args, self._workers, logs_dir, "")
            conformance.run()
            conformance.postprocess_logs()
        except:
            logger.error(f"Please check the output from `parallel_runner`. Something is wrong")
            exit(-1)
        final_report_name = f'report_{self._type}'
        merge_xml([parallel_report_dir], report_dir, final_report_name, self._type)
        logger.info(f"Conformance is successful. XML reportwas saved to {report_dir}")
        return (os.path.join(report_dir, final_report_name + ".xml"), report_dir)

    def summarize(self, xml_report_path:os.path, report_dir: os.path):
        summary_root = ET.parse(xml_report_path).getroot()
        create_summary(summary_root, report_dir, [], "", "", False, True)
        copytree(os.path.join(SCRIPT_DIR_PATH, "template"), os.path.join(report_dir, "template"))
        logger.info(f"Report was saved to {os.path.join(report_dir, 'report.html')}")

    def start_pipeline(self, dump_models: bool):
        command = f'{PIP_NAME} install -r {os.path.join(SCRIPT_DIR_PATH, "requirements.txt")}'
        process = Popen(command, shell=True)
        out, err = process.communicate()
        if err is None:
            for line in str(out).split('\n'):
                logger.info(line)
        else:
            logger.error(err)
            logger.error("Impossible to install requirements!")
            exit(-1)
        logger.info(f"[ARGUMENTS] --device = {self._device}")
        logger.info(f"[ARGUMENTS] --ov_path = {self._ov_path}")
        logger.info(f"[ARGUMENTS] --models_path = {self._model_path}")
        logger.info(f"[ARGUMENTS] --working_dir = {self._working_dir}")
        logger.info(f"[ARGUMENTS] --type = {self._type}")
        logger.info(f"[ARGUMENTS] --workers = {self._workers}")
        logger.info(f"[ARGUMENTS] --gtest_filter = {self._gtest_filter}")
        logger.info(f"[ARGUMENTS] --dump_conformance = {dump_models}")

        if dump_models:
            if self._model_path == NO_MODEL_CONSTANT:
                self.download_and_convert_models()
            self.dump_subgraph()
        if not os.path.exists(self._model_path):
            logger.error(f"Directory {self._model_path} does not exist")
            exit(-1)
        xml_report, report_dir = self.run_conformance()
        if self._type == "OP":
            self.summarize(xml_report, report_dir)

if __name__ == "__main__":
    args = parse_arguments()
    conformance = Conformance(args.device, args.models_path, args.ov_path, args.type, args.workers, args.gtest_filter, args.working_dir)
    conformance.start_pipeline(args.dump_conformance)
