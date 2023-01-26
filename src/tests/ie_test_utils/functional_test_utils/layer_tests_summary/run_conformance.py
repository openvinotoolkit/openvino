# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from subprocess import Popen
from shutil import copytree, rmtree
from summarize import create_summary
from merge_xmls import merge_xml
from run_parallel import TestParallelRunner
from rename_conformance_ir import create_hash
from pathlib import Path

import defusedxml.ElementTree as ET
from urllib.parse import urlparse

import os
import urllib.request as ur
from utils import constants
from utils.conformance_utils import get_logger
from utils import file_utils


logger = get_logger('conformance_runner')

API_CONFORMANCE_BIN_NAME = "apiConformanceTests"
OP_CONFORMANCE_BIN_NAME = "conformanceTests"
SUBGRAPH_DUMPER_BIN_NAME = "subgraphsDumper"

NO_MODEL_CONSTANT = "http://ov-share-03.sclab.intel.com/Shares/conformance_ir/dlb/master/2022.3.0-8953-8c3425ff698.tar"

SCRIPT_DIR_PATH, SCRIPT_NAME = os.path.split(os.path.abspath(__file__))

def get_default_working_dir():
    path = Path(__file__).parent.resolve()
    return os.path.join(path, "temp")

def parse_arguments():
    parser = ArgumentParser()

    models_path_help = "Path to the directory/ies containing models to dump subgraph (the default way is to download conformance IR). It may be directory, archieve file, .lst file or http link to download something . If --s=0, specify the Conformance IRs directory"
    device_help = " Specify the target device. The default value is CPU"
    ov_help = "OV repo path. The default way is try to find the absolute path of OV repo (by using script path)"
    working_dir_help = "Specify a working directory to save all artifacts, such as reports, models, conformance_irs, etc."
    type_help = "Specify conformance type: `OP` or `API`. The default value is `OP`"
    workers_help = "Specify number of workers to run in parallel. The default value is CPU count - 1"
    gtest_filter_helper = "Specify gtest filter to apply when running test. E.g. *Add*:*BinaryConv*. The default value is None"
    dump_conformance_help = "Set '1' if you want to create Conformance IRs from custom/downloaded models. In other cases, set 0. The default value is '1'"

    parser.add_argument("-m", "--models_path", help=models_path_help, type=str, required=False, default=NO_MODEL_CONSTANT)
    parser.add_argument("-d", "--device", help= device_help, type=str, required=False, default="CPU")
    parser.add_argument("-ov", "--ov_path", help=ov_help, type=str, required=False, default=file_utils.get_ov_path(SCRIPT_DIR_PATH))
    parser.add_argument("-w", "--working_dir", help=working_dir_help, type=str, required=False, default=get_default_working_dir())
    parser.add_argument("-t", "--type", help=type_help, type=str, required=False, default="OP")
    parser.add_argument("-j", "--workers", help=workers_help, type=int, required=False, default=os.cpu_count()-1)
    parser.add_argument("--gtest_filter", help=gtest_filter_helper, type=str, required=False, default="*")
    parser.add_argument("-s", "--dump_conformance", help=dump_conformance_help, type=int, required=False, default=0)

    return parser.parse_args()

class Conformance:
    def __init__(self, device:str, model_path:os.path, ov_path:os.path, type:str, workers:int, gtest_filter:str, working_dir:os.path):
        self._device = device
        self._model_path = model_path
        self._ov_path = ov_path
        self._ov_bin_path = file_utils.get_ov_path(SCRIPT_DIR_PATH, self._ov_path, True)
        self._working_dir = working_dir
        if os.path.exists(self._working_dir):
            logger.info(f"Working dir {self._working_dir} is cleaned up")
            rmtree(self._working_dir)
        os.mkdir(self._working_dir)
        if not (type == "OP" or type == "API"):
            logger.error(f"Incorrect conformance type: {type}. Please use 'OP' or 'API'")
            exit(-1)
        self._type = type
        self._workers = workers
        self._gtest_filter = gtest_filter

    def __download_conformance_ir(self):
        _, file_name = os.path.split(urlparse(self._model_path).path)
        model_archieve_path = os.path.join(self._working_dir, file_name)
        try:
            logger.info(f"Conformance IRs will be downloaded from {self._model_path} to {model_archieve_path}")
            ur.urlretrieve(self._model_path, filename=model_archieve_path)
        except:
            logger.error(f"Please verify URL: {self._model_path}. Looks like that is incorrect")
            exit(-1)
        logger.info(f"Conformance IRs were downloaded from {self._model_path} to {model_archieve_path}")
        if not file_utils.is_archieve(model_archieve_path):
            logger.error(f"The file {model_archieve_path} is not archieve! It should be the archieve!")
            exit()
        self._model_path = file_utils.unzip_archieve(model_archieve_path, self._working_dir)

    def __dump_subgraph(self):
        subgraph_dumper_path = os.path.join(self._ov_bin_path, f'{SUBGRAPH_DUMPER_BIN_NAME}{constants.OS_BIN_FILE_EXT}')
        if not os.path.isfile(subgraph_dumper_path):
            logger.error(f"{subgraph_dumper_path} is not exist!")
            exit(-1)
        conformance_ir_path = os.path.join(self._working_dir, "conformance_ir")
        if os.path.isdir(conformance_ir_path):
            logger.info(f"Remove directory {conformance_ir_path}")
            rmtree(conformance_ir_path)
        os.mkdir(conformance_ir_path)
        logger.info(f"Stating model dumping from {self._model_path}")
        cmd = f'{subgraph_dumper_path} --input_folders="{self._model_path}" --output_folder="{conformance_ir_path}"'
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
        if constants.PY_OPENVINO in os.listdir(self._ov_bin_path):
            create_hash(Path(self._model_path))
            logger.info(f"All conformance IRs in {self._ov_bin_path} were renamed based on hash")
        else:
            logger.warning("The OV Python was not built. Skip the step to rename XConformance IR based on a hash")

    def __run_conformance(self):
        conformance_path = None
        if self._type == constants.API_CONFORMANCE:
            conformance_path = os.path.join(self._ov_bin_path, f'{OP_CONFORMANCE_BIN_NAME}{constants.OS_BIN_FILE_EXT}')
        else:
            conformance_path = os.path.join(self._ov_bin_path, f'{API_CONFORMANCE_BIN_NAME}{constants.OS_BIN_FILE_EXT}')

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
        
        command_line_args = [f"--device={self._device}", f'--input_folders="{self._model_path}"', f"--report_unique_name", f'--output_folder="{parallel_report_dir}"', f'--gtest_filter={self._gtest_filter}']
        conformance = TestParallelRunner(f"{conformance_path}", command_line_args, self._workers, logs_dir, "")
        conformance.run()
        conformance.postprocess_logs()
        final_report_name = f'report_{self._type}'
        merge_xml([parallel_report_dir], report_dir, final_report_name, self._type)
        logger.info(f"Conformance is successful. XML reportwas saved to {report_dir}")
        return (os.path.join(report_dir, final_report_name + ".xml"), report_dir)

    def __summarize(self, xml_report_path:os.path, report_dir: os.path):
        summary_root = ET.parse(xml_report_path).getroot()
        create_summary(summary_root, report_dir, [], "", "", False, True)
        copytree(os.path.join(SCRIPT_DIR_PATH, "template"), os.path.join(report_dir, "template"))
        logger.info(f"Report was saved to {os.path.join(report_dir, 'report.html')}")

    def run(self, dump_models: bool):
        command = f'{constants.PIP_NAME} install -r {os.path.join(SCRIPT_DIR_PATH, "requirements.txt")}'
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

        if self._model_path == NO_MODEL_CONSTANT or file_utils.is_url(self._model_path):
            self.__download_conformance_ir()
        if dump_models:
            self.__dump_subgraph()
        if not os.path.exists(self._model_path):
            logger.error(f"The model direstory {self._model_path} does not exist!")
            exit(-1)
        if not os.path.exists(self._model_path):
            logger.error(f"Directory {self._model_path} does not exist")
            exit(-1)
        xml_report, report_dir = self.__run_conformance()
        if self._type == "OP":
            self.__summarize(xml_report, report_dir)

if __name__ == "__main__":
    args = parse_arguments()
    conformance = Conformance(args.device, args.models_path, args.ov_path, args.type, args.workers, args.gtest_filter, args.working_dir)
    conformance.run(args.dump_conformance)
