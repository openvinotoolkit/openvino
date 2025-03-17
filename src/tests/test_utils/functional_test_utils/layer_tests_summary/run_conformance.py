# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import urllib.request as ur
from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree, rmtree, copyfile
from subprocess import Popen
from urllib.parse import urlparse

import defusedxml.ElementTree as ET

from merge_xmls import merge_xml
from run_parallel import TestParallelRunner
from summarize import create_summary, create_api_summary
from utils import constants
from utils import file_utils
from utils.conformance_utils import get_logger

logger = get_logger('conformance_runner')
has_python_api = True
try:
    from rename_conformance_ir import create_hash, save_rel_weights
except:
    logger.warning("Please set the above env variable to get the same conformance ir names run by run!")
    has_python_api = False

API_CONFORMANCE_BIN_NAME = "ov_api_conformance_tests"
OP_CONFORMANCE_BIN_NAME = "ov_op_conformance_tests"
SUBGRAPH_DUMPER_BIN_NAME = "ov_subgraphs_dumper"

SCRIPT_DIR_PATH, SCRIPT_NAME = os.path.split(os.path.abspath(__file__))
NO_MODEL_CONSTANT = os.path.join(SCRIPT_DIR_PATH, "data", "models.lst")


def get_default_working_dir():
    path = Path(__file__).parent.resolve()
    return os.path.join(path, "temp")


def parse_arguments():
    parser = ArgumentParser()

    models_path_help = "Path to the directory/ies containing models to dump subgraph (the default way is to download conformance IR). It may be directory, archieve file, .lst file with models to download by a link, model file paths. If --s=0, specify the Conformance IRs directory. NOTE: Applicable only for Opset Conformance."
    dump_graph_help = "Set '1' to create Conformance IRs from models using subgraphsDumper tool. The default value is '0'. NOTE: Applicable only for Opset Conformance."
    device_help = "Specify a target device. The default value is `CPU`"
    ov_help = "OV binary path. The default way is to find the absolute path of latest bin in the repo (by using script path)"
    working_dir_help = "Specify a working directory to save a run artifacts"
    type_help = "Specify conformance type: `OP` or `API`. The default value is `OP`"
    workers_help = "Specify number of workers to run in parallel. The default value is `CPU_count`"
    gtest_filter_helper = "Specify gtest filter to apply for a test run. E.g. *Add*:*BinaryConv*. The default value is None"
    ov_config_path_helper = "Specify path to a plugin config file as `.lst` file. Default value is ``"
    special_mode_help = "Specify shape mode (`static`, `dynamic` or ``) for Opset conformance or API scope type (`mandatory` or ``). Default value is ``"
    entity_help = "Specify validation entity: `Inference`, `ImportExport`, `QueryModel` or `OpImpl` for `OP` or "\
        "`ov_compiled_model`, `ov_infer_request` or `ov_plugin` for `API`. Default value is ``(all)"
    parallel_help = "Parallel over HW devices. For example run tests over GPU.0 and GPU.1 in case when device are the same"
    expected_failures_help = "Excepted failures list file path as csv"
    cache_path_help = "Path to the cache file with test_name list sorted by execution time as `.lst` file!"
    expected_failures_update_help = "Overwrite expected failures list in case same failures were fixed"
    disable_rerun_help = "Disable re-run of interapted/lost tests. Default value is `False`"
    timeout_help = "Set a custom timeout per worker in s"

    parser.add_argument("-d", "--device", help=device_help, type=str, required=False, default="CPU")
    parser.add_argument("-t", "--type", help=type_help, type=str, required=False, default=constants.OP_CONFORMANCE)
    parser.add_argument("--gtest_filter", help=gtest_filter_helper, type=str, required=False, default="*")
    parser.add_argument("-w", "--working_dir", help=working_dir_help, type=str, required=False,
                        default=get_default_working_dir())
    parser.add_argument("-m", "--models_path", help=models_path_help, type=str, required=False,
                        default=NO_MODEL_CONSTANT)
    parser.add_argument("-ov", "--ov_path", help=ov_help, type=str, required=False, default="")
    parser.add_argument("-j", "--workers", help=workers_help, type=int, required=False, default=os.cpu_count())
    parser.add_argument("-c", "--ov_config_path", help=ov_config_path_helper, type=str, required=False, default="")
    parser.add_argument("-s", "--dump_graph", help=dump_graph_help, type=int, required=False, default=0)
    parser.add_argument("-sm", "--special_mode", help=special_mode_help, type=str, required=False, default="")
    parser.add_argument("-e", "--entity", help=entity_help, type=str, required=False, default="")
    parser.add_argument("-p", "--parallel_devices", help=parallel_help, type=bool, required=False, default=False)
    parser.add_argument("-f", "--expected_failures", help=expected_failures_help, type=str, required=False, default="")
    parser.add_argument("-u", "--expected_failures_update", help=expected_failures_update_help, required=False,
                        default=False, action='store_true')
    parser.add_argument("--cache_path", help=cache_path_help, type=str, required=False, default="")
    parser.add_argument("-r", "--disable_rerun", help=disable_rerun_help, required=False, default=False, type=bool)
    parser.add_argument("--timeout", help=timeout_help, required=False, default=-1, type=int)

    return parser.parse_args()


class Conformance:
    def __init__(self, device: str, model_path: os.path, ov_path: os.path, type: str, workers: int,
                 gtest_filter: str, working_dir: os.path, ov_config_path: os.path, special_mode: str,
                 entity:str, cache_path: str, parallel_devices: bool, expected_failures_file: str,
                 expected_failures_update: bool, disable_is_rerun: bool, timeout: int):
        self._device = device
        self._model_path = model_path
        if os.path.isdir(ov_path):
            self._ov_path = ov_path
        else:
            self._ov_path = file_utils.get_ov_path(SCRIPT_DIR_PATH, None, True)
        self._working_dir = working_dir
        if os.path.exists(self._working_dir):
            logger.info(f"Working dir {self._working_dir} is cleaned up")
            rmtree(self._working_dir)
        os.mkdir(self._working_dir)
        self._cache_path = cache_path if os.path.isfile(cache_path) else ""
        self.__entity = ""
        if type == constants.OP_CONFORMANCE:
            if entity == "Inference" or entity == "QueryModel" or entity == "ImportExport" or entity == "" or entity == "OpImpl":
                self.__entity = entity
            else:
                logger.error(f'Incorrect value to set entity type: {special_mode}. Please check `help` to get possible values')
                exit(-1)
            if special_mode == "static" or special_mode == "dynamic" or special_mode == "":
                self._special_mode = special_mode
            else:
                logger.error(f'Incorrect value to set shape mode: {special_mode}. Please check `help` to get possible values')
                exit(-1)
            self._gtest_filter = f"*{self.__entity}*{gtest_filter}*{'' if self.__entity == 'OpImpl' else ':-*OpImpl*'}"
        elif type == constants.API_CONFORMANCE:
            if entity == "ov_compiled_model" or entity == "ov_plugin" or entity == "ov_infer_request" or entity == "":
                self.__entity = entity
            else:
                logger.error(f'Incorrect value to set shape mode: {special_mode}. Please check to get possible values')
                exit(-1)
            self._special_mode = ""
            if special_mode == "mandatory":
                self._gtest_filter = f"*{self.__entity}*mandatory*{gtest_filter}*:*{self.__entity}*{gtest_filter}*mandatory*"
            elif special_mode == "":
                self._gtest_filter = f"*{self.__entity}*{gtest_filter}*"
            else:
                logger.error(f'Incorrect value to set API scope: {special_mode}. Please check to get possible values')
                exit(-1)
        else:
            logger.error(
                f"Incorrect conformance type: {type}. Please use '{constants.OP_CONFORMANCE}' or '{constants.API_CONFORMANCE}'")
            exit(-1)
        self._type = type
        self._workers = workers
        if not os.path.exists(ov_config_path) and ov_config_path != "":
            logger.error(f"Specified config file does not exist: {ov_config_path}.")
            exit(-1)
        self._ov_config_path = ov_config_path
        self._is_parallel_over_devices = parallel_devices
        self._expected_failures = set()
        self._unexpected_failures = set()
        self._expected_failures_file = expected_failures_file
        if os.path.isfile(expected_failures_file):
            self._expected_failures = self.__get_failed_test_from_csv(expected_failures_file)
        else:
            logger.warning(f"Expected failures testlist `{self._expected_failures_file}` does not exist!")
        self._expected_failures_update = expected_failures_update

        self.is_successful_run = False
        self._is_rerun = not disable_is_rerun
        self._timeout = timeout

    def __download_models(self, url_to_download, path_to_save):
        _, file_name = os.path.split(urlparse(url_to_download).path)
        download_path = os.path.join(path_to_save, file_name)
        try:
            logger.info(f"Conformance IRs will be downloaded from {url_to_download} to {download_path}")
            ur.urlretrieve(url_to_download, filename=download_path)
        except Exception as exc:
            logger.error(f"Please verify URL: {url_to_download}. It might be incorrect. See below for the full error.")
            logger.exception(f'FULL ERROR: {exc}')
            exit(-1)
        logger.info(f"Conformance IRs were downloaded from {url_to_download} to {download_path}")
        if not os.path.isfile(download_path):
            logger.error(f"{download_path} is not a file. Exit!")
            exit(-1)
        if file_utils.is_archieve(download_path):
            logger.info(f"The file {download_path} is archived. Should be unzipped to {path_to_save}")
            return file_utils.unzip_archieve(download_path, path_to_save)
        return download_path

    def __dump_subgraph(self):
        subgraph_dumper_path = os.path.join(self._ov_path, f'{SUBGRAPH_DUMPER_BIN_NAME}{constants.OS_BIN_FILE_EXT}')
        if not os.path.isfile(subgraph_dumper_path):
            logger.error(f"{subgraph_dumper_path} is not exist!")
            exit(-1)
        conformance_ir_path = os.path.join(self._working_dir, "conformance_ir")
        if os.path.isdir(conformance_ir_path):
            logger.info(f"Remove directory {conformance_ir_path}")
            rmtree(conformance_ir_path)
        os.mkdir(conformance_ir_path)
        self._model_path = file_utils.prepare_filelist(self._model_path,
                                                       constants.SUPPORTED_MODEL_EXTENSION)
        logger.info(f"Stating model dumping from {self._model_path}")
        log_path = os.path.join(self._working_dir, "ov_subgraphs_dumper.log")
        logger.info(f"ov_subgraphs_dumper.log will be saved to: {log_path}")
        cmd = f'{subgraph_dumper_path} --input_folders="{self._model_path}" --output_folder="{conformance_ir_path}" > {log_path}'
        process = Popen(cmd, shell=True)
        out, err = process.communicate()
        if err is None:
            logger.info(f"Conformance IRs were saved to {conformance_ir_path}")
        else:
            logger.error(err)
            logger.error("Process failed on step: 'Subgraph dumping'")
            exit(-1)
        self._model_path = conformance_ir_path
        if has_python_api:
            op_rel_weight = create_hash(Path(self._model_path))
            save_rel_weights(Path(self._model_path), op_rel_weight)
            logger.info(f"All conformance IRs in {self._model_path} were renamed based on hash")
        else:
            logger.warning(
                "The OV Python was not built or Environment was not updated to requirements. "
                "Skip the step to rename Conformance IR based on a hash")

    @staticmethod
    def __get_failed_test_from_csv(csv_file: str):
        failures = set()
        with open(csv_file, "r") as failures_file:
            for row in csv.reader(failures_file, delimiter=','):
                if row[0] == "Test Name":
                    continue
                failures.add(row[0])
            failures_file.close()
        return failures

    def __update_expected_failures(self):
        this_failures_file = os.path.join(self._working_dir, f"{self._device}_logs", "logs", "fix_priority.csv")
        if not os.path.isfile(this_failures_file):
            return
        this_run_failures = self.__get_failed_test_from_csv(this_failures_file)

        # we do not want to update the expected failures file if there are failures that were not present
        # in the passed expected failures file, i.e. if len(self._unexpected_failures) > 0
        if this_run_failures != self._expected_failures and self._expected_failures_update and \
                not len(self._unexpected_failures):
            logger.info(f"Expected failures file {self._expected_failures_file} will be updated! "
                        f"The following will be deleted as they are passing now: "
                        f"{self._expected_failures.difference(this_failures_file)}")
            os.remove(self._expected_failures_file)
            copyfile(this_failures_file, self._expected_failures_file)

            self.is_successful_run = True

    def __run_conformance(self):
        conformance_path = None
        if self._type == constants.OP_CONFORMANCE:
            conformance_path = os.path.join(self._ov_path, f'{OP_CONFORMANCE_BIN_NAME}{constants.OS_BIN_FILE_EXT}')
        else:
            conformance_path = os.path.join(self._ov_path, f'{API_CONFORMANCE_BIN_NAME}{constants.OS_BIN_FILE_EXT}')

        if not os.path.isfile(conformance_path):
            logger.error(f"{conformance_path} does not exist!")
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

        command_line_args = [f"--device={self._device}",
                             f'--input_folders="{self._model_path}"' if self._type == constants.OP_CONFORMANCE else '',
                             f"--report_unique_name", f'--output_folder="{parallel_report_dir}"',
                             f'--gtest_filter=\"{self._gtest_filter}\"', f'--config_path="{self._ov_config_path}"',
                             f'--shape_mode={self._special_mode}']
        conformance = TestParallelRunner(exec_file_path=f"{conformance_path}",
                                         test_command_line=command_line_args,
                                         worker_num=self._workers,
                                         working_dir=logs_dir,
                                         cache_path=self._cache_path,
                                         split_unit=constants.TEST_UNIT_NAME,
                                         repeat_failed=self._is_rerun,
                                         is_parallel_devices=self._is_parallel_over_devices,
                                         excluded_tests=self._expected_failures if not self._expected_failures_update else set(),
                                         timeout=self._timeout)
        conformance.run()
        self.is_successful_run = conformance.postprocess_logs()

        if os.path.isfile(self._expected_failures_file):
            self.__update_expected_failures()

        final_report_name = f'report_{self._type.lower()}'
        merge_xml([parallel_report_dir], report_dir, final_report_name, self._type, True)

        logger.info(f"XML report was saved to {report_dir}")
        return os.path.join(report_dir, final_report_name + ".xml"), report_dir

    def __summarize(self, xml_report_path: os.path, report_dir: os.path):
        if self._type == constants.OP_CONFORMANCE:
            summary_root = ET.parse(xml_report_path).getroot()
            rel_weights_path = os.path.join(self._model_path,
                                            constants.REL_WEIGHTS_FILENAME.replace(constants.REL_WEIGHTS_REPLACE_STR,
                                                                                   self._special_mode))
            create_summary(summary_root, report_dir, [], "", "", True, True, rel_weights_path)
        else:
            create_api_summary([xml_report_path], report_dir, [], "", "")
        copytree(os.path.join(SCRIPT_DIR_PATH, "template"), os.path.join(report_dir, "template"))
        logger.info(f"Report was saved to {os.path.join(report_dir, 'report.html')}")

    def run(self, dump_models: bool):
        command = f'{constants.PIP_NAME} install -r {os.path.join(SCRIPT_DIR_PATH, "requirements.txt")}'
        process = Popen(command, shell=True)
        out, err = process.communicate()
        if err is None:
            if not out is None:
                for line in str(out).split('\n'):
                    logger.info(line)
        else:
            logger.error(err)
            logger.error("Impossible to install requirements!")
            exit(-1)
        logger.info(f"[ARGUMENTS] --device = {self._device}")
        logger.info(f"[ARGUMENTS] --type = {self._type}")
        logger.info(f"[ARGUMENTS] --working_dir = {self._working_dir}")
        logger.info(f"[ARGUMENTS] --ov_path = {self._ov_path}")
        logger.info(f"[ARGUMENTS] --ov_config_path = {self._ov_config_path}")
        logger.info(f"[ARGUMENTS] --workers = {self._workers}")
        logger.info(f"[ARGUMENTS] --gtest_filter = {self._gtest_filter}")
        logger.info(f"[ARGUMENTS] --models_path = {self._model_path}")
        logger.info(f"[ARGUMENTS] --dump_graph = {dump_models}")
        logger.info(f"[ARGUMENTS] --shape_mode = {self._special_mode}")
        logger.info(f"[ARGUMENTS] --entity = {self.__entity}")
        logger.info(f"[ARGUMENTS] --parallel_devices = {self._is_parallel_over_devices}")
        logger.info(f"[ARGUMENTS] --cache_path = {self._cache_path}")
        logger.info(f"[ARGUMENTS] --expected_failures = {self._expected_failures_file}")
        logger.info(f"[ARGUMENTS] --expected_failures_update = {self._expected_failures_update}")
        logger.info(f"[ARGUMENTS] --disable_rerun = {not self._is_rerun}")
        logger.info(f"[ARGUMENTS] --timeout = {self._timeout}")

        if self._type == constants.OP_CONFORMANCE:
            if file_utils.is_url(self._model_path):
                self._model_path = self.__download_models(self._model_path, self._working_dir)
            if self._model_path == NO_MODEL_CONSTANT or os.path.splitext(self._model_path)[1] == ".lst":
                with open(self._model_path, "r") as model_list_file:
                    model_dir = os.path.join(self._working_dir, "models")
                    if not os.path.isdir(model_dir):
                        os.mkdir(model_dir)
                    for model in model_list_file.readlines():
                        self.__download_models(model, model_dir)
                    self._model_path = model_dir
            if dump_models:
                self.__dump_subgraph()
            if not os.path.exists(self._model_path):
                logger.error(f"The model directory {self._model_path} does not exist!")
                exit(-1)
            if not os.path.exists(self._model_path):
                logger.error(f"Directory {self._model_path} does not exist")
                exit(-1)
        xml_report, report_dir = self.__run_conformance()
        self.__summarize(xml_report, report_dir)


if __name__ == "__main__":
    args = parse_arguments()
    conformance = Conformance(args.device, args.models_path,
                              args.ov_path, args.type,
                              args.workers, args.gtest_filter,
                              args.working_dir, args.ov_config_path,
                              args.special_mode, args.entity,
                              args.cache_path, args.parallel_devices,
                              args.expected_failures, args.expected_failures_update,
                              args.disable_rerun, args.timeout)
    conformance.run(args.dump_graph)
    if not conformance.is_successful_run:
        exit(-1)
