# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils import utils
from argparse import ArgumentParser
from subprocess import Popen, STDOUT, TimeoutExpired

import os
import sys
import threading
from hashlib import sha256
from pathlib import Path
from shutil import rmtree

import csv

import datetime

if sys.version_info.major >= 3:
    import _thread as thread
else:
    import thread

FILENAME_LENGTH = 255

LOG_NAME_REPLACE_STR = "##NAME##"

DEFAUALT_PROCESS_TIMEOUT = 3600


TEST_STATUS = {'passed': "[       OK ]", 'failed': "[  FAILED  ]", 'hanged': "Test finished by timeout", 'crashed': "Crash happens", 'skipped': "[  SKIPPED ]"}
RUN = "[ RUN      ]"

GTEST_RESTRICTED_SYMBOLS = [':', '']

logger = utils.get_logger('test_parallel_runner')

def parse_arguments():
    parser = ArgumentParser()
    exec_file_path_help = "Path to the test executable file"
    worker_num_help = "Worker number. Default value is `cpu_count-1` "
    working_dir_num_help = "Working dir"
    test_batch_help = "Test counter in 1 process"
    process_timeout_help = "Process timeout in s"

    parser.add_argument("-e", "--exec_file", help=exec_file_path_help, type=str, required=True)
    parser.add_argument("-j", "--worker_num", help=worker_num_help, type=int, required=False, default=(os.cpu_count() - 1) if os.cpu_count() > 2 else 1)
    parser.add_argument("-w", "--working_dir", help=working_dir_num_help, type=str, required=False, default=".")
    parser.add_argument("-b", "--test_batch", help=test_batch_help, type=int, required=False, default=1)
    parser.add_argument("-t", "--process_timeout", help=process_timeout_help, type=int, required=False, default=DEFAUALT_PROCESS_TIMEOUT)
    return parser.parse_args()

def get_test_command_line_args():
    command_line_args = list()
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--':
            command_line_args = sys.argv[i+1:]
            sys.argv = sys.argv[:i]
            break
    return command_line_args

class TaskManager:
    process_timeout = -1

    def __init__(self, command_list:list, working_dir: os.path):
        self._command_list = command_list
        self._process_list = list()
        self._workers = list()
        self._timers = list()
        self._idx = 0
        self._log_filename = os.path.join(working_dir, f"log_{LOG_NAME_REPLACE_STR}.log")


    def __create_and_start_thread(self, func):
        thread = threading.Thread(target=func)
        thread.daemon = True
        thread.start()
        return thread
    
    def init_worker(self):
        # logger.info(f"Initialize worker {self._idx}")
        if len(self._command_list) <= self._idx:
            logger.warning(f"Skip worker initialiazation. Command list lenght <= worker index")
            return
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx))
        with open(log_file_name, "w") as log_file:
            worker = self.__create_and_start_thread(
                self._process_list.append(Popen(self._command_list[self._idx], shell=True, stdout=log_file, stderr=log_file)))
            self._workers.append(worker)
            worker.join()
            self._timers.append(datetime.datetime.now())
        self._idx += 1
    
    def __find_free_process(self):
        while True:
            for pid in range(len(self._process_list)):
                try:
                    if float((datetime.datetime.now() - self._timers[pid]).total_seconds()) > self.process_timeout:
                        logger.warning(f"Process {pid} exceed time limetattion per process")
                        self._workers[pid].exit()
                    self._process_list[pid].wait(timeout=0)
                    return pid
                except TimeoutExpired:
                    continue

    def __update_process(self, pid:int, log_file):
        self._process_list[pid] = Popen(self._command_list[self._idx], shell=True, stdout=log_file, stderr=log_file)

    def update_worker(self):
        if self._idx >= len(self._command_list):
            return False
        pid = self.__find_free_process()
        # logger.info(f"Update worker {pid}")
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx))
        with open(log_file_name, "w") as log_file:
            self._workers[pid] = self.__create_and_start_thread(self.__update_process(pid, log_file))
            self._workers[pid].join()
            self._timers[pid] = datetime.datetime.now()
        self._idx += 1
        return True
    
    def compelete_all_processes(self):        
        while len(self._process_list) > 0:
            for pid in range(len(self._process_list)):
                try:
                    if float((datetime.datetime.now() - self._timers[pid]).total_seconds()) > self.process_timeout:
                        logger.warning(f"Process {pid} exceed time limetattion per process")
                        self._process_list[pid].kill()
                    self._process_list[pid].wait(timeout=0)
                    self._process_list.pop(pid)
                    break
                except TimeoutExpired:
                    continue

class TestParallelRunner:
    def __init__(self, exec_file_path: os.path, test_command_line: list, worker_num: int, working_dir: os.path, test_batch:int):
        self._exec_file_path = exec_file_path
        self._command = self.__init_basic_command_line_for_exec_file(test_command_line)
        self._worker_num = worker_num
        self._test_batch = test_batch
        self._run_num_per_executor = 0
        self._working_dir = working_dir
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)


    def __init_basic_command_line_for_exec_file(self, test_command_line: list):
        command = f'{self._exec_file_path}'
        is_input_folder = False
        for argument in test_command_line:
            if "--input_folders" in argument:
                is_input_folder = True
                command += f" --input_folders"
                argument = argument[argument.find("=")+1:]
            if is_input_folder and argument[0] != "-":
                buf = ""
                for in_folder in argument.split(','):
                    buf = utils.prepare_filelist(argument.replace('"', ''), "*.xml", logger)
                    buf += ","
                argument = buf 
            else:
                is_input_folder = False
            command += f" {argument}"
        return command

    @staticmethod
    def __replace_restricted_symbols(input_string:str):
        restricted_symbols = "!@$%^&-+`~:;\",<>?"
        for symbol in restricted_symbols:
            input_string = input_string.replace(symbol, '*')
        return input_string


    def __parse_test_list_file(self):
        if not os.path.isfile(self._exec_file_path):
            logger.error(f"{self._exec_file_path} is not exist!")
            exit(-1)
        test_list_file_name = os.path.join(self._working_dir, "test_list.lst")
        command_to_get_test_list = self._command + f' --gtest_list_tests >> {test_list_file_name}'
        Popen(command_to_get_test_list, shell=True).communicate()

        test_list = list()
        with open(test_list_file_name) as test_list_file:
            test_suite = ""
            for test_name in test_list_file.read().split('\n'):
                pos = test_name.find('#')
                if pos > 0:
                    real_test_name = test_suite + test_name[2:pos-2]
                    test_list.append(f'"{self.__replace_restricted_symbols(real_test_name)}":')
                else:
                    test_suite = test_name
            test_list_file.close()
        os.remove(test_list_file_name)
        logger.info(f"Total command list lenght is {len(test_list)}")
        return test_list
        
    
    def __prepare_filters(self):
        test_list = self.__parse_test_list_file()
        filters = [test_list[i::self._worker_num] for i in range(self._worker_num)]
        self._run_num_per_executor = int(len(filters[0]) / self._test_batch) + 1
        test_filters = [[filter[self._test_batch * i:self._test_batch * (i + 1):] for i in range(self._run_num_per_executor)] for filter in filters]
        return [["".join(filter) for filter in worker_filters] for worker_filters in test_filters]

    def __generate_command_list(self):
        test_filters = self.__prepare_filters()
        cmd_list = list()
        for worker_filter in test_filters:
            for filter in worker_filter:
                if filter == "":
                    continue
                cmd_list.append(f'{self._command} --gtest_filter={filter}')
        return cmd_list


    def run(self):
        if TaskManager.process_timeout == -1:
            TaskManager.process_timeout = DEFAUALT_PROCESS_TIMEOUT
        logger.info(f"Run test parallel is started")
        t_start = datetime.datetime.now()
        task_manger = TaskManager(self.__generate_command_list(), self._working_dir)
        for index in range(self._worker_num):
            task_manger.init_worker()
        while task_manger.update_worker():
            pass
        task_manger.compelete_all_processes()
        t_end = datetime.datetime.now()
        logger.info(f"Run test parallel is finished successfully. Total time is {(t_end - t_start).total_seconds()}s")


    def postprocess_logs(self):
        test_results = dict()
        logger.info(f"Log analize is started")
        def __save_log(logs_dir, dir, test_name):
            test_log_filename = os.path.join(logs_dir, dir, f"{test_name}.txt".replace('/', '_'))
            if os.path.isfile(test_log_filename):
                log.warning(f"Log file {test_log_filename} is exist!")
                return False
            if len(test_log_filename) > FILENAME_LENGTH:
                hash_str = str(sha256(test_name.encode('utf-8')).hexdigest())
                test_log_filename = os.path.join(logs_dir, dir, f'{hash_str}.txt')
                if os.path.isfile(test_log_filename):
                    log.warning(f"Log file {test_log_filename} is exist!")
                    return False
                hash_map.append([dir, hash_str, test_name])
            with open(test_log_filename, "w") as log:
                log.writelines(test_log)
                log.close()
            return True

        logs_dir = os.path.join(self._working_dir, "logs")
        if os.path.exists(logs_dir):
            logger.info(f"Logs directory {logs_dir} is cleaned up")
            rmtree(logs_dir)
        os.mkdir(logs_dir)
        for test_st, string in TEST_STATUS.items():
            if not os.path.exists(os.path.join(logs_dir, test_st)):
                os.mkdir(os.path.join(logs_dir, test_st))
        hash_map = [["Dir", "Hash", "Test Name"]]
        for log in Path(self._working_dir).rglob("log_*.log"):
            test_cnt_log = 0
            log_filename = os.path.join(self._working_dir, log)
            with open(log_filename, "r") as log_file:
                test_name = None
                test_log = list()
                dir = None
                for line in log_file.readlines():
                    if RUN in line:
                        if test_name is not None:
                            dir = "interapted"
                            if __save_log(logs_dir, dir, test_name):
                                test_cnt_log += 1
                                if dir in test_results.keys():
                                    test_results[dir] += 1
                                else:
                                    test_results[dir] = 1
                            test_name = None
                            test_log = list()
                            dir = None
                        test_name = line[line.find(RUN) + len(RUN) + 1:-1:]
                        test_log.append(line)
                        continue
                    if dir is None:
                        for test_st, string in TEST_STATUS.items():
                            if string in line:
                                dir = test_st
                    if test_name is not None:
                        test_log.append(line)
                        if test_name in line:
                            if __save_log(logs_dir, dir, test_name):
                                test_cnt_log += 1
                                if dir in test_results.keys():
                                    test_results[dir] += 1
                                else:
                                    test_results[dir] = 1
                            test_name = None
                            test_log = list()
                            dir = None
            logger.info(f"Number of tests in {log}: {test_cnt_log}")
            os.remove(log) 
        with open(os.path.join(logs_dir, "hash_table.csv"), "w") as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            for row in hash_map:
                csv_writer.writerow(row)
        logger.info(f"Log analize is succesfully finished")
        is_successfull_run = True
        test_cnt = 0
        for test_st, test_res in test_results.items():
            logger.info(f"{test_st} test counter is: {test_res}")
            test_cnt += test_res
            if (test_st != "passed" or test_st != "skipped") and test_res > 0:
                is_successfull_run = False
        logger.info(f"Total test count is {test_cnt}. All logs is saved to {logs_dir}")
        return is_successfull_run

if __name__ == "__main__":
    exec_file_args = get_test_command_line_args()
    args = parse_arguments()
    logger.info(f"[ARGUMENTS] --exec_file={args.exec_file}")
    logger.info(f"[ARGUMENTS] --worker_num={args.worker_num}")
    logger.info(f"[ARGUMENTS] --working_dir={args.working_dir}")
    logger.info(f"[ARGUMENTS] --test_batch={args.test_batch}")
    logger.info(f"[ARGUMENTS] --process_timeout={args.process_timeout}")
    logger.info(f"[ARGUMENTS] Executable file arguments = {exec_file_args}")
    TaskManager.process_timeout = args.process_timeout
    conformance = TestParallelRunner(args.exec_file, exec_file_args, args.worker_num, args.working_dir, args.test_batch)
    conformance.run()
    if not conformance.postprocess_logs():
        logger.error("Run is not successful")
