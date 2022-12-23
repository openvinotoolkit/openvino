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

MAX_LENGHT = 4096


TEST_STATUS = {'passed': "[       OK ]", 'failed': "[  FAILED  ]", 'hanged': "Test finished by timeout", 'crashed': "Crash happens", 'skipped': "[  SKIPPED ]", "interapted": "interapted"}
RUN = "[ RUN      ]"

GTEST_RESTRICTED_SYMBOLS = [':', '']

logger = utils.get_logger('test_parallel_runner')

def parse_arguments():
    parser = ArgumentParser()
    exec_file_path_help = "Path to the test executable file"
    cache_path_help = "Path to the cache file with test_name list sorted by execution time. .lst file!"
    worker_num_help = "Worker number. Default value is `cpu_count-1` "
    working_dir_num_help = "Working dir"
    test_batch_help = "Test counter in 1 process"
    process_timeout_help = "Process timeout in s"

    parser.add_argument("-e", "--exec_file", help=exec_file_path_help, type=str, required=True)
    parser.add_argument("-c", "--cache_path", help=cache_path_help, type=str, required=False, default="")
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

    
class TestStructure:
    _name = ""
    _time = 0

    def __init__(self, name, time):
        self._name = name
        self._time = int(time)

class TaskManager:
    process_timeout = -1

    def __init__(self, command_list:list, working_dir: os.path):
        self._command_list = command_list
        self._process_list = list()
        self._workers = list()
        self._timers = list()
        self._idx = 0
        self._log_filename = os.path.join(working_dir, f"log_{LOG_NAME_REPLACE_STR}.log")

    def __create_thread(self, func):
        thread = threading.Thread(target=func)
        thread.daemon = True
        thread.start()
        return thread
    
    def init_worker(self):
        if len(self._command_list) <= self._idx:
            logger.warning(f"Skip worker initialiazation. Command list lenght <= worker index")
            return
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx))
        with open(log_file_name, "w") as log_file:
            worker = self.__create_thread(
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
                        self._process_list[pid].kill()
                    self._process_list[pid].wait(timeout=0)
                    logger.info(f"Process {pid} {float((datetime.datetime.now() - self._timers[pid]).total_seconds())}")
                    return pid
                except TimeoutExpired:
                    continue

    def __update_process(self, pid:int, log_file):
        self._process_list[pid] = Popen(self._command_list[self._idx], shell=True, stdout=log_file, stderr=log_file)

    def update_worker(self):
        if self._idx >= len(self._command_list):
            return False
        pid = self.__find_free_process()
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx))
        with open(log_file_name, "w") as log_file:
            self._workers[pid] = self.__create_thread(self.__update_process(pid, log_file))
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
                    logger.info(f"Process {pid} {float((datetime.datetime.now() - self._timers[pid]).total_seconds())}")
                    self._process_list.pop(pid)
                    break
                except TimeoutExpired:
                    continue

class TestParallelRunner:
    def __init__(self, exec_file_path: os.path, test_command_line: list, worker_num: int, working_dir: os.path, test_batch:int, cache_path: os.path):
        self._exec_file_path = exec_file_path
        self._command = self.__init_basic_command_line_for_exec_file(test_command_line)
        self._worker_num = worker_num
        self._test_batch = test_batch
        self._working_dir = working_dir
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        if cache_path == "":
            cache_path = os.path.join(self._working_dir, "test_cache.lst")
        self._cache_path = os.path.join(cache_path)

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
                for _ in argument.split(','):
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


    def __sort_test_list(self, test_list:list):
        test_list_size = len(test_list)
        while test_list_size % self._worker_num != 0:
            test_list.append("")
            test_list_size += 1

        index = 0
        pos_start = 0
        index_ = int(len(test_list) / self._worker_num)
        while pos_start < test_list_size / 2:
            if index % 2 == 1:
                pos_start = index * self._worker_num
                pos_end_ = index_ * self._worker_num
                index += 1
                index_ -= 1
                pos_end = index * self._worker_num
                pos_start_ = index_ * self._worker_num
                
                buf = test_list[pos_start_:pos_end_]
                buf.reverse()
                test_list[pos_start_:pos_end_] = test_list[pos_start:pos_end]
                test_list[pos_start:pos_end] = buf
            else:
                index += 1
                index_ -= 1
        for idx in range(len(test_list)):
            col_num = idx % self._worker_num
            row_num = int(idx / self._worker_num)
            if row_num < col_num:
                new_idx = self._worker_num * col_num + row_num 
                buf = test_list[new_idx]
                test_list[new_idx] = test_list[idx]
                test_list[idx] = buf
        return test_list

    def __get_test_list_by_runtime(self):
        test_list_file_name = os.path.join(self._working_dir, "test_list.lst")
        command_to_get_test_list = self._command + f' --gtest_list_tests >> {test_list_file_name}'
        logger.info(f"Get test list using command: {command_to_get_test_list}")
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
        return test_list


    def __generate_gtest_filters(self, test_list: list):
        res_test_filters = list()
        def_length = len(self._command) + len(" --gtest_filter=")

        while len(test_list) > 0:
            test_times = []
            is_not_full = True
            worker_test_filters = list()

            for _ in range(self._worker_num):
                if len(test_list) == 0:
                    break
                worker_test_filters.append(test_list[0]._name)
                test_times.append(test_list[0]._time)
                test_list.pop(0)
            while is_not_full and len(test_list) > 0:
                for i in range(self._worker_num):
                    if i >= len(test_list):
                        break
                    if i == 0:
                        continue
                    while test_times[0] > test_times[i] + test_list[len(test_list) - 1]._time:
                        final_pos = len(test_list) - 1
                        if len(worker_test_filters[i]) + def_length + len(test_list[final_pos]._name) < MAX_LENGHT:
                            worker_test_filters[i] += test_list[final_pos]._name
                            test_times[i] += test_list[final_pos]._time
                            test_list.pop(final_pos)
                        else:
                            is_not_full = False
                            break
                if is_not_full and len(test_list) > 0:
                    worker_test_filters[0] += test_list[0]._name
                    test_times[0] += test_list[0]._time
                    test_list.pop(0)
            for filter in worker_test_filters:
                res_test_filters.append(filter)
            is_not_full = True
        return res_test_filters
            
    def __get_test_list(self):
        if not os.path.isfile(self._exec_file_path):
            logger.error(f"{self._exec_file_path} is not exist!")
            exit(-1)

        test_list_runtime = self.__get_test_list_by_runtime()
        test_list_cache = list()

        if os.path.isfile(self._cache_path):
            logger.info(f"Get test list from cache file: {self._cache_path}")
            with open(self._cache_path, "r") as cache_file:
                for line in cache_file.readlines():
                    pos = line.find(":")
                    time = line[:pos]
                    test_name = line[pos+1:]
                    test_list_cache.append(TestStructure(test_name.replace("\n", ""), time))
                    
        if len(test_list_cache) >= len(test_list_runtime):
            logger.info("Test list in taken from cache")
            logger.info(f"Total test counter is {len(test_list_cache)}")
            return self.__generate_gtest_filters(test_list_cache)
        else:
            logger.info("Test list in taken from runtime")
            logger.info(f"Total test counter is {len(test_list_runtime)}")
            test_list_runtime = self.__sort_test_list(test_list_runtime)
            return test_list_runtime


    def __generate_command_list(self):
        test_filters = self.__get_test_list()
        cmd_list = [f'{self._command} --gtest_filter={filter}' for filter in test_filters]
        return cmd_list
        


    def run(self):
        if TaskManager.process_timeout == -1:
            TaskManager.process_timeout = DEFAUALT_PROCESS_TIMEOUT
        logger.info(f"Run test parallel is started. Worker num is {self._worker_num}")
        t_start = datetime.datetime.now()
        task_manger = TaskManager(self.__generate_command_list(), self._working_dir)
        a = datetime.datetime.now()
        for index in range(self._worker_num):
            task_manger.init_worker()
        b = datetime.datetime.now()
        while task_manger.update_worker():
            pass
        c = datetime.datetime.now()
        task_manger.compelete_all_processes()
        t_end = datetime.datetime.now()
        
        logger.info(f"Total time is {(a - t_start).total_seconds()}s")
        logger.info(f"Total time is {(b - a).total_seconds()}s")
        logger.info(f"Total time is {(c - b).total_seconds()}s")
        logger.info(f"Total time is {(t_end - c).total_seconds()}s")
        logger.info(f"Run test parallel is finished successfully. Total time is {(t_end - t_start).total_seconds()}s")


    def postprocess_logs(self):
        test_results = dict()
        logger.info(f"Log analize is started")
        def __save_log(logs_dir, dir, test_name):
            test_log_filename = os.path.join(logs_dir, dir, f"{test_name}.txt".replace('/', '_'))
            if os.path.isfile(test_log_filename):
                logger.warning(f"Log file {test_log_filename} is exist!")
                return False
            if len(test_log_filename) > FILENAME_LENGTH:
                hash_str = str(sha256(test_name.encode('utf-8')).hexdigest())
                test_log_filename = os.path.join(logs_dir, dir, f'{hash_str}.txt')
                if os.path.isfile(test_log_filename):
                    logger.warning(f"Log file {test_log_filename} is exist!")
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
        test_times = list()
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
                            test_times.append((1000000, test_name))
                            if __save_log(logs_dir, dir, test_name):
                                test_cnt_log += 1
                                if dir in test_results.keys():
                                    test_results[dir] += 1
                                else:
                                    test_results[dir] = 1
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
                            time = line[line.rfind("(") + 1:line.rfind("ms)")-1]
                            try:
                                test_times.append((int(time), test_name))
                            except:
                                continue
                            if __save_log(logs_dir, dir, test_name):
                                test_cnt_log += 1
                                if dir in test_results.keys():
                                    test_results[dir] += 1
                                else:
                                    test_results[dir] = 1
                                test_name = None
                                test_log = list()
                                dir = None
            # logger.info(f"Number of tests in {log}: {test_cnt_log}")
            os.remove(log)
        test_times.sort(reverse=True)
        head, tail = os.path.split(self._cache_path)
        if not os.path.isdir(head) and head != "":
            logger.error(f"Impossible to create cche file! The dir {head} does not exist!")
            exit(-1)
        with open(self._cache_path, "w") as cache_file:
            cache_file.writelines([f"{time}:\"" + test_name + "\":\n" for time, test_name in test_times])
            cache_file.close()
        logger.info(f"Cache file: {self._cache_path} was created")
        hash_table_path = os.path.join(logs_dir, "hash_table.csv")
        with open(hash_table_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            for row in hash_map:
                csv_writer.writerow(row)
            logger.info(f"Hash file: {hash_table_path} was created")

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
    conformance = TestParallelRunner(args.exec_file, exec_file_args, args.worker_num, args.working_dir, args.test_batch, args.cache_path)
    conformance.run()
    if not conformance.postprocess_logs():
        logger.error("Run is not successful")