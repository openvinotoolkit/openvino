# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils import utils
from argparse import ArgumentParser
from subprocess import Popen, STDOUT, TimeoutExpired

import os
import sys
import threading
from hashlib import sha256

import csv

import datetime

if sys.version_info.major >= 3:
    import _thread as thread
else:
    import thread

# MAX_LENGTH = 32767 if "windows" in sys.platform or "win32" in sys.platform else os.sysconf('SC_ARG_MAX')
TEST_BATCH = 1
FILENAME_LENGTH = 255

TEST_TIMEOUT = 60

logger = utils.get_logger('test_parallel_runner')

def parse_arguments():
    parser = ArgumentParser()
    exec_file_path_help = "Path to the test executable file"
    worker_num_help = "Worker number. Default value is cpu_count() - 1 "
    working_dir_num_help="Working dir"
    parser.add_argument("-e", "--exec_file", help=exec_file_path_help, type=str, required=True)
    parser.add_argument("-j", "--worker_num", help=worker_num_help, type=int, required=False, default=(os.cpu_count() - 1) if os.cpu_count() > 2 else 1)
    parser.add_argument("-w", "--working_dir", help=working_dir_num_help, type=str, required=False, default=".")
    return parser.parse_args()

def get_test_command_line_args():
    command_line_args = list()
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--':
            command_line_args = sys.argv[i+1:]
            sys.argv = sys.argv[:i]
            break
    return command_line_args

class ParallelRunner:
    def __init__(self, exec_file_path: os.path, test_command_line: list, worker_num: int, working_dir: os.path):
        self._exec_file_path = exec_file_path
        self._command = self.__init_basic_command_line_for_exec_file(test_command_line)
        self._worker_num = worker_num
        self._test_filters = list()
        self._process = list()
        self._iterations = 0
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
                argument = utils.prepare_filelist(argument, "*.xml", logger)
            else:
                is_input_folder = False
            command += f" {argument}"
        return command

    def __parse_test_list_file(self):
        if not os.path.isfile(self._exec_file_path):
            logger.error(f"{self._exec_file_path} is not exist!")
            exit(-1)
        test_list_file_name = os.path.join(self._working_dir, "test_list.lst")
        command_to_get_test_list = self._command + f' --gtest_list_tests >> {test_list_file_name}'
        Popen(command_to_get_test_list, shell=True).communicate()

        test_list = list()
        with open(test_list_file_name) as test_list_file:
            for test_name in test_list_file.read().split('\n'):
                pos = test_name.find('#')
                if pos > 0:
                    test_list.append(f"*{test_name[0:pos].replace(' ', '')}*:")
            test_list_file.close()
        os.remove(test_list_file_name)
        filters = [test_list[i::self._worker_num] for i in range(self._worker_num)]
        # for filter in filters:
        self._iterations = int(len(filters[0]) / TEST_BATCH) + 1
        self._test_filters = [["".join(filter[TEST_BATCH*i:TEST_BATCH*(i+1):]).replace(' ', '') for i in range(self._iterations)] for filter in filters]
        pass

        # self._test_filters = ["".join(test_list[i::self._worker_num]).replace(" ", "") for i in range(self._worker_num)]

    def run_task(self, real_command:str, idx:int, p = -1):
        # log_file_name = os.path.join(self._working_dir, f"log_{idx}.log")
        log_file_name = os.path.join(self._working_dir, f"log_{idx}.log")
        # real_command = f'{self._command} --gtest_filter="{self._test_filters[worker_idx][job_id]}"'

        with open(log_file_name, 'w') as log_file:
            try:
                if p == -1:
                    self._process.append(Popen(real_command, shell=True, stdout=log_file, stderr=log_file))
                else:
                    self._process[p] = Popen(real_command, shell=True, stdout=log_file, stderr=log_file)
            except:
                thread.exit()

    def generate_command(self):
        cmd_list = list()
        for it in range(self._iterations):
            for idx in range(self._worker_num):
                if self._test_filters[idx][it] == "":
                    continue
                # print({self._test_filters[idx][it]})
                cmd_list.append(f'{self._command} --gtest_filter="{self._test_filters[idx][it]}"')
        return cmd_list



    def run(self):
        self.__parse_test_list_file()
        def create_and_start_thread(func):
            t = threading.Thread(target=func)
            t.daemon = True
            t.start()
            return t

        t1 = datetime.datetime.now()
        cmds = self.generate_command()
        
        workers = [create_and_start_thread(self.run_task(cmds[index], index)) for index in range(self._worker_num)]
        timer = list()
        t1___ = datetime.datetime.now()
        for worker in workers:
            worker.join()
            timer.append(datetime.datetime.now())
            
        idx = self._worker_num
        while idx < len(cmds):
            # finally:
                # for i in range(len(self._process)):
                #     try:
                #         self._process[i].wait(timeout=1)
                #         worker[i] = create_and_start_thread(self.run_task(i, it))
                #     except TimeoutExpired:
                #         assert self._process[i].returncode is None
            for p in range(len(self._process)):
                try:
                    if float((datetime.datetime.now() - timer[p]).total_seconds()) > TEST_TIMEOUT:
                        logger.warning("Exit")
                        workers[p].exit()
                    self._process[p].wait(timeout=0)
                    # print(cmds[idx])
                    workers[p] = create_and_start_thread(self.run_task(cmds[idx], idx, p))
                    workers[p].join()

                    # print(f"{p} {(datetime.datetime.now() - ts_[p]).total_seconds()}")
                    timer[p] = datetime.datetime.now()
                    idx += 1
                    if idx == len(cmds):
                        break
                except TimeoutExpired:
                    # print("Not finished")
                    continue
        print(f"FINISh {(datetime.datetime.now() - t1___).total_seconds()}")
        while len(self._process) > 0:
            l = len(self._process)
            # print(l)
            for p in range(l):
                try:
                    if float((datetime.datetime.now() - timer[p]).total_seconds()) > TEST_TIMEOUT:
                        logger.warning("Exit")
                        self._process[p].kill()
                    self._process[p].wait(timeout=0)
                    self._process.pop(p)
                    break
                except TimeoutExpired:
                    # print("Not finished")
                    continue
                
            

        # for it in range(self._iterations):
        #     try:
        #         workers = [create_and_start_thread(self.run_task(idx, it)) for idx in range(self._worker_num)]
        #         for worker in workers:
        #             worker.join()
        #     finally:
        #         # for i in range(len(self._process)):
        #         #     try:
        #         #         self._process[i].wait(timeout=1)
        #         #         worker[i] = create_and_start_thread(self.run_task(i, it))
        #         #     except TimeoutExpired:
        #         #         assert self._process[i].returncode is None
        #         for p in self._process:
        #             p.wait()
        t2 = datetime.datetime.now()
        print((t2 - t1).total_seconds())
        pass
    
    
    def get_command(self):
        return self._command

    def postprocess_logs(self):
        logs_dir = os.path.join(self._working_dir, "logs")
        if not os.path.exists(logs_dir):
            os.mkdir(logs_dir)
        test_status = {'passed': "[       OK ]", 'failed': "[  FAILED  ]", 'hanged': "Test finished by timeout", 'crashed': "Crash happens", 'skipped': "[  SKIPPED ]"}
        for test_st, string in test_status.items():
            if not os.path.exists(os.path.join(logs_dir, test_st)):
                os.mkdir(os.path.join(logs_dir, test_st))

        RUN = "[ RUN      ]"
        hash_map = [["Dir", "Hash", "TestName"]]
        for it in range(self._iterations):
            for idx in range(self._worker_num):
                log_filename = os.path.join(self._working_dir, f"log_{idx}_{it}.log")
                with open(log_filename, "r") as log_file:
                    test_name = None
                    test_log = list()
                    dir = None
                    for line in log_file.readlines():
                        if RUN in line and test_name is None:
                            test_name = line[line.find(RUN) + len(RUN) + 1:-1:]
                            test_log.append(line)
                            continue
                        if dir is None:
                            for test_st, string in test_status.items():
                                if string in line:
                                    dir = test_st
                        if test_name is not None:
                            test_log.append(line)
                            if test_name in line:
                                test_log_filename = os.path.join(logs_dir, dir, f'{test_name}.txt'.replace('/', '_'))
                                if len(test_log_filename) > FILENAME_LENGTH:
                                    hash_str = str(sha256(test_name.encode('utf-8')).hexdigest())
                                    test_log_filename = os.path.join(logs_dir, dir, f'{hash_str}.txt')
                                    hash_map.append([dir, hash_str, test_name])
                                with open(test_log_filename, "w") as log:
                                    log.writelines(test_log)
                                    log.close()
                                    test_name = None
                                    test_log = list()
                                    dir = None
        with open(os.path.join(logs_dir, "hash_table.csv"), "w") as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            for row in hash_map:
                csv_writer.writerow(row)

                    



if __name__ == "__main__":
    l = get_test_command_line_args()
    args = parse_arguments()
    conformance = ParallelRunner(args.exec_file, l, args.worker_num, args.working_dir)
    # command, test_list = conformance.get_test_list()
    t1 = datetime.datetime.now()
    print(t1)
    conformance.run()
    # conformance.postprocess_logs()
    t2 = datetime.datetime.now()

    t1_ = datetime.datetime.now()
    # with open("/home/efode/repo/openvino/src/tests/ie_test_utils/functional_test_utils/layer_tests_summary/log.txt", 'w') as log:
    #     task = Popen(conformance.get_command(), shell=True, stdout=log, stderr=log)
    #     task.communicate()
    t2_ = datetime.datetime.now()

    t1__ = datetime.datetime.now()   
    with open("/home/efode/repo/openvino/src/tests/ie_test_utils/functional_test_utils/layer_tests_summary/log__.txt", 'w') as log:
        from run_conformance import Conformance
        # def __init__(self, device:str, model_path:os.path, ov_path:os.path, type:str, working_dir:os.path):
        conformance_ = Conformance("CPU", "/home/efode/repo/temp", "/home/efode/repo/openvino", "OP", "/home/efode/repo/temp_temp")
        conformance_.start_pipeline(False)
    t2__ = datetime.datetime.now()
    print((t2 - t1).total_seconds())
    print((t2_ - t1_).total_seconds())
    print((t2__ - t1__).total_seconds())

