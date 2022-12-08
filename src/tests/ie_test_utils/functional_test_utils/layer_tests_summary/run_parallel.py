# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils import utils, process_handler
from argparse import ArgumentParser
from subprocess import Popen, STDOUT

import os
import sys
import threading

import datetime

if sys.version_info.major >= 3:
    import _thread as thread
else:
    import thread

logger = utils.get_logger('test_parallel_runner')
sigint_handler = process_handler.SigintHandler()

def parse_arguments():
    parser = ArgumentParser()
    exec_file_path_help = "Path to the test executable file"
    worker_num_help = "Worker number. Default value is cpu_count() - 1 "
    parser.add_argument("-e", "--exec_file", help=exec_file_path_help, type=str, required=True)
    parser.add_argument("-w", "--worker_num", help=worker_num_help, type=int, required=False, default=(os.cpu_count() - 1) if os.cpu_count() > 2 else 1)
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
    def __init__(self, exec_file_path: os.path, test_command_line: list, worker_num: int):
        self._exec_file_path = exec_file_path
        self._command = self.__init_basic_command_line_for_exec_file(test_command_line)
        self._worker_num = worker_num
        self._test_filters = list()

    def __init_basic_command_line_for_exec_file(self, test_command_line: list):
        command = f'{self._exec_file_path}'
        for argument in test_command_line:
            command += f" {argument}"
        return command

    def __parse_test_list_file(self):
        if not os.path.isfile(self._exec_file_path):
            logger.error(f"{self._exec_file_path} is not exist!")
            exit(-1)
        test_list_file_name = "test_list.lst"
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
        self._test_filters = ["".join(test_list[i::self._worker_num]).replace(" ", "") for i in range(self._worker_num)]
    
    def run_task(self, worker_idx: int):
        log_file_name = f"log_{worker_idx}.log"
        real_command = f'{self._command} --gtest_filter="{self._test_filters[worker_idx]}"'

        with open(log_file_name, 'w') as log_file:
            process = Popen(real_command, shell=True, stdout=log_file, stderr=log_file)
        try:
            pass
            # self.exit_code = sigint_handler.wait(process)
        except sigint_handler.ProcessWasInterrupted:
            thread.exit()
        return process
        # self.runtime_ms = int(1000 * (time.time() - begin))
        # self.last_execution_time = None if self.exit_code else self.runtime_ms

    def run(self): #, timeout):
        self.__parse_test_list_file()
        def start_daemon(func):
            t = threading.Thread(target=func)
            t.daemon = True
            t.start()
            return t

        try:
            # if timeout:
                # timeout.start()
            workers = [start_daemon(self.run_task(idx)) for idx in range(self._worker_num)]
            for worker in workers:
                worker.join()
        finally:
            pass
            # if timeout:
                # timeout.cancel()
    
    
    def get_command(self):
        return self._command

if __name__ == "__main__":
    l = get_test_command_line_args()
    args = parse_arguments()
    conformance = ParallelRunner(args.exec_file, l, args.worker_num)
    # command, test_list = conformance.get_test_list()
    t1 = datetime.datetime.now()
    conformance.run()
    t2 = datetime.datetime.now()

    t1_ = datetime.datetime.now()
    with open("/home/efode/repo/openvino/src/tests/ie_test_utils/functional_test_utils/layer_tests_summary/log.txt", 'w') as log:
        task = Popen(conformance.get_command(), shell=True, stdout=log, stderr=log)
        task.communicate()
    t2_ = datetime.datetime.now()

    t1__ = datetime.datetime.now()   
    with open("/home/efode/repo/openvino/src/tests/ie_test_utils/functional_test_utils/layer_tests_summary/log__.txt", 'w') as log:
        from run_conformance import Conformance
        conformance_ = Conformance("CPU", "/home/efode/repo/temp/edward_case", "/home/efode/repo/openvino", "OP", "/home/efode/repo/temp_temp")
        conformance_.start_pipeline(False)
    t2__ = datetime.datetime.now()
    print((t2 - t1).total_seconds())
    print((t2_ - t1_).total_seconds())
    print((t2__ - t1__).total_seconds())

