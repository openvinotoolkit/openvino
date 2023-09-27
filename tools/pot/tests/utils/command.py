# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import os.path
import sys
import signal
import time
import threading


class Command:
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.exec_time = -1
        self.output = []  # store output here
        self.kwargs = {}
        self.timeout = False

        # set system/version dependent 'start_new_session' analogs
        if sys.platform == 'win32':
            self.kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        elif sys.version_info < (3, 8):  # assume posix
            self.kwargs.update(preexec_fn=os.setsid)
        else:  # Python 3.8+ and Unix
            self.kwargs.update(start_new_session=True)

    def kill_process_tree(self, pid):
        try:
            if sys.platform != 'win32':
                os.killpg(pid, signal.SIGKILL)
            else:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
        except OSError as err:
            print(err)

    def run(self, timeout=3600):

        def target():
            start_time = time.time()
            self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                                            bufsize=1, **self.kwargs)
            self.timeout = False

            self.output = []
            for line in self.process.stdout:
                line = line.decode('utf-8')
                self.output.append(line)
                sys.stdout.write(line)

            sys.stdout.flush()
            self.process.stdout.close()

            self.process.wait()
            self.exec_time = time.time() - start_time

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            try:
                print('Error: process taking too long to complete--terminating' + ', [ ' + self.cmd + ' ]')
                self.kill_process_tree(self.process.pid)
                self.exec_time = timeout
                self.timeout = True
                thread.join()
            except OSError as e:
                print(self.process.pid, 'Exception when try to kill task by PID, ' + e.strerror)
                raise
        returncode = self.process.wait()
        print('Process returncode = ' + str(returncode))
        return returncode

    def get_execution_time(self):
        return self.exec_time
