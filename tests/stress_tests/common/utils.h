// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <sys/unistd.h>
#include <sys/wait.h>

#ifdef _WIN32
#define OS_SEP std::string("\\")
#else
#define OS_SEP std::string("/")
#endif


#define log_info(str) std::cout << "[ INFO ] " << str << std::endl
#define log_warn(str) std::cout << "[ WARNING ] " << str << std::endl
#define log_err(str) std::cout << "[ ERROR ] " << str << std::endl
#define log_debug(str) std::cout << "[ DEBUG ] " << str << std::endl

std::string OS_PATH_JOIN(std::initializer_list<std::string> list);

std::string fileNameNoExt(const std::string &filepath);

#define getVmValues(vmsize, vmpeak, vmrss, vmhwm) vmsize = (long) getVmSizeInKB();    \
                                                  vmpeak = (long) getVmPeakInKB();    \
                                                  vmrss = (long) getVmRSSInKB();      \
                                                  vmhwm = (long) getVmHWMInKB();

size_t getVmSizeInKB();
size_t getVmPeakInKB();
size_t getVmRSSInKB();
size_t getVmHWMInKB();

template<typename Function, typename ... Args>
int run_in_processes(const int &numprocesses, Function const &function, Args ... args) {
    std::vector<pid_t> child_pids(numprocesses);

    for (int i = 0; i < numprocesses; i++) {
        child_pids[i] = fork();
        if (child_pids[i] == 0) {
            function(args...);
            exit(EXIT_SUCCESS);
        }
    }

    int status = 0;
    for (int i = 0; i < numprocesses; i++) {
        int _status = 0;
        waitpid(child_pids[i], &_status, WSTOPPED);
        if (_status) {
            log_err("Process run # " << i << " failed with exitcode " << _status);
            status = _status;
        }
    }
    return status;
}

template<typename Function, typename ... Args>
inline void run_in_threads(const int &numthreads, Function const &function, Args ... args) {
    std::vector<std::thread> v(numthreads);
    for (int thr_i = 0; thr_i < numthreads; thr_i++) {
        v[thr_i] = std::thread(function, args...);
    }

    for (int thr_i = 0; thr_i < numthreads; thr_i++) {
        v[thr_i].join();
    }
    v.clear();
}

void auto_expand_env_vars(std::string &input);
std::string expand_env_vars(const std::string &input);
