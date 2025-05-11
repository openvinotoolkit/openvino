// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <functional>

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
size_t getThreadsNum();

int run_in_processes(const int &numprocesses, const std::function<void()> &function);

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
