// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>


#define _AS_STR(x) #x
#define AS_STR(x) _AS_STR(x)


struct MemoryCounters {
    // memory size in kb
    int64_t virtual_size = -1;
    int64_t virtual_peak = -1;
    int64_t resident_size = -1;
    int64_t resident_peak = -1;

    int32_t thread_count = -1;

    static MemoryCounters sample();
};


#ifdef _WIN32

#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>

static PROCESS_MEMORY_COUNTERS getMemoryInfo() {
    static PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb))
        throw std::runtime_error("Can't get system memory values");
    return pmc;
}

static size_t getThreadsNum() {
    // first determine the id of the current process
    DWORD const id = GetCurrentProcessId();

    // then get a process list snapshot.
    HANDLE const snapshot = CreateToolhelp32Snapshot( TH32CS_SNAPALL, 0 );

    // initialize the process entry structure.
    PROCESSENTRY32 entry = { 0 };
    entry.dwSize = sizeof( entry );

    // get the first process info.
    BOOL  ret = true;
    ret = Process32First( snapshot, &entry );
    while( ret && entry.th32ProcessID != id ) {
        ret = Process32Next( snapshot, &entry );
    }
    CloseHandle( snapshot );
    return ret
        ?   entry.cntThreads
        :   -1;
}

MemoryCounters MemoryCounters::sample() {
    MemoryCounters out;
    auto meminfo = getMemoryInfo();
    out.virtual_size = meminfo.PagefileUsage / 1024;
    out.virtual_peak = meminfo.PeakPagefileUsage / 1024;
    out.resident_size = meminfo.WorkingSetSize / 1024;
    out.resident_peak = meminfo.PeakWorkingSetSize / 1024;
    out.thread_count = (int32_t) getThreadsNum();
    return out;
}

#else

MemoryCounters MemoryCounters::sample() {
    MemoryCounters out;
    std::ifstream file;
    file.open("/proc/self/status");
    std::string line;
    while (true) {
        if (!std::getline(file, line)) {
            break;
        }
        auto delim_pos = line.find(':');
        if (delim_pos == std::string::npos) {
            continue;
        }
        auto prefix = line.substr(0, delim_pos);
        auto value_start = line.find_first_not_of("\t ", delim_pos + 1);
        auto value_end = line.find_first_of("\t ", value_start);
        if (value_start == std::string::npos) {
            continue;
        }
        auto value = line.substr(value_start, value_end - value_start);
        long ivalue = std::atol(value.c_str());
        if (prefix == "VmSize") {
            out.virtual_size = ivalue;
        } else if (prefix == "VmPeak") {
            out.virtual_peak = ivalue;
        } else if (prefix == "VmRSS") {
            out.resident_size = ivalue;
        } else if (prefix == "VmHWM") {
            out.resident_peak = ivalue;
        } else if (prefix == "Threads") {
            out.thread_count = (int32_t) ivalue;
        }
    }
    return out;
}

#endif


inline std::string jsonescape(const std::string &str) {
    std::string newstr;
    newstr.reserve(str.size() * 2);
    for (auto chr: str) {
        if (chr == '\\' || chr == '/') {
            newstr.push_back('\\');
            newstr.push_back(chr);
        } else if (chr == '\b') {
            newstr.push_back('\\');
            newstr.push_back('b');
        } else if (chr == '\f') {
            newstr.push_back('\\');
            newstr.push_back('f');
        } else if (chr == '\n') {
            newstr.push_back('\\');
            newstr.push_back('n');
        } else if (chr == '\r') {
            newstr.push_back('\\');
            newstr.push_back('r');
        } else if (chr == '\t') {
            newstr.push_back('\\');
            newstr.push_back('t');
        } else {
            newstr.push_back(chr);
        }
    }
    return newstr;
}


// To be defined in the test
std::vector<std::string> test_samples();

std::vector<std::string> registered_samples_init() {
    auto samples = test_samples();
    samples.emplace_back("unload");
    return samples;
}

static std::vector<std::string> registered_samples = registered_samples_init();


struct TestContext {
    std::string model_path;
    std::string device;
    std::vector<std::pair<std::string, MemoryCounters>> samples;

    static TestContext from_args(int argc, char **argv) {
        std::string model_path;
        std::string device = "CPU";

        if (argc <= 1 || argc > 3) {
            std::cerr << "Usage: <executable> MODEL_PATH [DEVICE=CPU]" << std::endl;
            exit(-1);
        }
        if (argc > 1) {
            model_path = argv[1];
        }
        if (argc > 2) {
            device = argv[2];
        }

        return {model_path, device};
    }

    bool is_sample_registered(std::string &sample_name) {
        for (auto registered_sample_name: registered_samples) {
            if (registered_sample_name == sample_name) {
                return true;
            }
        }
        return false;
    }

    void sample(std::string sample_name) {
        if (!is_sample_registered(sample_name)) {
            // registered_samples must contain all possible sample names that this
            // test can yield. It is required to properly report crashed tests
            std::string error_msg = "sample \"" + sample_name + "\" is not defined in registered_samples";
            throw std::runtime_error(error_msg);
        }
        samples.emplace_back(std::move(sample_name), MemoryCounters::sample());
    }

    void report() {
        std::cout << "TEST_RESULTS: {"
        << "\"test\": \"" << AS_STR(TEST_NAME) << "\", "
        << "\"model_path\": \"" << jsonescape(model_path) << "\", "
        << "\"device\": \"" << device << "\", "
        << "\"samples\": {";
        for (auto &sample: samples) {
            std::cout << "\"" << sample.first << "\": {"
            << "\"vmsize\": " << sample.second.virtual_size << ", "
            << "\"vmpeak\": " << sample.second.virtual_peak << ", "
            << "\"vmrss\": " << sample.second.resident_size << ", "
            << "\"vmhwm\": " << sample.second.resident_peak << ", "
            << "\"threads\": " << sample.second.thread_count << "}";
            if (&sample != &samples.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "}}" << std::endl;
    }
};


// To be defined in the test
void do_test(TestContext &test);


int main(int argc, char **argv) {
    if (argc == 2 && std::string("--info") == argv[1]) {
        std::cout << "TEST_INFO: {\"samples\": [";
        for (auto &sample_name: registered_samples) {
            std::cout << "\"" << sample_name << "\"";
            if (&sample_name != &registered_samples.back()) {
                std::cout << ", ";
            }
        }
        std::cout << "]}" << std::endl;
        return 0;
    }

    TestContext test = TestContext::from_args(argc, argv);
    do_test(test);
    test.sample("unload");
    test.report();

    return 0;
}
