// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "system_memory_sampling.hpp"
#include "gpu_memory_sampling.hpp"


#define _AS_STR(x) #x
#define AS_STR(x) _AS_STR(x)


struct MemoryCounters {
    // memory size in kb
    int64_t virtual_size = -1;
    int64_t virtual_peak = -1;
    int64_t resident_size = -1;
    int64_t resident_peak = -1;

    int32_t thread_count = -1;

    int64_t gpu_local_used = -1;
    int64_t gpu_local_total = -1;
    int64_t gpu_nonlocal_used = -1;
    int64_t gpu_nonlocal_total = -1;
};


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
    bool gpu_ready = false;

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

        bool gpu_ready = false;
        if (device.find("GPU") != std::string::npos) {
            // GPU is used -> initialize GPU sampling
            gpu_ready = initGpuSampling() == InitGpuStatus::SUCCESS;
            if (!gpu_ready) {
                std::cerr << "GPU memory sampling will not be available" << std::endl;
            }
        }

        return {model_path, device, gpu_ready};
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
        auto sys_sample = sampleSystemMemory();
        auto sample = MemoryCounters{
            .virtual_size = sys_sample.virtual_size,
            .virtual_peak = sys_sample.virtual_peak,
            .resident_size = sys_sample.resident_size,
            .resident_peak = sys_sample.resident_peak,
            .thread_count = sys_sample.thread_count
        };
        if (gpu_ready) {
            auto gpu_sample = sampleGpuMemory();
            sample.gpu_local_used = gpu_sample.local_used;
            sample.gpu_local_total = gpu_sample.local_total;
            sample.gpu_nonlocal_used = gpu_sample.nonlocal_used;
            sample.gpu_nonlocal_total = gpu_sample.nonlocal_total;
        }
        samples.push_back({sample_name, sample});
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
            << "\"threads\": " << sample.second.thread_count << ", "
            << "\"gpu_local_used\": " << sample.second.gpu_local_used << ", "
            << "\"gpu_local_total\": " << sample.second.gpu_local_total << ", "
            << "\"gpu_nonlocal_used\": " << sample.second.gpu_nonlocal_used << ", "
            << "\"gpu_nonlocal_total\": " << sample.second.gpu_nonlocal_total << "}";
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
