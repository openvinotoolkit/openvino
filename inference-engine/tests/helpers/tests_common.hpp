// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "inference_engine.hpp"
#include "test_model_path.hpp"
#include <tests_file_utils.hpp>
#include <cctype>
#include <chrono>

#ifdef WIN32
#define UNUSED
#else
#define UNUSED  __attribute__((unused))
#endif

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#ifdef _WIN32
	#include "Psapi.h"
#endif

class TestsCommon : public ::testing::Test {
public:
    static size_t parseLine(char* line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        size_t i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = (size_t)atoi(p);
        return i;
    }

    static size_t getVmSizeInKB(){
        FILE* file = fopen("/proc/self/status", "r");
        size_t result = 0;
        if (file != nullptr) {
            char line[128];

            while (fgets(line, 128, file) != NULL) {
                if (strncmp(line, "VmSize:", 7) == 0) {
                    result = parseLine(line);
                    break;
                }
            }
            fclose(file);
        }
        return result;
    }
#ifdef _WIN32
	static size_t getVmSizeInKBWin() {
		PROCESS_MEMORY_COUNTERS pmc;
		pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
		GetProcessMemoryInfo(GetCurrentProcess(),&pmc, pmc.cb);
		return pmc.WorkingSetSize;
	}
#endif

 public:
#ifdef _WIN32
    static std::string library_path() {return ".";};
#else
    static std::string library_path() { return "./lib";};
#endif  // _WIN32

    static std::string archPath() {
        if (sizeof(void*) == 8) {
            return  "../../lib/intel64";
        } else {
            return  "../../lib/ia32";
        }
    }

    protected:
    void TearDown() override {}

    void SetUp() override {
        auto memsize = getVmSizeInKB();
        if (memsize != 0) {
            std::cout << "\nMEM_USAGE=" << getVmSizeInKB() << "KB\n";
        }
    }
    public:


    inline std::string get_mock_engine_name() {
        return make_plugin_name("mock_engine");
    }

    inline std::string get_mock_extension_name() {
        return make_plugin_name("mock_extensions");
    }
    static std::string get_data_path(){
        const char* data_path = std::getenv("DATA_PATH");

        if (data_path == NULL){
            if(DATA_PATH != NULL){
                data_path = DATA_PATH;
            } else{
                ::testing::AssertionFailure()<<"DATA_PATH not defined";
            }
        }
        return std::string(data_path);
    }

    static std::string make_so_name(const std::string & input) {
#ifdef _WIN32
    #ifdef __MINGW32__
        std::string pre = "lib";
        std::string ext = ".dll";
    #else
        std::string pre = "";
        std::string ext = ".dll";
    #endif
#elif __APPLE__
        std::string pre = "lib";
        std::string ext = ".dylib";
#else
        std::string pre = "lib";
        std::string ext = ".so";
#endif
        return pre + input + IE_BUILD_POSTFIX + ext;

    }

    static std::string make_plugin_name(const std::string & input) {
        return make_so_name(input);
    }

    static void fill_data(InferenceEngine::Blob::Ptr blob) {
        fill_data(blob->buffer().as<float*>(), blob->size());
    }

    static void fill_data(float *data, size_t size, size_t duty_ratio = 10) {
        for (size_t i = 0; i < size; i++) {
            if ( ( i / duty_ratio)%2 == 1) {
                data[i] = 0.0;
            } else {
                data[i] = sin((float)i);
            }
        }
    }

    static void fill_data_sine(float *data, size_t size, float center, float ampl, float omega) {
        for (size_t i = 0; i < size; i++) {
            data[i] = center + ampl * sin((float)i * omega);
        }
    }

    static void fill_data_const(float *data, size_t size, float value) {
        for (size_t i = 0; i < size; i++) {
            data[i] = value;
        }
    }

    static void fill_data_dbgval(float *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
    }

    static void compare(InferenceEngine::Blob &res, InferenceEngine::Blob &ref, float max_diff = 0.01f) {

        float *res_ptr = res.buffer().as<float*>();
        size_t res_size = res.size();

        float *ref_ptr = ref.buffer().as<float*>();
        size_t ref_size = ref.size();

        ASSERT_EQ(res_size, ref_size);

        for (size_t i = 0; i < ref_size; i++) {
            ASSERT_NEAR(res_ptr[i], ref_ptr[i], max_diff);
        }
    }

    static void compare_NRMSD(InferenceEngine::Blob &res, InferenceEngine::Blob &ref, float max_nrmsd = 0.01f) {

        float *res_ptr = res.buffer().as<float*>();
        size_t res_size = res.size();

        float *ref_ptr = ref.buffer().as<float*>();
        size_t ref_size = ref.size();

        ASSERT_EQ(res_size, ref_size);

        float sum = 0;

        float mmin = ref_ptr[0], mmax = ref_ptr[0];

        for (size_t i = 0; i < ref_size; i++) {
            float sqr = (ref_ptr[i] - res_ptr[i]);
            sqr *= sqr;
            sum += sqr;

            mmin = (std::min)(mmin, ref_ptr[i]);
            mmax = (std::max)(mmax, ref_ptr[i]);

            if (i % 10007 == 0) {
                std::cout << i << ": " << res_ptr[i] << "\t" << ref_ptr[i] << "\t" << "\tdiv: " << ref_ptr[i] / res_ptr[i] << std::endl;
            }

        }
        sum /= ref_size;

        sum = pow(sum, 0.5);

        sum /= mmax - mmin;

        ASSERT_LE(sum, max_nrmsd);
    }

    static void compare(float* res, float* ref, size_t size, float max_diff = 0.01f) {
        for (size_t i = 0; i < size; i++) {
            ASSERT_NEAR(res[i], ref[i], max_diff);
        }
    }

    void replace(std::string& str, const std::string& from, const std::string& to)
    {
        std::string::size_type pos = 0;

        while((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    }

    std::string replace(std::string& str, const std::string& from, const int& to) {
        replace(str, from, std::to_string(to));
        return str;
    }

    std::string replace(std::string& str, const std::string& from, const size_t& to) {
        replace(str, from, std::to_string(to));
        return str;
    }

    std::string replace(std::string& str, const std::string& from, const float& to) {
        replace(str, from, std::to_string(to));
        return str;
    }
    // trim from both ends (in place)
    static inline std::string &trim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }


    template <class T>
    static InferenceEngine::StatusCode measurePerformance(const T & callInfer) {
        bool isPerformance = nullptr != getenv("DLSDK_performance_test");
        if (!isPerformance) {
            return callInfer();
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::nanoseconds ns;
        typedef std::chrono::duration<float> fsec;

        size_t niter = atoi(getenv("DLSDK_ITER_NUM"));
        std::vector<double> times(niter);
        InferenceEngine::StatusCode sts = InferenceEngine::OK;

        for (size_t i = 0; i < niter; ++i)
        {
            auto t0 = Time::now();
            sts = callInfer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ns d = std::chrono::duration_cast<ns>(fs);
            double total = static_cast<double>(d.count());

            times[i] = total*0.000001;
        }

        for (size_t i = 0; i < times.size(); i++)
            std::cout << "Iteration: " << i << " | infer time: " << times[i] << " ms" << std::endl;

        std::sort(times.begin(), times.end());

        size_t first_index = (size_t)floor(times.size() * 0.25);
        size_t last_index = (size_t)floor(times.size() * 0.75);
        size_t num = last_index - first_index;

        std::cout << "Q25: " << times[first_index] << std::endl;
        std::cout << "Q75: " << times[last_index]  << std::endl;

        if (niter < 4)
        {
            first_index = 0;
            last_index = times.size();
            num = times.size();
        }

        std::vector<double> clipped_times;
        double mean = 0;
        for (auto i = first_index; i < last_index; i++)
        {
            clipped_times.push_back(times[i]);
            mean += times[i];
        }

        mean = mean/clipped_times.size();

        double median = 0;
        if (clipped_times.size()%2 != 0)
            median = clipped_times[int(clipped_times.size()/2)];
        else median = (clipped_times[int(clipped_times.size()/2)] + clipped_times[int(clipped_times.size()/2)-1])/2;

        std::cout << "mean: " << mean << std::endl;
        std::cout << "median: " << median << std::endl;

        times.clear();
        clipped_times.clear();

        return sts;
    }

};


// Check bitness
#include <stdint.h>
#if UINTPTR_MAX == 0xffffffff
    /* 32-bit */
    #define ENVIRONMENT32
#elif UINTPTR_MAX == 0xffffffffffffffff
    /* 64-bit */
    #define ENVIRONMENT64
#else
    # error Unsupported architecture
#endif


template <typename T,typename S>
std::shared_ptr<InferenceEngine::TBlob<T>> to_tblob(const std::shared_ptr<S> &obj)
{
    return std::dynamic_pointer_cast<InferenceEngine::TBlob<T>>(obj);
}

inline InferenceEngine::InputInfo::Ptr getFirstInput(InferenceEngine::ICNNNetwork *pNet)
{
    InferenceEngine::InputsDataMap inputs;
    pNet->getInputsInfo(inputs);
    //ASSERT_GT(inputs.size(), 0);
    return inputs.begin()->second;
}

using OptionsMap = std::map<std::string, std::string>;
