// Copyright (C) 2018-2019 Intel Corporation
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

class BaseTestCreator {
protected:
    std::string _type;
public:
    explicit BaseTestCreator(const std::string& type) : _type(type) {}

    virtual InferenceEngine::CNNLayerPtr create(const std::string& type)  = 0;

    virtual bool shouldCreate(const std::string& type) = 0;
};

template<class LT>
class LayerTestCreator : public BaseTestCreator {
public:
    explicit LayerTestCreator(const std::string& type) : BaseTestCreator(type) {}

    InferenceEngine::CNNLayerPtr create(const std::string& type) override {
        InferenceEngine::LayerParams params;
        params.type = type;
        return std::make_shared<LT>(params);
    }

    bool shouldCreate(const std::string& type) override {
        return type == _type;
    }
};

class TestsCommon : public ::testing::Test {
private:
    static std::vector<std::shared_ptr<BaseTestCreator>>& getCreators() {
        // there should be unique_ptr but it cant be used with initializer lists
        static std::vector<std::shared_ptr<BaseTestCreator> > creators = {
                std::make_shared<LayerTestCreator<InferenceEngine::PowerLayer>>("Power"),
                std::make_shared<LayerTestCreator<InferenceEngine::ConvolutionLayer>>("Convolution"),
                std::make_shared<LayerTestCreator<InferenceEngine::DeconvolutionLayer>>("Deconvolution"),
                std::make_shared<LayerTestCreator<InferenceEngine::PoolingLayer>>("Pooling"),
                std::make_shared<LayerTestCreator<InferenceEngine::FullyConnectedLayer>>("InnerProduct"),
                std::make_shared<LayerTestCreator<InferenceEngine::FullyConnectedLayer>>("FullyConnected"),
                std::make_shared<LayerTestCreator<InferenceEngine::NormLayer>>("LRN"),
                std::make_shared<LayerTestCreator<InferenceEngine::NormLayer>>("Norm"),
                std::make_shared<LayerTestCreator<InferenceEngine::SoftMaxLayer>>("Softmax"),
                std::make_shared<LayerTestCreator<InferenceEngine::SoftMaxLayer>>("LogSoftMax"),
                std::make_shared<LayerTestCreator<InferenceEngine::GRNLayer>>("GRN"),
                std::make_shared<LayerTestCreator<InferenceEngine::MVNLayer>>("MVN"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReLULayer>>("ReLU"),
                std::make_shared<LayerTestCreator<InferenceEngine::ClampLayer>>("Clamp"),
                std::make_shared<LayerTestCreator<InferenceEngine::SplitLayer>>("Split"),
                std::make_shared<LayerTestCreator<InferenceEngine::SplitLayer>>("Slice"),
                std::make_shared<LayerTestCreator<InferenceEngine::ConcatLayer>>("Concat"),
                std::make_shared<LayerTestCreator<InferenceEngine::EltwiseLayer>>("Eltwise"),
                std::make_shared<LayerTestCreator<InferenceEngine::ScaleShiftLayer>>("ScaleShift"),
                std::make_shared<LayerTestCreator<InferenceEngine::PReLULayer>>("PReLU"),
                std::make_shared<LayerTestCreator<InferenceEngine::CropLayer>>("Crop"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReshapeLayer>>("Reshape"),
                std::make_shared<LayerTestCreator<InferenceEngine::TileLayer>>("Tile"),
                std::make_shared<LayerTestCreator<InferenceEngine::BatchNormalizationLayer>>("BatchNormalization"),
                std::make_shared<LayerTestCreator<InferenceEngine::GemmLayer>>("Gemm"),
                std::make_shared<LayerTestCreator<InferenceEngine::PadLayer>>("Pad"),
                std::make_shared<LayerTestCreator<InferenceEngine::GatherLayer>>("Gather"),
                std::make_shared<LayerTestCreator<InferenceEngine::StridedSliceLayer>>("StridedSlice"),
                std::make_shared<LayerTestCreator<InferenceEngine::ShuffleChannelsLayer>>("ShuffleChannels"),
                std::make_shared<LayerTestCreator<InferenceEngine::DepthToSpaceLayer>>("DepthToSpace"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReverseSequenceLayer>>("ReverseSequence"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Abs"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Acos"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Acosh"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Asin"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Asinh"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Atan"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Atanh"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Ceil"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Cos"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Cosh"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Erf"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Floor"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("HardSigmoid"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Log"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Reciprocal"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Selu"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Sign"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Sin"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Sinh"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Softplus"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Softsign"),
                std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Tan"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceAnd"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceL1"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceL2"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceLogSum"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceLogSumExp"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceMax"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceMean"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceMin"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceOr"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceProd"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceSum"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReduceLayer>>("ReduceSumSquare"),
                std::make_shared<LayerTestCreator<InferenceEngine::TopKLayer>>("TopK")
        };
        return creators;
    }
public:
    static InferenceEngine::CNNLayer::Ptr createLayer(const std::string& type) {
        for (auto& creator : getCreators()) {
            if (!creator->shouldCreate(type))
                continue;
            return creator->create(type);
        }
        static LayerTestCreator<InferenceEngine::GenericLayer> genericCreator("");
        return genericCreator.create(type);
    }

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

    static void fill_data(InferenceEngine::Blob::Ptr& blob) {
        fill_data(blob->buffer().as<float*>(), blob->byteSize() / sizeof(float));
    }

    static void fill_data_const(InferenceEngine::Blob::Ptr& blob, float val) {
        fill_data_const(blob->buffer().as<float*>(), blob->size(), val);
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

    static void fill_data_non_zero(int32_t *data, size_t size, int n) {
        for (size_t i = 0; i < size; i++) {
            data[i] = n*i%254+1;
        }
    }

    static void fill_data_bin(float *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = sinf((float)i) > 0.f ? 1.f : -1.f;
        }
    }

    static void fill_data_bin_packed(int8_t *data, size_t size) {
        int nbits = 8;
        for (size_t i = 0; i < div_up(size, nbits); i++) {
            data[i] = static_cast<int8_t>(i % 255);
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

    static void fill_data_dbgval(float *data, size_t size, float alpha = 1.0f) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i * alpha;
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

    static void compare(const float* res, const float* ref, size_t size, float max_diff = 0.01f) {
        for (size_t i = 0; i < size; i++) {
            ASSERT_NEAR(res[i], ref[i], max_diff);
        }
    }

    void replace(std::string& str, const std::string& from, const std::string& to) {
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

    template <typename T, typename U>
    static inline T div_up(const T a, const U b) {
        assert(b);
        return (a + b - 1) / b;
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
