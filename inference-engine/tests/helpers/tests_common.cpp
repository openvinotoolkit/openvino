// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <vector>

#include "tests_common.hpp"
#include <cpp_interfaces/ie_executor_manager.hpp>

#ifdef _WIN32
# ifndef NOMINMAX
# define NOMINMAX
# endif
# include "Psapi.h"
#endif

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

void TestsCommon::SetUp() {
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << getVmSizeInKB() << "KB\n";
    }
    InferenceEngine::ExecutorManager::getInstance()->clear();
}

void TestsCommon::TearDown() {
    InferenceEngine::ExecutorManager::getInstance()->clear();
}

class BaseTestCreator {
protected:
    std::string _type;
public:
    explicit BaseTestCreator(const std::string& type) : _type(type) {}
    virtual ~BaseTestCreator() = default;

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
            std::make_shared<LayerTestCreator<InferenceEngine::MathLayer>>("Exp"),
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
            std::make_shared<LayerTestCreator<InferenceEngine::TopKLayer>>("TopK"),
            std::make_shared<LayerTestCreator<InferenceEngine::NonMaxSuppressionLayer>>("NonMaxSuppression"),
            std::make_shared<LayerTestCreator<InferenceEngine::ScatterLayer>>("ScatterUpdate")
    };
    return creators;
}

InferenceEngine::CNNLayer::Ptr TestsCommon::createLayer(const std::string& type) {
    for (auto& creator : getCreators()) {
        if (!creator->shouldCreate(type))
            continue;
        return creator->create(type);
    }
    static LayerTestCreator<InferenceEngine::GenericLayer> genericCreator("");
    return genericCreator.create(type);
}
