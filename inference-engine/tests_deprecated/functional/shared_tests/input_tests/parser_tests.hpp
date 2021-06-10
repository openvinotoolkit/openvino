// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <utility>
#include <cctype>
#include <ie_core.hpp>
#include "xml_net_builder.hpp"

struct layer_params {
    layer_params(std::string type, std::vector<size_t> in, std::vector<size_t> out,
                 int weights, int biases, std::map<std::string, std::string> params)
            : type(std::move(type)), in(std::move(in)), out(std::move(out)), params(std::move(params)),
              weights(weights), biases(biases) {}

    std::string type;
    std::vector<size_t> in;
    std::vector<size_t> out;
    int weights;
    int biases;
    std::map<std::string, std::string> params;
};

struct ir_test_params :  layer_params {
    ir_test_params(std::string name, std::string precision, layer_params param)
            : layer_params(param), device_name(name), precision(std::move(precision)) {}

    std::string device_name;
    std::string precision;
};

std::map<std::string, std::vector<std::string>> smokeTests{};

std::string getTestName(testing::TestParamInfo<ir_test_params> obj) {
    std::string name = obj.param.device_name + "__" + obj.param.precision + "__" + obj.param.type + "__";

    bool isSmoke{ false };
    if (smokeTests.find(obj.param.device_name) == smokeTests.end()) {
        smokeTests.insert(std::make_pair(obj.param.device_name, std::vector<std::string>{ obj.param.type }));
        isSmoke = true;
    }
    else {
        auto& typeVector = smokeTests.at(obj.param.device_name);
        bool flag = (std::find(typeVector.begin(), typeVector.end(), obj.param.type) == typeVector.end());
        if (flag) {
            typeVector.push_back(obj.param.type);
            isSmoke = true;
        }
    }

    if (isSmoke)
        name = obj.param.device_name + "__" + obj.param.precision + "__" + obj.param.type + "__";

    for (size_t i = 0; i < obj.param.in.size(); i++) {
        if (i)
            name += "_";
        name += std::to_string(obj.param.in[i]);
    }
    name += "__";
    for (size_t i = 0; i < obj.param.out.size(); i++) {
        if (i)
            name += "_";
        name += std::to_string(obj.param.out[i]);
    }
    name += "__";
    if (obj.param.weights < 0)
        name += "n";
    name += std::to_string(abs(obj.param.weights));
    name += "__";
    if (obj.param.biases < 0)
        name += "n";
    name += std::to_string(abs(obj.param.biases));
    name += "__";

    std::string param;
    for (const auto& it : obj.param.params) {
        if (!param.empty())
            name += "__";
        std::string key = it.first;
        std::string value = it.second;
        for (char &i : key) {
            if (!isalnum(i))
                i = '_';
        }
        for (char &i : value) {
            if (!isalnum(i))
                i = '_';
        }
        name += key + "___" + value;
    }
    name += param;
    return name;
}

class IncorrectIRTests: public TestsCommon,
                        public testing::WithParamInterface<ir_test_params> {
protected:
    InferenceEngine::TBlob<uint8_t>::Ptr GetNetworkWeights(const layer_params &p) {
        size_t weigtsSize = (abs(p.weights) + abs(p.biases))*sizeof(float);
        if (weigtsSize == 0)
            return nullptr;
        InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({
                InferenceEngine::Precision::U8, { weigtsSize }, InferenceEngine::Layout::C});
        weights->allocate();
        fill_data(weights->buffer().as<float*>(),
                  weights->size() / sizeof(float));
        InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

        return weights_ptr;
    }
};

TEST_F(IncorrectIRTests, smoke_loadIRWithIncorrectInput) {
    std::map<std::string, std::string> params = {{"negative_slope", "0"}};

    std::string model = CommonTestUtils::V2NetBuilder::buildNetworkWithOneInput("ReLU_WithInput_Only", {1, 3, 4, 4}, "FP32")
            .addLayer("ReLU", "FP32", &params, {{{1, 3, 2, 2}}, {{1, 3, 4, 4}}})
            .finish(false);

    InferenceEngine::Core ie;
    ASSERT_THROW(ie.ReadNetwork(model, InferenceEngine::Blob::CPtr()), 
        InferenceEngine::Exception);
}

TEST_P(IncorrectIRTests, loadIncorrectLayer) {
    auto param = GetParam();

    std::string model = CommonTestUtils::V2NetBuilder::buildNetworkWithOneInput(param.type + "_Only", param.in, param.precision)
            .addLayer(param.type, param.precision, &param.params, {{param.in}, {param.out}}, param.weights, param.biases)
            .finish(false);

    try {
        InferenceEngine::Core ie;
        auto network = ie.ReadNetwork(model, GetNetworkWeights(param));
        auto exec = ie.LoadNetwork(network, param.device_name);
    } catch(...) {
        return;
    }
    FAIL() << "Topology was loaded successfully.";
}

// Convolution
#define negative_conv_kernel_x_case layer_params("Convolution", {1, 3, 224, 224}, {1, 64, 112, 112}, 64*3*7*7, 64, {{"kernel-x", "-7"}, {"kernel-y", "7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_conv_kernel_y_case layer_params("Convolution", {1, 3, 224, 224}, {1, 64, 112, 112}, 64*3*7*7, 64, {{"kernel-x", "7"}, {"kernel-y", "-7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_conv_stride_x_case layer_params("Convolution", {1, 3, 224, 224}, {1, 64, 112, 112}, 64*3*7*7, 64, {{"kernel-x", "7"}, {"kernel-y", "7"}, {"stride-x", "-2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_conv_weights_case layer_params("Convolution", {1, 3, 224, 224}, {1, 64, 112, 112}, -64*3*7*7, 64, {{"kernel-x", "7"}, {"kernel-y", "7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_conv_biases_case layer_params("Convolution", {1, 3, 224, 224}, {1, 64, 112, 112}, 64*3*7*7, -64, {{"kernel-x", "7"}, {"kernel-y", "7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})

// Fully connected
#define negative_fc_out_size_case layer_params("InnerProduct", {1, 3, 224, 224}, {1, 64, 112, 112}, 224*224*3*1000, 1000, {{"out-size", "-1000"}})
#define negative_fc_weights_case layer_params("InnerProduct", {1, 3, 224, 224}, {1, 64, 112, 112}, -224*224*3*1000, 1000, {{"out-size", "1000"}})
#define negative_fc_biases_case layer_params("InnerProduct", {1, 3, 224, 224}, {1, 64, 112, 112}, 224*224*3*1000, -1000, {{"out-size", "1000"}})

// Deconvolution
#define negative_deconv_kernel_x_case layer_params("Deconvolution", {1, 64, 224, 224}, {1, 3, 112, 112}, 64*3*7*7, 64, {{"kernel-x", "-7"}, {"kernel-y", "7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_deconv_kernel_y_case layer_params("Deconvolution", {1, 64, 224, 224}, {1, 3, 112, 112}, 64*3*7*7, 64, {{"kernel-x", "7"}, {"kernel-y", "-7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_deconv_stride_x_case layer_params("Deconvolution", {1, 64, 224, 224}, {1, 3, 112, 112}, 64*3*7*7, 64, {{"kernel-x", "7"}, {"kernel-y", "7"}, {"stride-x", "-2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_deconv_weights_case layer_params("Deconvolution", {1, 64, 224, 224}, {1, 3, 112, 112}, -64*3*7*7, 64, {{"kernel-x", "7"}, {"kernel-y", "7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})
#define negative_deconv_biases_case layer_params("Deconvolution", {1, 64, 224, 224}, {1, 3, 112, 112}, 64*3*7*7, -64, {{"kernel-x", "7"}, {"kernel-y", "7"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "2"}, {"pad-y", "2"}, {"dilation-x", "0"}, {"dilation-y", "0"}, {"output", "64"}, {"group", "1"}})

// Pooling
#define negative_pool_kernel_x_case layer_params("Pooling", {1, 3, 224, 224}, {1, 3, 112, 112}, 0, 0, {{"kernel-x", "-2"}, {"kernel-y", "2"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "0"}, {"pad-y", "0"}, {"rounding-type", "ceil"}, {"pool-method", "max"}})
#define negative_pool_kernel_y_case layer_params("Pooling", {1, 3, 224, 224}, {1, 3, 112, 112}, 0, 0, {{"kernel-x", "2"}, {"kernel-y", "-2"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "0"}, {"pad-y", "0"}, {"rounding-type", "ceil"}, {"pool-method", "max"}})
#define negative_pool_stride_x_case layer_params("Pooling", {1, 3, 224, 224}, {1, 3, 112, 112}, 0, 0, {{"kernel-x", "2"}, {"kernel-y", "2"}, {"stride-x", "-2"}, {"stride-y", "2"}, {"pad-x", "0"}, {"pad-y", "0"}, {"rounding-type", "ceil"}, {"pool-method", "max"}})
#define incorrect_pool_type_case layer_params("Pooling", {1, 3, 224, 224}, {1, 3, 112, 112}, 0, 0, {{"kernel-x", "2"}, {"kernel-y", "2"}, {"stride-x", "2"}, {"stride-y", "2"}, {"pad-x", "0"}, {"pad-y", "0"}, {"rounding-type", "ceil"}, {"pool-method", "unknown"}})

// Norm
#define negative_norm_local_size_case layer_params("Norm", {1, 3, 224, 224}, {1, 3, 224, 224}, 0, 0, {{"alpha", "9.9999997e-05"}, {"beta", "0.75"}, {"local_size", "-5"}, {"region", "across"}, {"k", "1"}})
#define negative_norm_k_case layer_params("Norm", {1, 3, 224, 224}, {1, 3, 224, 224}, 0, 0, {{"alpha", "9.9999997e-05"}, {"beta", "0.75"}, {"local_size", "5"}, {"region", "across"}, {"k", "-2"}})


// TODO: Add Concat and split tests
