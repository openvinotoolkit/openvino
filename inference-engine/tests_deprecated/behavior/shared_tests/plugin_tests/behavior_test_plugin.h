// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BEHAVIOR_TEST_PLUGIN_H_
#define BEHAVIOR_TEST_PLUGIN_H_

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <inference_engine.hpp>
#include <ie_plugin_config.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>
#include <gna/gna_config.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <memory>
#include <fstream>

#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;
using namespace InferenceEngine::GNAConfigParams;

class BehTestParams {
public:
    std::string device;

    std::string model_xml_str;
    Blob::Ptr weights_blob;

    Precision input_blob_precision;
    Precision output_blob_precision;

    std::map<std::string, std::string> config;
    uint8_t batch_size;

    BehTestParams() = default;

    BehTestParams(
            const std::string &_device,
            const std::string &_model_xml_str,
            const Blob::Ptr &_weights_blob,
            Precision _input_blob_precision,
            const std::map<std::string, std::string> &_config = {},
            Precision _output_blob_precision = Precision::FP32) : device(_device),
                                                                  model_xml_str(_model_xml_str),
                                                                  weights_blob(_weights_blob),
                                                                  input_blob_precision(_input_blob_precision),
                                                                  output_blob_precision(_output_blob_precision),
                                                                  config(_config) {}

    BehTestParams &withIn(Precision _input_blob_precision) {
        input_blob_precision = _input_blob_precision;
        return *this;
    }

    BehTestParams &withOut(Precision _output_blob_precision) {
        output_blob_precision = _output_blob_precision;
        return *this;
    }

    BehTestParams &withConfig(std::map<std::string, std::string> _config) {
        config = _config;
        return *this;
    }

    BehTestParams &withIncorrectConfigItem() {
        config.insert({"some_nonexistent_key", "some_unknown_value"});
        return *this;
    }

    BehTestParams &withBatchSize(uint8_t _batch_size) {
        batch_size = _batch_size;
        return *this;
    }

    static std::vector<BehTestParams>
    concat(std::vector<BehTestParams> const &v1, std::vector<BehTestParams> const &v2) {
        std::vector<BehTestParams> retval;
        std::copy(v1.begin(), v1.end(), std::back_inserter(retval));
        std::copy(v2.begin(), v2.end(), std::back_inserter(retval));
        return retval;
    }
};

class BehaviorPluginTest : public TestsCommon, public WithParamInterface<BehTestParams> {
protected:

    StatusCode sts;
    InferenceEngine::ResponseDesc response;

};

class TestModel {
public:
    std::string model_xml_str;
    InferenceEngine::Blob::Ptr weights_blob;
    TestModel(const std::string &model, const InferenceEngine::Blob::Ptr &weights) : model_xml_str(model) , weights_blob(weights) {}
};

TestModel getConvReluNormPoolFcModel(InferenceEngine::Precision netPrc);
const TestModel convReluNormPoolFcModelFP32 = getConvReluNormPoolFcModel(InferenceEngine::Precision::FP32);
const TestModel convReluNormPoolFcModelFP16 = getConvReluNormPoolFcModel(InferenceEngine::Precision::FP16);
const TestModel convReluNormPoolFcModelQ78 = getConvReluNormPoolFcModel(InferenceEngine::Precision::Q78);

class FPGAHangingTest : public BehaviorPluginTest {
};

#endif