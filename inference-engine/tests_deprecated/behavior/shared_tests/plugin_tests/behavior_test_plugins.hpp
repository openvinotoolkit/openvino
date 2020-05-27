// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include <thread>

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
    std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
        return obj.param.device + "_" + obj.param.input_blob_precision.name() + "_" + getModelName(obj.param.model_xml_str)
               + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
    }

    std::string getOutputTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
        return obj.param.device + "_" + obj.param.output_blob_precision.name()
               + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
    }
}

Blob::Ptr BehaviorPluginTest::makeNotAllocatedBlob(Precision eb, Layout l, const SizeVector &dims) {
    TensorDesc tdesc (eb, dims, l);
    switch (eb) {
        case Precision::I8:
            return make_shared<TBlob<int8_t>>(tdesc);
        case Precision::I16:
            return make_shared<TBlob<int16_t>>(tdesc);
        case Precision::I32:
            return make_shared<TBlob<int32_t>>(tdesc);
        case Precision::U8:
            return make_shared<TBlob<uint8_t>>(tdesc);
        case Precision::U16:
            return make_shared<TBlob<uint16_t>>(tdesc);
        case Precision::FP16:
            return make_shared<TBlob<uint16_t>>(tdesc);
        case Precision::FP32:
            return make_shared<TBlob<float>>(tdesc);
        case Precision::Q78:
            return make_shared<TBlob<uint16_t>>(tdesc);
        case Precision::UNSPECIFIED:
            return make_shared<TBlob<float>>(tdesc);
        default:
            break;
    }
    throw std::runtime_error("unexpected");
}

void BehaviorPluginTest::setInputNetworkPrecision(CNNNetwork &network, InputsDataMap &inputs_info,
    Precision input_precision) {
    inputs_info = network.getInputsInfo();
    ASSERT_TRUE(inputs_info.size() == 1u);
    inputs_info.begin()->second->setPrecision(input_precision);
}

void BehaviorPluginTest::setOutputNetworkPrecision(CNNNetwork &network, OutputsDataMap &outputs_info,
    Precision output_precision) {
    outputs_info = network.getOutputsInfo();
    ASSERT_EQ(outputs_info.size(), 1u);
    outputs_info.begin()->second->setPrecision(output_precision);
}

class BehaviorPluginTestInput : public BehaviorPluginTest { };
class BehaviorPluginTestOutput : public BehaviorPluginTest { };

TEST_F(BehaviorPluginTest, smoke_llocateNullBlob) {
    TensorDesc tdesc = TensorDesc(Precision::FP32, NCHW);
    InferenceEngine::TBlob<float> blob(tdesc);
    ASSERT_NO_THROW(blob.allocate());
}

// Create Plugin
// TEST_P(BehaviorPluginTest, canCreatePlugin) {
//     ASSERT_NO_THROW(InferenceEnginePluginPtr plugin(make_plugin_name(GetParam().pluginName)));
// }

// Load correct network to Plugin
// TODO
// TEST_P(BehaviorPluginTest, canLoadCorrectNetwork) {
//     InferenceEnginePluginPtr plugin(make_plugin_name(GetParam().pluginName));
//     ASSERT_NO_THROW(pluginLoadCorrectNetwork(GetParam(), plugin));
// }

// // TODO
// // Load correct network to Plugin
// TEST_P(BehaviorPluginTest, canLoadTwoNetworks) {
//     auto param = GetParam();
//     InferenceEnginePluginPtr plugin(make_plugin_name(param.pluginName));
//     pluginLoadCorrectNetwork(param, plugin);
//     ASSERT_NO_THROW(pluginLoadCorrectNetwork(param, plugin));
// }

// Load incorrect network to Plugin
TEST_P(BehaviorPluginTest, canNotLoadNetworkWithoutWeights) {
    InferenceEngine::Core core;
    ASSERT_THROW(core.ReadNetwork(GetParam().model_xml_str, Blob::CPtr()), InferenceEngineException);
}

bool static compare_two_files_lexicographically(const std::string& name_a, const std::string& name_b) {
    std::ifstream a(name_a), b(name_b);

    std::string line_a, line_b;
    while (std::getline(a, line_a)) {
        std::string str_a, str_b;
        std::istringstream(line_a) >> str_a;

        if (!std::getline(b, line_b))
            throw std::logic_error("Second file is shorter than first");
        else
            std::istringstream(line_b) >> str_b;

        if (line_a != line_b) {
            std::cout << "Line A: " << line_a << std::endl;
            std::cout << "Line B: " << line_b << std::endl;
            throw std::logic_error("Files are different");
        }
    }

    if (std::getline(b, line_b))
        throw std::logic_error("First file is shorter than second");
    else
        return true;
}

TEST_P(BehaviorPluginTest, pluginDoesNotChangeOriginalNetwork) {
    const std::string name_a = "a.xml";
    const std::string name_b = "b.xml";
    auto param = GetParam();

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(param.model_xml_str, param.weights_blob);
    network.serialize(name_a);

    ASSERT_NO_THROW(core.LoadNetwork(network, param.device, param.config));
    network.serialize(name_b);

    ASSERT_NO_THROW(compare_two_files_lexicographically(name_a, name_b));
    EXPECT_EQ(0, std::remove(name_a.c_str()));
    EXPECT_EQ(0, std::remove(name_b.c_str()));
}

TEST_P(BehaviorPluginTestInput, canSetInputPrecisionForNetwork) {
    auto param = GetParam();
    InputsDataMap inputs_info;

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(param.model_xml_str, param.weights_blob);
    setInputNetworkPrecision(network, inputs_info, param.input_blob_precision);

    // Input image format I16 is not supported yet.
    // Disable verification for myriad plugin: CVS-7979, CVS-8144
    if ( (  param.device == CommonTestUtils::DEVICE_MYRIAD
            || param.device == CommonTestUtils::DEVICE_HDDL
            || param.device == CommonTestUtils::DEVICE_KEEMBAY)
         && param.input_blob_precision == Precision::I16) {
        std::string msg;
        StatusCode sts = StatusCode::OK;
        try {
            core.LoadNetwork(network, GetParam().device, param.config);
        } catch (InferenceEngineException ex) {
            msg = ex.what();
            sts = ex.getStatus();
        }
        ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << msg;
        std::string refError = "Input image format I16 is not supported yet.";
        //response.msg[refError.length()] = '\0';
        ASSERT_EQ(refError, msg);
    }
    else {
        ASSERT_NO_THROW(core.LoadNetwork(network, GetParam().device, param.config));
    }
}

TEST_P(BehaviorPluginTestOutput, canSetOutputPrecisionForNetwork) {
    auto param = GetParam();
    OutputsDataMap outputs_info;

    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(param.model_xml_str, param.weights_blob);

    setOutputNetworkPrecision(network, outputs_info, param.output_blob_precision);

    StatusCode sts = StatusCode::OK;
    std::string msg;

    try {
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, param.device, GetParam().config);
    } catch (InferenceEngineException ex) {
        sts = ex.getStatus();
        msg = ex.what();
        std::cout << "LoadNetwork() threw InferenceEngineException. Status: " << sts << ", message: " << msg << std::endl;
    }

    if ((param.output_blob_precision == Precision::I16 || param.output_blob_precision == Precision::U8)) {
        if (param.device == "CPU") {
            ASSERT_EQ(StatusCode::OK, sts);
        }
        else if (param.device == "GPU") {
            // Supported precisions: FP32, FP16
            ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << msg;
            std::string refError = "The plugin does not support output";
            ASSERT_STR_CONTAINS(msg, refError);
        }
        else {
            // Supported precisions: FP32, FP16
            ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << msg;
            std::string refError = "Unsupported output precision!";
            ASSERT_STR_CONTAINS(msg, refError);
        }
    }
    else {
        ASSERT_EQ(StatusCode::OK, sts);
    }
}
