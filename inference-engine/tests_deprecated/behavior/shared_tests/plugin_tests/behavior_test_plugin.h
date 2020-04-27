// Copyright (C) 2018-2020 Intel Corporation
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
#include <multi-device/multi_device_config.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <tests_utils.hpp>
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

namespace {
inline std::string getModelName(std::string strXML) {
    auto itBegin = strXML.find("<net name=\"");
    auto itEnd = strXML.find(">", itBegin + 1);
    auto substr = strXML.substr(itBegin + 1, itEnd - itBegin - 1);

    itBegin = substr.find("\"");
    itEnd = substr.find("\"", itBegin + 1);
    substr = substr.substr(itBegin + 1, itEnd - itBegin - 1);

    return substr;
}
}

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

    BehTestParams &withConfigItem(std::pair<std::string, std::string> _config_item) {
        config.insert(_config_item);
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

    static Blob::Ptr makeNotAllocatedBlob(Precision eb, Layout l, const SizeVector &dims);

    void setInputNetworkPrecision(CNNNetwork &network, InputsDataMap &inputs_info,
                                  Precision input_precision);

    void setOutputNetworkPrecision(CNNNetwork &network, OutputsDataMap &outputs_info,
                                   Precision output_precision);
};

class BehaviorPluginTestAllUnsupported : public BehaviorPluginTest {
};

class BehaviorPluginTestTypeUnsupported : public BehaviorPluginTest {
};

class BehaviorPluginTestBatchUnsupported : public BehaviorPluginTest {
};

class BehaviorPluginCorrectConfigTest : public BehaviorPluginTest {
};

class BehaviorPluginIncorrectConfigTest : public BehaviorPluginTest {
};

class BehaviorPluginIncorrectConfigTestInferRequestAPI : public BehaviorPluginTest {
};

class BehaviorPluginCorrectConfigTestInferRequestAPI : public BehaviorPluginTest {
};

class BehaviorPluginTestVersion : public BehaviorPluginTest {
};

class BehaviorPluginTestInferRequest : public BehaviorPluginTest {
public:
    struct TestEnv {
        // Intentionally defined Core before IInferRequest.
        // Otherwise plugin will be freed with unloading dll before freeing IInferRequest::Ptr and that will cause memory corruption.
        // TODO: the same story with IExecutableNetwork and IInferRequest, shared syncEnv object may cause seg fault,
        // if IExecutableNetwork was freed before IInferRequest
        InferenceEngine::Core core;
        InferenceEngine::ExecutableNetwork exeNetwork;
        IInferRequest::Ptr inferRequest;
        InferenceEngine::InferRequest actualInferRequest;
        CNNNetwork network;
        InputInfo::Ptr networkInput;
        DataPtr networkOutput;
        SizeVector inputDims;
        SizeVector outputDims;
        std::string inputName;
        std::string outputName;
        typedef std::shared_ptr<TestEnv> Ptr;
    };

    static Blob::Ptr prepareInputBlob(Precision blobPrecision, SizeVector inputDims);

protected:
    Blob::Ptr _prepareOutputBlob(Precision blobPrecision, SizeVector outputDims);

    void _setInputPrecision(
            const BehTestParams &param,
            CNNNetwork &cnnNetwork,
            TestEnv::Ptr &testEnv,
            const size_t expectedNetworkInputs = 0);

    void _setOutputPrecision(
            const BehTestParams &param,
            CNNNetwork &cnnNetwork,
            TestEnv::Ptr &testEnv,
            const size_t expectedNetworkOutputs = 0);

    void _createAndCheckInferRequest(
            const BehTestParams &param,
            TestEnv::Ptr &testEnv,
            const std::map<std::string, std::string> &config = {},
            const size_t expectedNetworkInputs = 1,
            const size_t expectedNetworkOutputs = 1,
            InferenceEngine::IExtensionPtr extension = nullptr);

    bool _wasDeviceBusy(ResponseDesc response);

};

class FPGAHangingTest : public BehaviorPluginTest {
};

class BehaviorPluginTestInferRequestInput : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestInferRequestOutput : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestInferRequestConfig : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestInferRequestConfigExclusiveAsync : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestInferRequestCallback : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestExecGraphInfo : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestPerfCounters : public BehaviorPluginTestInferRequest {
};

class BehaviorPluginTestPreProcess : public BehaviorPluginTestInferRequest {
};

#endif
