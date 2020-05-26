// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <thread>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/infer_request_input.hpp"


namespace LayerTestsDefinitions {
    std::string InferRequestInputTests::getTestCaseName(testing::TestParamInfo<InferRequestInputParams> obj) {
        InferenceEngine::Precision  netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            result << "configItem=" << configuration.begin()->first << "_" << configuration.begin()->second;
        }
        return result.str();
    }

    void InferRequestInputTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void InferRequestInputTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

    TEST_P(InferRequestInputTests, canSetInputBlobForSyncRequest) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        InferenceEngine::Blob::Ptr inputBlob =
                FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
        InferenceEngine::Blob::Ptr actualBlob;
        ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
        ASSERT_EQ(inputBlob, actualBlob);
        function.reset();
    }

    TEST_P(InferRequestInputTests, canInferWithSetInOut) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        InferenceEngine::Blob::Ptr inputBlob =
                FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
        InferenceEngine::Blob::Ptr outputBlob =
                FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
        ASSERT_NO_THROW(req.Infer());
        function.reset();
    }

    TEST_P(InferRequestInputTests, canGetInputBlob_deprecatedAPI) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        std::shared_ptr<InferenceEngine::Blob> actualBlob;

        ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
        ASSERT_TRUE(actualBlob) << "Plugin didn't allocate input blobs";
        ASSERT_FALSE(actualBlob->buffer() == nullptr) << "Plugin didn't allocate input blobs";

        auto tensorDescription = actualBlob->getTensorDesc();
        auto dims = tensorDescription.getDims();
        ASSERT_TRUE(cnnNet.getInputsInfo().begin()->second->getTensorDesc().getDims() == dims)
                                    << "Input blob dimensions don't match network input";

        ASSERT_EQ(execNet.GetInputsInfo().begin()->second->getPrecision(), tensorDescription.getPrecision())
                                    << "Input blob precision don't match network input";
        function.reset();
    }

TEST_P(InferRequestInputTests, getAfterSetInputDoNotChangeInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
    // Get Core from cache
        auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    // Create InferRequest
        InferenceEngine::InferRequest req = execNet.CreateInferRequest();
        std::shared_ptr<InferenceEngine::Blob> inputBlob = FuncTestUtils::createAndFillBlob(
                cnnNet.getInputsInfo().begin()->second->getTensorDesc());
        ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
        std::shared_ptr<InferenceEngine::Blob> actualBlob;
        ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
        ASSERT_EQ(inputBlob.get(), actualBlob.get());
        function.reset();
    }

    TEST_P(InferRequestInputTests, canInferWithGetInOut) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
        InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
        ASSERT_NO_THROW(req.Infer());
        function.reset();
    }

    TEST_P(InferRequestInputTests, canStartAsyncInferWithGetInOut) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(function);
        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        // Create InferRequest
        InferenceEngine::InferRequest req;
        ASSERT_NO_THROW(req = execNet.CreateInferRequest());
        InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
        InferenceEngine::StatusCode sts;
        ASSERT_NO_THROW(req.Infer());
        ASSERT_NO_THROW(req.StartAsync());
        sts = req.Wait(500);
        ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
        InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
        function.reset();
    }


}  // namespace LayerTestsDefinitions