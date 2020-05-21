// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "behavior/set_preprocess.hpp"

namespace LayerTestsDefinitions {
    std::string PreProcessTests::getTestCaseName(testing::TestParamInfo<PreProcessParams> obj) {
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

    void PreProcessTests::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void PreProcessTests::TearDown() {
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

TEST_P(PreProcessTests, SetPreProcessToInputInfo) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);

    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    {
        InferenceEngine::ConstInputsDataMap inputsMap = execNet.GetInputsInfo();
        const auto& name = inputsMap.begin()->second->name();
        const InferenceEngine::PreProcessInfo *info = &req.GetPreProcess(name.c_str());
        ASSERT_EQ(info->getResizeAlgorithm(), InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        ASSERT_PREPROCESS_INFO_EQ(preProcess, *info);
    }
    function.reset();
    }

TEST_P(PreProcessTests, SetPreProcessToInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);

    // Get Core from cache
    auto ie = PluginCache::get().ie();
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    InferenceEngine::ConstInputsDataMap inputsMap = execNet.GetInputsInfo();
    const auto& name = inputsMap.begin()->second->name();
    auto inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    {
        const InferenceEngine::PreProcessInfo *info = &req.GetPreProcess(name.c_str());
        ASSERT_EQ(cnnNet.getInputsInfo().begin()->second->getPreProcess().getResizeAlgorithm(),
                info->getResizeAlgorithm());
    }
    function.reset();
    }

} // namespace LayerTestsDefinitions