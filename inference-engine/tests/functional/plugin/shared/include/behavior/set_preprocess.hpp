// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ie_preprocess.hpp"
#include "common_test_utils/behavior_test_utils.hpp"

using PreprocessBehTest = BehaviorTestsUtils::BehaviorTestsBasic;

TEST_P(PreprocessBehTest, SetPreProcessToInputInfo) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    {
        InferenceEngine::ConstInputsDataMap inputsMap = execNet.GetInputsInfo();
        const auto &name = inputsMap.begin()->second->name();
        const InferenceEngine::PreProcessInfo *info = &req.GetPreProcess(name.c_str());
        ASSERT_EQ(info->getResizeAlgorithm(), InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        ASSERT_PREPROCESS_INFO_EQ(preProcess, *info);
    }
}

TEST_P(PreprocessBehTest, SetPreProcessToInferRequest) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);

    auto &preProcess = cnnNet.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    InferenceEngine::ConstInputsDataMap inputsMap = execNet.GetInputsInfo();
    const auto &name = inputsMap.begin()->second->name();
    auto inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    {
        const InferenceEngine::PreProcessInfo *info = &req.GetPreProcess(name.c_str());
        ASSERT_EQ(cnnNet.getInputsInfo().begin()->second->getPreProcess().getResizeAlgorithm(),
                  info->getResizeAlgorithm());
    }
}
