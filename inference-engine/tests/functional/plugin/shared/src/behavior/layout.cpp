// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/layout.hpp"

namespace BehaviorTestsUtils {
std::string LayoutTest::getTestCaseName(testing::TestParamInfo<LayoutParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    InferenceEngine::Layout layout;
    std::vector<size_t> inputShapes;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration, layout, inputShapes) = obj.param;
    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    result << "layout=" << layout << "_";
    if (!inputShapes.empty()) {
        for (auto &Item : inputShapes) {
            result << "inputShapes=" << Item << "_";
        }
    }
    return result.str();
}

void LayoutTest::SetUp()  {
    std::tie(netPrecision, targetDevice, configuration, layout, inputShapes) = this->GetParam();
    function = ngraph::builder::subgraph::make2InputSubtractIn(inputShapes, netPrecision);
}

void LayoutTest::TearDown()  {
    if ((targetDevice == CommonTestUtils::DEVICE_GPU) || (!configuration.empty())) {
        PluginCache::get().reset();
    }
    function.reset();
}

inline bool checkLayout(InferenceEngine::Layout layout, std::vector<size_t> &inputShapes) {
    bool check = false;
    switch (layout) {
        case InferenceEngine::Layout::C:
            check = 1 == inputShapes.size();
            break;
        case InferenceEngine::Layout::BLOCKED:
        case InferenceEngine::Layout::ANY:
            check = true;
            break;
        case InferenceEngine::Layout::GOIDHW:
            check = 6 == inputShapes.size();
            break;
        case InferenceEngine::Layout::NCDHW:
        case InferenceEngine::Layout::NDHWC:
        case InferenceEngine::Layout::OIDHW:
        case InferenceEngine::Layout::GOIHW:
            check = 5 == inputShapes.size();
            break;
        case InferenceEngine::Layout::OIHW:
        case InferenceEngine::Layout::NCHW:
        case InferenceEngine::Layout::NHWC:
            check = 4 == inputShapes.size();
            break;
        case InferenceEngine::Layout::CHW:
            check = 3 == inputShapes.size();
            break;
        case InferenceEngine::Layout::CN:
        case InferenceEngine::Layout::NC:
        case InferenceEngine::Layout::HW:
            check = 2 == inputShapes.size();
            break;
        default:
            break;
    }
    return check;
}

TEST_P(LayoutTest, NetWithLayout) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    // Create CNNNetwork from ngrpah::Function
    InferenceEngine::CNNNetwork cnnNet(function);
    if (checkLayout(layout, inputShapes)) {
        ASSERT_NO_THROW(cnnNet.getInputsInfo().begin()->second->setLayout(layout));
    } else {
        ASSERT_THROW(cnnNet.getInputsInfo().begin()->second->setLayout(layout),
                     InferenceEngine::details::InferenceEngineException);
    }
    if (targetDevice != CommonTestUtils::DEVICE_GNA) {
        ASSERT_NO_THROW(InferenceEngine::ExecutableNetwork exeNetwork =
                                ie->LoadNetwork(cnnNet, targetDevice, configuration));
    }
}
}  // namespace BehaviorTestsUtils