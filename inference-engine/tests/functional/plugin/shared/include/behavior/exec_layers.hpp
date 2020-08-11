// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/behavior_test_utils.hpp>

namespace BehaviorTestsDefinitions {
    using ExecLayerTests = BehaviorTestsUtils::BehaviorTestsBasic;

    TEST_P(ExecLayerTests, CheckMultiplyExecution) {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        // Create ngraph function
        auto params = ngraph::builder::makeParams(ngraph::element::f32, {{9, 256}});
        auto const_mult = ngraph::builder::makeConstant(ngraph::element::f32, {}, {-1.0f});
        auto mult = std::make_shared<ngraph::opset1::Multiply>(params[0], const_mult);
        auto func = std::make_shared<ngraph::Function>(mult, params);
        auto ie = InferenceEngine::Core();
        InferenceEngine::CNNNetwork cnnNet(func);
        ASSERT_THROW(ie.LoadNetwork(cnnNet, targetDevice), InferenceEngine::details::InferenceEngineException);
    }
} // namespace BehaviorTestsDefinitions
