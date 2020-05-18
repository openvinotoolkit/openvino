// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <debug.h>
#include "ie_core.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "single_layer_tests/multiply.hpp"

namespace LayerTestsDefinitions {
    std::string MultiplyLayerTest::getTestCaseName(const testing::TestParamInfo<MultiplyParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::tie(inputShapes, netPrecision, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void MultiplyLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShapes, netPrecision, targetDevice) = this->GetParam();
        const std::size_t input_dim = InferenceEngine::details::product(inputShapes[0]);
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> shape_input{1, input_dim};
        auto input = ngraph::builder::makeParams(ngPrc, {shape_input});
        auto const_mul = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1}, std::vector<float>{-1.0f});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(input[0], const_mul);
        function = std::make_shared<ngraph::Function>(mul, input, "multiply");
    }

    TEST_P(MultiplyLayerTest, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
