
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/codegen_add.hpp"

namespace LayerTestsDefinitions {

    std::string CodegenAdd::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes0, inputShapes1, newInputShapes;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes0, inputShapes1, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
        result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void CodegenAdd::SetUp() {
        std::vector<size_t> inputShape0, inputShape1;
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, inputShape0, inputShape1, targetDevice) = this->GetParam();

        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShape0});
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShape1});

        auto add = std::make_shared<ngraph::opset1::Add>(input0, input1);
        auto neg = std::make_shared<ngraph::opset1::Negative>(add);
        auto result = std::make_shared<ngraph::opset1::Result>(neg);

        function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result},
            ngraph::ParameterVector{input0, input1},
            "CodegenAdd");
    }

TEST_P(CodegenAdd, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
