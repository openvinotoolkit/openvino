
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

#include "subgraph_tests/codegen_gelu.hpp"

#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

namespace LayerTestsDefinitions {

    std::string CodegenGelu::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes0, newInputShapes;
        bool useSubgraph;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes0, useSubgraph, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "overSnippet=" << (useSubgraph ? "yes" : "no") << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // Gelu from bert-large-uncased-whole-word-masking-squad-fp32-onnx-0001
    void CodegenGelu::SetUp() {
        std::vector<size_t> inputShape0;
        InferenceEngine::Precision netPrecision;
        bool useSubgraph;
        std::tie(netPrecision, inputShape0, useSubgraph, targetDevice) = this->GetParam();

        auto input0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShape0});
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShape0});
        auto add = std::make_shared<ngraph::opset1::Add>(input0, input1);

        auto gelu = std::make_shared<ngraph::opset2::Gelu>(add);
        auto result = std::make_shared<ngraph::opset1::Result>(gelu);

        function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result},
            ngraph::ParameterVector{input0, input1},
            "CodegenGelu");

        if (useSubgraph) {
            ngraph::pass::InitNodeInfo().run_on_function(function);
            ngraph::pass::ConstantFolding().run_on_function(function);
        }
    }

TEST_P(CodegenGelu, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
