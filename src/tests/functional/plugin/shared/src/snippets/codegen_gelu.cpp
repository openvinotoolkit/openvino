
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/constant_folding.hpp>
#include "common_test_utils/common_utils.hpp"
#include "snippets/codegen_gelu.hpp"
#include "subgraph_simple.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

    std::string CodegenGelu::getTestCaseName(testing::TestParamInfo<ov::test::snippets::CodegenGeluParams> obj) {
        ov::element::Type_t netPrecision;
        InputShape inputShapes0, inputShapes1;
        bool useSubgraph;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes0, inputShapes1, useSubgraph, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes0.first}) << "_";
        result << "TS[0]=";
        for (const auto& shape : inputShapes0.second) {
            result << "(" << ov::test::utils::vec2str(shape) << ")_";
        }
        result << "IS[1]=" << ov::test::utils::partialShape2str({inputShapes1.first}) << "_";
        result << "TS[1]=";
        for (const auto& shape : inputShapes1.second) {
            result << "(" << ov::test::utils::vec2str(shape) << ")_";
        }
        result << "netPRC=" << netPrecision << "_";
        result << "overSnippet=" << (useSubgraph ? "yes" : "no") << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // Gelu from bert-large-uncased-whole-word-masking-squad-fp32-onnx-0001
    void CodegenGelu::SetUp() {
        InputShape inputShape0, inputShapes1;
        ov::element::Type_t netPrecision;
        bool useSubgraph;
        std::tie(netPrecision, inputShape0, inputShapes1, useSubgraph, targetDevice) = this->GetParam();

        init_input_shapes({inputShape0, inputShapes1});

        auto input0 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[0]);
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[1]);
        auto add = std::make_shared<ngraph::opset1::Add>(input0, input1);

        auto gelu = std::make_shared<ngraph::opset2::Gelu>(add);
        auto result = std::make_shared<ngraph::opset1::Result>(gelu);

        function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result},
            ngraph::ParameterVector{input0, input1},
            "CodegenGelu");

        if (useSubgraph) {
            ov::pass::InitNodeInfo().run_on_model(function);
            ngraph::pass::ConstantFolding().run_on_model(function);
        }
    }

TEST_P(CodegenGelu, CompareWithRefImpl) {
    run();
};


} // namespace snippets
} // namespace test
} // namespace ov
