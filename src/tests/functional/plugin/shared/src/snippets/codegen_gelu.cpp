
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/codegen_gelu.hpp"

#include "common_test_utils/common_utils.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_gelu.hpp"

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
        auto [netPrecision, inputShape0, inputShape1, useSubgraph, targetDeviceValue] = this->GetParam();
        targetDevice = targetDeviceValue;

        init_input_shapes({inputShape0, inputShape1});

        auto f = ov::test::snippets::CodegenGeluFunction(inputDynamicShapes, netPrecision);
        function = f.getOriginal();

        setInferenceType(netPrecision);

        if (useSubgraph) {
            ov::pass::InitNodeInfo().run_on_model(function);
            ov::pass::ConstantFolding().run_on_model(function);
        }
        setIgnoreCallbackMode();
    }

TEST_P(CodegenGelu, CompareWithRefImpl) {
    run();
};


} // namespace snippets
} // namespace test
} // namespace ov
