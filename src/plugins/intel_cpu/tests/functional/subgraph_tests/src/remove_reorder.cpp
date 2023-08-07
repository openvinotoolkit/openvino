// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

/*
  This test runs the following subgraph:

              param1          param2
                |               |
              (INT)             |
                |               |
             Convert            |
                |               |
         (No Reorder BF16)      |
                |               |
      const     |   const       |
           \    |    /          |
            \   |   /          /
              Range           /
                |            /
                \           /
                 \         /
                  \       /
                    MatMul 
                      |
                    Result
  
  The main purpose of this test is checking the code path when `Any-Convert-(FP32)-Range` pattern is disabled to transformate `Any-Convert-(BF16)-Range` on the SPR machine.
  The reason to disable this conversion is `Any-Convert-(BF16)` may cause precision loss if `Any` is a relative big INTEGER value.
  This scenario is always generated from PT FE or PPP. In order to avaid conversion loss, just keep this pattern and ignore automatic `EnforceInferencePrecision`.
*/

using RemoveUselessReorderCPUTestParams = std::tuple<
    ElementType,
    std::vector<InputShape>,
    std::vector<ov::Shape>
>;

class RemoveUselessReorderCPUTest: public testing::WithParamInterface<RemoveUselessReorderCPUTestParams>,
                                 virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RemoveUselessReorderCPUTestParams> obj) {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::vector<ov::Shape> targetShapes;
        std::tie(inType, inputShapes, targetShapes) = obj.param;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        result << "Prc=" << inType;
        return result.str();
    }
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::vector<ov::Shape> targetShapes;
        std::tie(inType, inputShapes, targetShapes) = this->GetParam();

        if (inType == ElementType::bf16) {
            configuration.insert({"INFERENCE_PRECISION_HINT", "bf16"});
        } else if (inType == ElementType::f16) {
            configuration.insert({"INFERENCE_PRECISION_HINT", "f16"});
        }

        init_input_shapes(inputShapes);
        auto input_params = ngraph::builder::makeDynamicParams(ov::element::f32, inputDynamicShapes);
        auto zero = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(input_params[0], ov::element::i32);
        auto gather = std::make_shared<ov::op::v8::Gather>(shapeof, one, zero);
        auto convert = std::make_shared<ov::op::v0::Convert>(gather, ov::element::f32);
        auto start = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0);
        auto step = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1);
        auto range = std::make_shared<ov::op::v4::Range>(start, convert, step, ov::element::f32);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(range, input_params[1], false, true);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        result->set_friendly_name("output");
        ov::ResultVector output_results = {result};

        function = std::make_shared<ov::Model>(output_results, input_params, "remove_bf16_reorder");
    };
};

namespace {
TEST_P(RemoveUselessReorderCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 4);
}

const std::vector<std::vector<InputShape>> input_shapes = {
    {
        {{1, -1}, {{1, 291}}},  // input 1
        {{1, -1}, {{1, 291}}},  // input 2
    }
};

const std::vector<std::vector<ov::Shape>> target_shapes = {
    {
        {1},
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_RemoveUselessReorderCPUTest,
                         RemoveUselessReorderCPUTest,
                         ::testing::Combine(::testing::Values(ElementType::bf16, ElementType::f16),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(target_shapes)),
                         RemoveUselessReorderCPUTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
