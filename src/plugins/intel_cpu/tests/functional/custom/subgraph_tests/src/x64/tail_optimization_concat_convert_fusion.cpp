// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Define test parameter structure
struct ConcatTestParams {
    std::vector<InputShape> inputShapes;
    int concatAxis;
    std::string testName;
    std::vector<int> reshapePattern;
    std::vector<int> mulConstShape;
};

using TailOptimizationConcatConvertFusionTestParams = ConcatTestParams;

class TailOptimizationConcatConvertFusionTest
    : public testing::WithParamInterface<TailOptimizationConcatConvertFusionTestParams>,
      virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TailOptimizationConcatConvertFusionTestParams> obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << params.testName << "_";
        result << "axis" << params.concatAxis << "_";
        result << "IS=";
        for (const auto& shape : params.inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration = {{ov::hint::inference_precision.name(), ov::element::f16}};

        auto params = this->GetParam();
        init_input_shapes(params.inputShapes);

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        ov::Shape weights_shape = {16, 16, 1, 1};  // OIHW for 1x1 conv

        auto weights = ov::op::v0::Constant::create(ov::element::f16, weights_shape, {1.0f});
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});
        conv->set_friendly_name("conv_node");

        // Set reshape pattern according to parameters
        auto pattern = ov::op::v0::Constant::create(ov::element::i64,
                                                    ov::Shape{params.reshapePattern.size()},
                                                    params.reshapePattern);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(conv, pattern, false);

        // Set mul_const shape according to parameters
        auto mul_const =
            ov::op::v0::Constant::create(ov::element::f16,
                                         ov::Shape(params.mulConstShape.begin(), params.mulConstShape.end()),
                                         {2.0f});
        auto mul = std::make_shared<ov::op::v1::Multiply>(reshape, mul_const);

        auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(reshape);

        // Use concat axis from parameters
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{mul, sigmoid}, params.concatAxis);

        auto convert_to_f32 = std::make_shared<ov::op::v0::Convert>(concat, ov::element::f32);

        auto result = std::make_shared<ov::op::v0::Result>(convert_to_f32);

        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{input},
                                               "enforce_inference_precision_fp16_tail_" + params.testName);
    }

    void checkResults() {
        for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
            if (node->get_friendly_name() == "conv_node") {
                ASSERT_EQ(node->get_output_element_type(0), ElementType::f16);
            }
        }
        CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    }
};

namespace {
TEST_P(TailOptimizationConcatConvertFusionTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx512_core_amx_fp16())
        GTEST_SKIP() << "Skipping test, only fp16 runtime inference precision platform needed" << std::endl;
    run();
    checkResults();
}

// Define test parameters
const std::vector<ConcatTestParams> concatTestParams = {
    // 1. InPlace case (concat axis=0, dimension 0 is static)
    {
        {{{16, 16, 16, -1}, {{16, 16, 16, 8}, {16, 16, 16, 12}}}},
        0,  // concat axis=0, dimension 0 is static, triggers inplace
        "InPlace",
        {256, 16, -1},  // reshape pattern: [256, 16, W]
        {256, 1, 1}     // mul_const shape: [256, 1, 1]
    },

    // 2. Non-InPlace hasOuterLoop case
    {
        {{{16, 16, 16, -1}, {{16, 16, 16, 8}, {16, 16, 16, 12}}}},
        1,  // concat axis=1, has outer dimension 0, so hasOuterLoop=true
        "HasOuterLoop",
        {16, 256, -1},  // reshape pattern: [16, 256, W]
        {1, 256, 1}     // mul_const shape: [1, 256, 1]
    },

    // 3. Non-InPlace !hasOuterLoop small data case (concat axis=0, dimension 0 is dynamic)
    {
        {{{-1, 16, 16, -1}, {{2, 16, 16, 2}, {2, 16, 16, 2}}}},  // dimension 0 is dynamic, small W value
        0,  // concat axis=0, outermost dimension, but dimension 0 is dynamic, so non-inplace
        "SmallData",
        {-1, 256, 4},  // reshape pattern: [N, 256, 4]
        {1, 256, 1}    // mul_const shape: [1, 256, 1]
    },

    // 4. Non-InPlace !hasOuterLoop large data case (concat axis=0, dimension 0 is dynamic)
    {
        {{{-1, 16, 16, -1}, {{16, 16, 16, 64}, {16, 16, 16, 64}}}},  // dimension 0 is dynamic, large W value
        0,  // concat axis=0, outermost dimension, but dimension 0 is dynamic, so non-inplace
        "LargeData",
        {-1, 256, 128},  // reshape pattern: [N, 256, 128]
        {1, 256, 1}      // mul_const shape: [1, 256, 1]
    }};

INSTANTIATE_TEST_SUITE_P(smoke_TailOptimizationConcatConvertFusion,
                         TailOptimizationConcatConvertFusionTest,
                         ::testing::ValuesIn(concatTestParams),
                         TailOptimizationConcatConvertFusionTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov