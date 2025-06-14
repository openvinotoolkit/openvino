// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class EnforceInferencePrecisionFP16TailTest : virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<>> /*obj*/) {
        return "EnforceInferencePrecisionFP16TailTest";
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration = {{ov::hint::inference_precision.name(), ov::element::f16}};

        std::vector<InputShape> inputShapes = {{{-1, 16, 16, 16}, {{1, 16, 16, 16}, {2, 16, 16, 16}}}};

        init_input_shapes(inputShapes);

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
        auto mul_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 16, 16}, {2.0f});
        auto mul = std::make_shared<ov::op::v1::Multiply>(conv, mul_const);

        auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(conv);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{mul, sigmoid}, 1);

        auto convert_to_f32 = std::make_shared<ov::op::v0::Convert>(concat, ov::element::f32);

        auto result = std::make_shared<ov::op::v0::Result>(convert_to_f32);

        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{input},
                                               "enforce_inference_precision_fp16_tail");
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
TEST_F(EnforceInferencePrecisionFP16TailTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx512_core_amx_fp16())
        GTEST_SKIP() << "Skipping test, only fp16 runtime inference precision platform needed" << std::endl;
    run();
    serialize();
    checkResults();
}
}  // namespace
}  // namespace test
}  // namespace ov