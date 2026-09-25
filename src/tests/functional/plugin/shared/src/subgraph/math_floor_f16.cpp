// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/math_floor_f16.hpp"

#include "common_test_utils/node_builders/activation.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/single_op/activation.hpp"

namespace ov {
namespace test {

const std::vector<MathFloorF16Case>& MathFloorF16Test::all_cases() {
    using ov::test::utils::ActivationTypes;
    // expected = Floor with f16 rounding kept; the value in the comment is what a pure f32 path computes.
    static const std::vector<MathFloorF16Case> cases = {
        {ActivationTypes::Cos, 0.0130767822265625f, 1.0f, 0.0f},     // cos = 0.99991   -> 0
        {ActivationTypes::Cosh, 3.63671875f, 19.0f, 0.0f},           // cosh = 18.9967  -> 18
        {ActivationTypes::Sin, 1.5546875f, 1.0f, 0.0f},              // sin = 0.99987   -> 0
        {ActivationTypes::Sinh, 2.998046875f, 10.0f, 0.0f},          // sinh = 9.99823  -> 9
        {ActivationTypes::Acos, -0.416015625f, 2.0f, 0.0f},          // acos = 1.99986  -> 1
        {ActivationTypes::Acosh, 1.54296875f, 1.0f, 0.0f},           // acosh = 0.99990 -> 0
        {ActivationTypes::Asin, 1.54296875f, 0.0f, 0.54541015625f},  // asin(0.8415508) = 1.00016 -> 1
        {ActivationTypes::Asinh, -3.62890625f, -2.0f, 0.0f},         // asinh = -2.00054 -> -3
        {ActivationTypes::Atan, -1.55859375f, -1.0f, 0.0f},          // atan = -1.00035 -> -2
        {ActivationTypes::Atanh, -0.76171875f, -1.0f, 0.0f},         // atanh = -1.00030 -> -2
        {ActivationTypes::Tan, 1.2490234375f, 3.0f, 0.0f},           // tan = 2.99978   -> 2
        {ActivationTypes::Sign, 0.0001f, 0.0f, 0.0001f},             // sign(1.0e-8) = 1 -> 1
        {ActivationTypes::SoftPlus, 4.9921875f, 5.0f, 0.0f},         // softplus = 4.99896 -> 4
        {ActivationTypes::SoftSign, 6304.0f, 1.0f, 0.0f},            // softsign = 0.99984 -> 0
        {ActivationTypes::Selu, -0.841796875f, -1.0f, 0.0f},         // selu = -1.00030 -> -2
        {ActivationTypes::HardSigmoid, 2.5f, 1.0f, 0.0f},            // hardsigmoid = 0.99988 -> 0
    };
    return cases;
}

std::string MathFloorF16Test::getTestCaseName(const testing::TestParamInfo<MathFloorF16Params>& obj) {
    const auto& [test_case, device] = obj.param;
    std::ostringstream result;
    result << "MathOp=" << activationNames[test_case.math_type] << "_input=" << test_case.input;
    if (test_case.pre_multiplier != 0.0f) {
        result << "_preMultiplier=" << test_case.pre_multiplier;
    }
    result << "_targetDevice=" << device;
    return result.str();
}

void MathFloorF16Test::SetUp() {
    std::tie(test_case, targetDevice) = GetParam();
    configuration[ov::hint::inference_precision.name()] = ov::element::f16;

    const ov::Shape shape{1};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
    ov::Output<ov::Node> math_input = param;
    if (test_case.pre_multiplier != 0.0f) {
        auto multiplier = ov::op::v0::Constant::create(ov::element::f16, shape, {test_case.pre_multiplier});
        math_input = std::make_shared<ov::op::v1::Multiply>(param, multiplier);
    }
    // HardSigmoid (alpha, beta) and Selu (alpha, lambda) constants; ignored by the other ops.
    const std::vector<float> constants = test_case.math_type == ov::test::utils::ActivationTypes::HardSigmoid
                                             ? std::vector<float>{0.2f, 0.5f}
                                             : std::vector<float>{1.6732632423543772f, 1.0507009873554805f};
    auto math =
        ov::test::utils::make_activation(math_input, ov::element::f16, test_case.math_type, ov::Shape{}, constants);
    auto floor = std::make_shared<ov::op::v0::Floor>(math);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(floor)},
                                           ov::ParameterVector{param},
                                           "MathFloorF16");
}

void MathFloorF16Test::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto tensor = ov::Tensor(ov::element::f16, targetInputStaticShapes[0]);
    tensor.data<ov::float16>()[0] = ov::float16(test_case.input);
    inputs.insert({function->get_parameters()[0], tensor});
}

void MathFloorF16Test::check_floor_result() {
    compile_model();
    generate_inputs({ov::Shape{1}});
    infer();

    const auto output_tensor = inferRequest.get_output_tensor();
    ASSERT_EQ(output_tensor.get_element_type(), ov::element::f16);
    EXPECT_EQ(static_cast<float>(output_tensor.data<ov::float16>()[0]), test_case.expected);
}

}  // namespace test
}  // namespace ov
