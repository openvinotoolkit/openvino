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
    // Value computed in f32 -> Floor with f16 rounding kept / Floor of the pure f32 path.
    static const std::vector<MathFloorF16Case> cases = {
        {ActivationTypes::Cos, 0.0130767822265625f, 0.0f},     // 0.99991  -> 1 / 0
        {ActivationTypes::Cosh, 3.63671875f, 0.0f},            // 18.9967  -> 19 / 18
        {ActivationTypes::Sin, 1.5546875f, 0.0f},              // 0.99987  -> 1 / 0
        {ActivationTypes::Sinh, 2.998046875f, 0.0f},           // 9.99823  -> 10 / 9
        {ActivationTypes::Acos, -0.416015625f, 0.0f},          // 1.99986  -> 2 / 1
        {ActivationTypes::Acosh, 1.54296875f, 0.0f},           // 0.99990  -> 1 / 0
        {ActivationTypes::Asin, 1.54296875f, 0.54541015625f},  // asin(0.8415508) = 1.00016 -> 0 / 1
        {ActivationTypes::Asinh, -3.62890625f, 0.0f},          // -2.00054 -> -2 / -3
        {ActivationTypes::Atan, -1.55859375f, 0.0f},           // -1.00035 -> -1 / -2
        {ActivationTypes::Atanh, -0.76171875f, 0.0f},          // -1.00030 -> -1 / -2
        {ActivationTypes::Tan, 1.2490234375f, 0.0f},           // 2.99978  -> 3 / 2
        {ActivationTypes::Sign, 0.0001f, 0.0001f},             // sign(1.0e-8) = 1 -> 0 / 1
        {ActivationTypes::SoftPlus, 4.9921875f, 0.0f},         // 4.99896  -> 5 / 4
        {ActivationTypes::SoftSign, 6304.0f, 0.0f},            // 0.99984  -> 1 / 0
        {ActivationTypes::Selu, -0.841796875f, 0.0f},          // -1.00030 -> -1 / -2
        {ActivationTypes::HardSigmoid, 2.5f, 0.0f},            // 0.99988  -> 1 / 0
    };
    return cases;
}

std::string MathFloorF16Test::getTestCaseName(const testing::TestParamInfo<MathFloorF16Params>& obj) {
    const auto& [test_case, inference_precision, device] = obj.param;
    std::ostringstream result;
    result << "MathOp=" << activationNames[test_case.math_type] << "_input=" << test_case.input;
    if (test_case.pre_multiplier != 0.0f) {
        result << "_preMultiplier=" << test_case.pre_multiplier;
    }
    result << "_inferencePrecision=" << inference_precision << "_targetDevice=" << device;
    return result.str();
}

void MathFloorF16Test::SetUp() {
    ov::element::Type inference_precision_hint;
    std::tie(test_case, inference_precision_hint, targetDevice) = GetParam();
    configuration[ov::hint::inference_precision.name()] = inference_precision_hint;

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

void MathFloorF16Test::compile_model() {
    SubgraphBaseStaticTest::compile_model();
    // Plugin test configs may lower the reference model to f32, which would drop the f16 rounding under test.
    convert_precisions.clear();
}

}  // namespace test
}  // namespace ov
