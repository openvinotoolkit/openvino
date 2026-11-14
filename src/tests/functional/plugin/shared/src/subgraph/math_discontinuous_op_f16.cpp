// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/math_discontinuous_op_f16.hpp"

#include <sstream>

#include "common_test_utils/node_builders/activation.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/single_op/activation.hpp"

namespace ov::test {

namespace {

std::string op_name(DiscontinuousOp op) {
    switch (op) {
    case DiscontinuousOp::Floor:
        return "Floor";
    case DiscontinuousOp::Ceiling:
        return "Ceiling";
    case DiscontinuousOp::RoundHalfToEven:
        return "RoundHalfToEven";
    case DiscontinuousOp::RoundHalfAwayFromZero:
        return "RoundHalfAwayFromZero";
    case DiscontinuousOp::Sign:
        return "Sign";
    case DiscontinuousOp::Mod:
        return "Mod";
    case DiscontinuousOp::FloorMod:
        return "FloorMod";
    }
    OPENVINO_THROW("Unsupported DiscontinuousOp");
}

std::shared_ptr<ov::Node> make_discontinuous_op(DiscontinuousOp op, const ov::Output<ov::Node>& input) {
    auto one = [] {
        return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {1.0f});
    };
    switch (op) {
    case DiscontinuousOp::Floor:
        return std::make_shared<ov::op::v0::Floor>(input);
    case DiscontinuousOp::Ceiling:
        return std::make_shared<ov::op::v0::Ceiling>(input);
    case DiscontinuousOp::RoundHalfToEven:
        return std::make_shared<ov::op::v5::Round>(input, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    case DiscontinuousOp::RoundHalfAwayFromZero:
        return std::make_shared<ov::op::v5::Round>(input, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    case DiscontinuousOp::Sign:
        return std::make_shared<ov::op::v0::Sign>(input);
    case DiscontinuousOp::Mod:
        return std::make_shared<ov::op::v1::Mod>(input, one());
    case DiscontinuousOp::FloorMod:
        return std::make_shared<ov::op::v1::FloorMod>(input, one());
    }
    OPENVINO_THROW("Unsupported DiscontinuousOp");
}

}  // namespace

const std::vector<MathDiscontinuousOpF16Case>& MathDiscontinuousOpF16Test::all_cases() {
    using ov::test::utils::ActivationTypes;
    using Op = DiscontinuousOp;
    // Each input makes the Math value land just next to the value where the discontinuous op jumps (about a quarter
    // of an f16 ulp away, e.g. 0.99976 vs 1). Rounded to f16 it becomes exactly that value, so the f16 result differs
    // from the one of the pure f32 path. Cases with a pre-multiplier either need the product rounded to f16 or, for
    // 0.0001, rely on the product underflowing to 0 in f16 but not in f32.
    static const std::vector<MathDiscontinuousOpF16Case> cases = {
        // Floor: Math value -> result with the f16 rounding kept / result of the pure f32 path.
        {ActivationTypes::Cos, Op::Floor, 0.0130767822265625f, 0.0f},     // 0.99991  -> 1 / 0
        {ActivationTypes::Cosh, Op::Floor, 3.63671875f, 0.0f},            // 18.9967  -> 19 / 18
        {ActivationTypes::Sin, Op::Floor, 1.5546875f, 0.0f},              // 0.99987  -> 1 / 0
        {ActivationTypes::Sinh, Op::Floor, 2.998046875f, 0.0f},           // 9.99823  -> 10 / 9
        {ActivationTypes::Acos, Op::Floor, -0.416015625f, 0.0f},          // 1.99986  -> 2 / 1
        {ActivationTypes::Acosh, Op::Floor, 1.54296875f, 0.0f},           // 0.99990  -> 1 / 0
        {ActivationTypes::Asin, Op::Floor, 1.54296875f, 0.54541015625f},  // asin(0.8415508) = 1.00016 -> 0 / 1
        {ActivationTypes::Asinh, Op::Floor, -3.62890625f, 0.0f},          // -2.00054 -> -2 / -3
        {ActivationTypes::Atan, Op::Floor, -1.55859375f, 0.0f},           // -1.00035 -> -1 / -2
        {ActivationTypes::Atanh, Op::Floor, -0.76171875f, 0.0f},          // -1.00030 -> -1 / -2
        {ActivationTypes::Tan, Op::Floor, 1.2490234375f, 0.0f},           // 2.99978  -> 3 / 2
        {ActivationTypes::Sign, Op::Floor, 0.0001f, 0.0001f},             // sign(1.0e-8) = 1 -> 0 / 1
        {ActivationTypes::SoftPlus, Op::Floor, 4.9921875f, 0.0f},         // 4.99896  -> 5 / 4
        {ActivationTypes::Selu, Op::Floor, 4.7578125f, 0.0f},             // 4.99940  -> 5 / 4
        {ActivationTypes::HardSigmoid, Op::Floor, 2.5f, 0.0f},            // 0.99988  -> 1 / 0

        // Ceiling: e.g. cos(3.1621) = -0.99979 -> -1 / -0.
        {ActivationTypes::Cos, Op::Ceiling, 3.162109375f, 0.0f},
        {ActivationTypes::Cosh, Op::Ceiling, 0.0220947265625f, 0.0f},
        {ActivationTypes::Sin, Op::Ceiling, 4.734375f, 0.0f},
        {ActivationTypes::Sinh, Op::Ceiling, -2.998046875f, 0.0f},
        {ActivationTypes::Acos, Op::Ceiling, -0.41650390625f, 0.0f},
        {ActivationTypes::Acosh, Op::Ceiling, 3.763671875f, 0.0f},
        {ActivationTypes::Asinh, Op::Ceiling, 3.62890625f, 0.0f},
        {ActivationTypes::Atan, Op::Ceiling, -1.556640625f, 0.0f},
        {ActivationTypes::Atanh, Op::Ceiling, 0.76171875f, 0.0f},
        {ActivationTypes::Tan, Op::Ceiling, -1.6162109375f, 0.0f},
        {ActivationTypes::SoftPlus, Op::Ceiling, 7.0f, 0.0f},
        {ActivationTypes::Selu, Op::Ceiling, 2.85546875f, 0.0f},

        // RoundHalfToEven: e.g. asin(0.4795) = 0.50008 -> 0 / 1, the f16 value is exactly 0.5.
        {ActivationTypes::Cosh, Op::RoundHalfToEven, 4.69140625f, 0.0f},
        {ActivationTypes::Sinh, Op::RoundHalfToEven, 4.890625f, 0.0f},
        {ActivationTypes::Acos, Op::RoundHalfToEven, 0.07098388671875f, 0.0f},
        {ActivationTypes::Acosh, Op::RoundHalfToEven, 6.13671875f, 0.0f},
        {ActivationTypes::Asin, Op::RoundHalfToEven, 0.4794921875f, 0.0f},
        {ActivationTypes::Asinh, Op::RoundHalfToEven, 2.12890625f, 0.0f},
        {ActivationTypes::Atan, Op::RoundHalfToEven, 0.54638671875f, 0.0f},
        {ActivationTypes::Atanh, Op::RoundHalfToEven, 0.462158203125f, 0.0f},
        {ActivationTypes::Tan, Op::RoundHalfToEven, 1.5927734375f, 0.0f},
        {ActivationTypes::Sign, Op::RoundHalfToEven, 6.103515625e-05f, 0.0001f},
        {ActivationTypes::SoftPlus, Op::RoundHalfToEven, 3.46875f, 0.0f},
        {ActivationTypes::Selu, Op::RoundHalfToEven, 7.13671875f, 0.0f},
        {ActivationTypes::HardSigmoid, Op::RoundHalfToEven, 0.0006103515625f, 0.0f},

        // RoundHalfAwayFromZero: e.g. acosh(6.1289) = 2.4994 -> 3 / 2, the f16 value is exactly 2.5.
        {ActivationTypes::Cosh, Op::RoundHalfAwayFromZero, 4.828125f, 0.0f},
        {ActivationTypes::Sin, Op::RoundHalfAwayFromZero, 6.4140625f, 4.0f},
        {ActivationTypes::Sinh, Op::RoundHalfAwayFromZero, 4.9765625f, 0.0f},
        {ActivationTypes::Acos, Op::RoundHalfAwayFromZero, 0.07098388671875f, 0.0f},
        {ActivationTypes::Acosh, Op::RoundHalfAwayFromZero, 6.12890625f, 0.0f},
        {ActivationTypes::Asinh, Op::RoundHalfAwayFromZero, 6.046875f, 0.0f},
        {ActivationTypes::Atan, Op::RoundHalfAwayFromZero, 7.02734375f, 2.0f},
        {ActivationTypes::Tan, Op::RoundHalfAwayFromZero, 1.6015625f, 0.0f},
        {ActivationTypes::Sign, Op::RoundHalfAwayFromZero, 6.103515625e-05f, 0.0001f},
        {ActivationTypes::SoftPlus, Op::RoundHalfAwayFromZero, 3.46875f, 0.0f},
        {ActivationTypes::Selu, Op::RoundHalfAwayFromZero, 7.13671875f, 0.0f},
        {ActivationTypes::HardSigmoid, Op::RoundHalfAwayFromZero, -0.0006103515625f, 0.0f},

        // Sign: the Math value (about 1.0e-8) is non-zero in f32 but underflows to 0 in f16 -> 0 / 1.
        {ActivationTypes::Sin, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Sinh, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Asin, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Asinh, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Atan, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Atanh, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Tan, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Sign, Op::Sign, 6.103515625e-05f, 0.0001f},
        {ActivationTypes::SoftSign, Op::Sign, 0.0001f, 0.0001f},
        {ActivationTypes::Selu, Op::Sign, 0.0001f, 0.0001f},

        // Mod: e.g. cos(0.0221) = 0.99976 -> 0 / 0.99976.
        {ActivationTypes::Cos, Op::Mod, 0.0220947265625f, 0.0f},
        {ActivationTypes::Cosh, Op::Mod, 3.63671875f, 0.0f},
        {ActivationTypes::Sin, Op::Mod, 4.734375f, 0.0f},
        {ActivationTypes::Sinh, Op::Mod, 2.998046875f, 0.0f},
        {ActivationTypes::Acos, Op::Mod, -0.415771484375f, 0.0f},
        {ActivationTypes::Acosh, Op::Mod, 5.03125f, 2.0f},
        {ActivationTypes::Asinh, Op::Mod, 6.81640625f, 4.0f},
        {ActivationTypes::Atan, Op::Mod, 1.556640625f, 0.0f},
        {ActivationTypes::Tan, Op::Mod, 1.892578125f, 0.0f},
        {ActivationTypes::SoftPlus, Op::Mod, 4.9921875f, 0.0f},
        {ActivationTypes::Selu, Op::Mod, 4.7578125f, 0.0f},
        {ActivationTypes::HardSigmoid, Op::Mod, 2.5f, 0.0f},

        // FloorMod: e.g. cos(0.0221) = 0.99976 -> 0 / 0.99976.
        {ActivationTypes::Cos, Op::FloorMod, 0.0220947265625f, 0.0f},
        {ActivationTypes::Cosh, Op::FloorMod, 3.63671875f, 0.0f},
        {ActivationTypes::Sin, Op::FloorMod, -4.734375f, 0.0f},
        {ActivationTypes::Sinh, Op::FloorMod, 2.998046875f, 0.0f},
        {ActivationTypes::Acos, Op::FloorMod, -0.415771484375f, 0.0f},
        {ActivationTypes::Acosh, Op::FloorMod, 5.03125f, 2.0f},
        {ActivationTypes::Asinh, Op::FloorMod, -3.62890625f, 0.0f},
        {ActivationTypes::Atan, Op::FloorMod, 1.556640625f, 0.0f},
        {ActivationTypes::Atanh, Op::FloorMod, -0.76171875f, 0.0f},
        {ActivationTypes::Tan, Op::FloorMod, 1.6162109375f, 0.0f},
        {ActivationTypes::SoftPlus, Op::FloorMod, 4.9921875f, 0.0f},
        {ActivationTypes::Selu, Op::FloorMod, 4.7578125f, 0.0f},
        {ActivationTypes::HardSigmoid, Op::FloorMod, 2.5f, 0.0f},
    };
    return cases;
}

std::string MathDiscontinuousOpF16Test::getTestCaseName(
    const testing::TestParamInfo<MathDiscontinuousOpF16Params>& obj) {
    const auto& [test_case, inference_precision, device] = obj.param;
    std::ostringstream result;
    result << "MathOp=" << activationNames[test_case.math_type] << "_Op=" << op_name(test_case.op)
           << "_input=" << test_case.input;
    if (test_case.pre_multiplier != 0.0f) {
        result << "_preMultiplier=" << test_case.pre_multiplier;
    }
    result << "_inferencePrecision=" << inference_precision << "_targetDevice=" << device;
    return result.str();
}

void MathDiscontinuousOpF16Test::SetUp() {
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
    auto discontinuous_op = make_discontinuous_op(test_case.op, math);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(discontinuous_op)},
                                           ov::ParameterVector{param},
                                           "MathDiscontinuousOpF16");
}

void MathDiscontinuousOpF16Test::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto tensor = ov::Tensor(ov::element::f16, targetInputStaticShapes[0]);
    tensor.data<ov::float16>()[0] = ov::float16(test_case.input);
    inputs.insert({function->get_parameters()[0], tensor});
}

void MathDiscontinuousOpF16Test::compile_model() {
    SubgraphBaseStaticTest::compile_model();
    // Plugin test configs may lower the reference model to f32, which would drop the f16 rounding under test.
    convert_precisions.clear();
}

}  // namespace ov::test
