// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "pass/simplify_activation_shapes.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/symbolic_transformations/dereshape_matmul.hpp"

namespace {

using namespace ov::op;

class GGUFActivationShapes : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::frontend::gguf::pass::SimplifyActivationShapes>();
        comparator.enable(FunctionsComparator::ACCURACY);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

std::shared_ptr<ov::Model> residual_model(bool compact) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 8});
    const auto axis = v0::Constant::create(ov::element::i64, {1}, {1});
    const auto w = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(x, w, false, true);
    ov::Output<ov::Node> value = fc;
    ov::Output<ov::Node> residual = x;
    if (!compact) {
        value = std::make_shared<v0::Unsqueeze>(value, axis);
        residual = std::make_shared<v0::Unsqueeze>(residual, axis);
    }
    value = std::make_shared<v4::Swish>(std::make_shared<v1::Add>(value, residual));
    if (compact)
        value = std::make_shared<v0::Unsqueeze>(value, axis);
    return std::make_shared<ov::Model>(ov::OutputVector{value}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, KeepsResidualPathAtRankThree) {
    model = residual_model(false);
    model_ref = residual_model(true);
}

TEST_F(GGUFActivationShapes, KeepsReshapeFeedingMatMul) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 8});
    const auto target = v0::Constant::create(ov::element::i64, {4}, {2, 1, -1, 8});
    const auto reshape = std::make_shared<v1::Reshape>(std::make_shared<v4::Swish>(x), target, false);
    const auto w = v0::Constant::create(ov::element::f32, {4, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(reshape, w, false, true);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{x});
}

std::shared_ptr<ov::Model> get_matmul_output_consumers_model() {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, 1, -1, 8});
    const auto w = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(x, w, false, true);
    const auto add = std::make_shared<v1::Add>(fc, x);
    return std::make_shared<ov::Model>(
        ov::OutputVector{std::make_shared<v4::Swish>(add), std::make_shared<v0::Exp>(add)},
        ov::ParameterVector{x});
}

std::shared_ptr<ov::Model> get_matmul_output_consumers_model_ref() {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, 1, -1, 8});
    const auto w = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(x, w, false, true);
    const auto add = std::make_shared<v1::Add>(fc, x);
    const auto axis = v0::Constant::create(ov::element::i64, {1}, {1});
    const auto squeezed = std::make_shared<v0::Squeeze>(add, axis);
    return std::make_shared<ov::Model>(
        ov::OutputVector{std::make_shared<v0::Unsqueeze>(std::make_shared<v4::Swish>(squeezed), axis),
                         std::make_shared<v0::Unsqueeze>(std::make_shared<v0::Exp>(squeezed), axis)},
        ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, KeepsMatMulOutputConsumersAtRankFour) {
    model = get_matmul_output_consumers_model();
    model_ref = get_matmul_output_consumers_model_ref();
    comparator.enable(FunctionsComparator::CONSUMERS_COUNT);
}

std::shared_ptr<ov::Model> matmul_rms_model(bool with_scale, bool with_residual, const ov::PartialShape& input_shape) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, input_shape);
    const auto target = v0::Constant::create(ov::element::i64, {4}, {-1, 1, 1, 8});
    const auto expanded = std::make_shared<v1::Reshape>(x, target, false);
    const auto weights = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(expanded, weights, false, true);
    const auto axes = v0::Constant::create(ov::element::i64, {1}, {-1});
    const auto eps = v0::Constant::create(ov::element::f32, {}, {1e-5f});
    const auto gamma = v0::Constant::create(ov::element::f32, {8}, {0.5f});
    ov::pass::NodeRegistry reg;
    ov::Output<ov::Node> value =
        ov::decomposition::rms_norm(reg, fc, axes, eps, with_scale ? gamma->output(0) : ov::Output<ov::Node>{});
    if (with_residual)
        value = std::make_shared<v1::Add>(value, expanded);
    return std::make_shared<ov::Model>(ov::OutputVector{value}, ov::ParameterVector{x});
}

std::shared_ptr<ov::Model> matmul_rms_model_ref(bool with_scale,
                                                bool with_residual,
                                                const ov::PartialShape& input_shape) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, input_shape);
    const auto target = v0::Constant::create(ov::element::i64, {4}, {-1, 1, 1, 8});
    const auto expanded = std::make_shared<v1::Reshape>(x, target, false);
    const auto weights = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(expanded, weights, false, true);
    const auto gamma = v0::Constant::create(ov::element::f32, {8}, {0.5f});
    ov::Output<ov::Node> value = with_scale
                                     ? std::make_shared<ov::op::internal::RMS>(fc, gamma, 1e-5f, ov::element::f32)
                                     : std::make_shared<ov::op::internal::RMS>(fc, 1e-5f, ov::element::f32);
    if (with_residual)
        value = std::make_shared<v1::Add>(value, expanded);
    return std::make_shared<ov::Model>(ov::OutputVector{value}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, FusesMatMulRmsNormWithScaleAndResidual) {
    model = matmul_rms_model(true, true, {-1, 1, 8});
    model_ref = matmul_rms_model_ref(true, true, {-1, 1, 8});
    manager.register_pass<ov::pass::RMSFusion>(false, true, true);
    manager.register_pass<ov::pass::EliminateSqueeze>();
    manager.register_pass<ov::pass::DeReshapeFullyConnected>();
}

TEST_F(GGUFActivationShapes, FusesStaticMatMulRmsNormWithScaleAndResidual) {
    model = matmul_rms_model(true, true, {2, 1, 8});
    model_ref = matmul_rms_model_ref(true, true, {2, 1, 8});
    manager.register_pass<ov::pass::RMSFusion>(false, true, true);
    manager.register_pass<ov::pass::EliminateSqueeze>();
    manager.register_pass<ov::pass::DeReshapeFullyConnected>();
}

TEST_F(GGUFActivationShapes, FusesMatMulRmsNormWithoutScale) {
    model = matmul_rms_model(false, false, {2, 1, 8});
    model_ref = matmul_rms_model_ref(false, false, {2, 1, 8});
    manager.register_pass<ov::pass::RMSFusion>(false, true, true);
    manager.register_pass<ov::pass::EliminateSqueeze>();
    manager.register_pass<ov::pass::DeReshapeFullyConnected>();
}

TEST_F(GGUFActivationShapes, FusesDynamicMatMulRmsNormWithoutScaleWithResidual) {
    model = matmul_rms_model(false, true, {-1, 1, 8});
    model_ref = matmul_rms_model_ref(false, true, {-1, 1, 8});
    manager.register_pass<ov::pass::RMSFusion>(false, true, true);
    manager.register_pass<ov::pass::EliminateSqueeze>();
    manager.register_pass<ov::pass::DeReshapeFullyConnected>();
}

std::shared_ptr<ov::Model> rotary_model(bool compact,
                                        int64_t split_axis,
                                        int64_t concat_axis,
                                        bool split_input = false,
                                        bool split_cos = false) {
    const bool split_last = split_axis == -1 || split_axis == 3;
    const auto x_shape = split_cos    ? ov::PartialShape{2, 1, 2, 8}
                         : split_last ? ov::PartialShape{2, 1, -1, 8}
                                      : ov::PartialShape{2, 1, 8, 4};
    const auto table_shape = split_cos    ? ov::PartialShape{2, 1, 2, 4}
                             : split_last ? ov::PartialShape{2, 1, -1, 4}
                                          : ov::PartialShape{2, 1, 4, 4};
    const auto input =
        std::make_shared<v0::Parameter>(ov::element::f32, split_input ? ov::PartialShape{2, 2, -1, 8} : x_shape);
    ov::Output<ov::Node> x = input;
    if (split_input) {
        x = std::make_shared<v1::VariadicSplit>(input,
                                                v0::Constant::create(ov::element::i64, {}, {1}),
                                                v0::Constant::create(ov::element::i64, {2}, {1, 1}))
                ->output(0);
    }
    const auto cos =
        std::make_shared<v0::Parameter>(ov::element::f32, split_cos ? ov::PartialShape{2, 1, 4, 4} : table_shape);
    const auto sin = std::make_shared<v0::Parameter>(ov::element::f32, table_shape);
    const auto split = std::make_shared<v1::VariadicSplit>(x,
                                                           v0::Constant::create(ov::element::i64, {}, {split_axis}),
                                                           v0::Constant::create(ov::element::i64, {2}, {4, 4}));
    const auto axis = v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> first = split->output(0);
    ov::Output<ov::Node> second = split->output(1);
    ov::Output<ov::Node> cos_value = cos;
    ov::Output<ov::Node> sin_value = sin;
    if (split_cos) {
        cos_value = std::make_shared<v1::VariadicSplit>(cos,
                                                        v0::Constant::create(ov::element::i64, {}, {2}),
                                                        v0::Constant::create(ov::element::i64, {2}, {2, 2}))
                        ->output(0);
    }
    if (compact) {
        first = std::make_shared<v0::Squeeze>(first, axis);
        second = std::make_shared<v0::Squeeze>(second, axis);
        cos_value = std::make_shared<v0::Squeeze>(cos_value, axis);
        sin_value = std::make_shared<v0::Squeeze>(sin_value, axis);
    }
    const auto first_cos =
        split_cos ? std::make_shared<v1::Multiply>(cos_value, first) : std::make_shared<v1::Multiply>(first, cos_value);
    const auto second_sin = std::make_shared<v1::Multiply>(second, sin_value);
    const auto negative =
        std::make_shared<v1::Multiply>(second_sin, v0::Constant::create(ov::element::f32, {}, {-1.f}));
    ov::Output<ov::Node> result_first = std::make_shared<v1::Add>(first_cos, negative);
    const auto second_cos = split_cos ? std::make_shared<v1::Multiply>(cos_value, second)
                                      : std::make_shared<v1::Multiply>(second, cos_value);
    const auto first_sin = std::make_shared<v1::Multiply>(first, sin_value);
    ov::Output<ov::Node> result_second = std::make_shared<v1::Add>(second_cos, first_sin);
    if (compact) {
        result_first = std::make_shared<v0::Unsqueeze>(result_first, axis);
        result_second = std::make_shared<v0::Unsqueeze>(result_second, axis);
    }
    const auto output = std::make_shared<v0::Concat>(ov::OutputVector{result_first, result_second}, concat_axis);
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input, cos, sin});
}

std::shared_ptr<ov::Model> rotary_model_ref(bool split_input = false, bool split_cos = false) {
    const auto input = std::make_shared<v0::Parameter>(ov::element::f32,
                                                       split_input ? ov::PartialShape{2, 2, -1, 8}
                                                       : split_cos ? ov::PartialShape{2, 1, 2, 8}
                                                                   : ov::PartialShape{2, 1, -1, 8});
    ov::Output<ov::Node> x = input;
    if (split_input) {
        x = std::make_shared<v1::VariadicSplit>(input,
                                                v0::Constant::create(ov::element::i64, {}, {1}),
                                                v0::Constant::create(ov::element::i64, {2}, {1, 1}))
                ->output(0);
    }
    const auto cos =
        std::make_shared<v0::Parameter>(ov::element::f32,
                                        split_cos ? ov::PartialShape{2, 1, 4, 4} : ov::PartialShape{2, 1, -1, 4});
    const auto sin =
        std::make_shared<v0::Parameter>(ov::element::f32,
                                        split_cos ? ov::PartialShape{2, 1, 2, 4} : ov::PartialShape{2, 1, -1, 4});
    ov::Output<ov::Node> cos_value = cos;
    if (split_cos) {
        cos_value = std::make_shared<v1::VariadicSplit>(cos,
                                                        v0::Constant::create(ov::element::i64, {}, {2}),
                                                        v0::Constant::create(ov::element::i64, {2}, {2, 2}))
                        ->output(0);
    }
    ov::op::internal::RoPE::Config config;
    config.rotary_ndims = 8;
    config.cos_sin_ndims = 4;
    const auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{x, cos_value, sin}, config);
    return std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, cos, sin});
}

TEST_F(GGUFActivationShapes, FusesSingletonHeadRotaryWithNegativeAxes) {
    model = rotary_model(false, -1, -1);
    model_ref = rotary_model_ref();
    manager.register_pass<ov::pass::RoPEFusion>();
    comparator.disable(FunctionsComparator::ACCURACY);
    disable_rt_info_check();
}

TEST_F(GGUFActivationShapes, FusesSingletonHeadRotaryWithPositiveAxes) {
    model = rotary_model(false, 3, 3);
    model_ref = rotary_model_ref();
    manager.register_pass<ov::pass::RoPEFusion>();
    comparator.disable(FunctionsComparator::ACCURACY);
    disable_rt_info_check();
}

TEST_F(GGUFActivationShapes, FusesSingletonHeadRotaryFedByAnotherSplit) {
    model = rotary_model(false, -1, -1, true);
    model_ref = rotary_model_ref(true);
    manager.register_pass<ov::pass::RoPEFusion>();
    comparator.disable(FunctionsComparator::ACCURACY);
    disable_rt_info_check();
}

TEST_F(GGUFActivationShapes, FusesSingletonHeadRotaryWithSplitCosine) {
    model = rotary_model(false, -1, -1, false, true);
    model_ref = rotary_model_ref(false, true);
    manager.register_pass<ov::pass::RoPEFusion>();
    comparator.disable(FunctionsComparator::ACCURACY);
    disable_rt_info_check();
}

TEST_F(GGUFActivationShapes, SimplifiesRotaryArithmeticWithNonLastConcatAxis) {
    model = rotary_model(false, -1, 2);
    model_ref = rotary_model(true, -1, 2);
}

TEST_F(GGUFActivationShapes, SimplifiesRotaryArithmeticWithNonLastSplitAxis) {
    model = rotary_model(false, 2, -1);
    model_ref = rotary_model(true, 2, -1);
}

TEST_F(GGUFActivationShapes, KeepsUnrelatedMatMulElementwiseConsumersAtRankFour) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 3, 8});
    const auto weights = v0::Constant::create(ov::element::f32, {8, 8}, {0.25f});
    const auto fc = std::make_shared<v0::MatMul>(x, weights, false, true);
    const auto scale = v0::Constant::create(ov::element::f32, {}, {0.5f});
    const auto exponent = v0::Constant::create(ov::element::f32, {}, {2.0f});
    model = std::make_shared<ov::Model>(
        ov::OutputVector{std::make_shared<v1::Multiply>(fc, scale), std::make_shared<v1::Power>(fc, exponent)},
        ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, PreservesOtherOutputsOfReshapeProducer) {
    const auto make_model = [](bool compact) {
        const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 8});
        const auto split = std::make_shared<v1::VariadicSplit>(x,
                                                               v0::Constant::create(ov::element::i64, {}, {1}),
                                                               v0::Constant::create(ov::element::i64, {2}, {2, 3}));
        ov::Output<ov::Node> second = split->output(1);
        if (compact) {
            second = std::make_shared<v0::Unsqueeze>(second, v0::Constant::create(ov::element::i64, {1}, {1}));
        } else {
            second =
                std::make_shared<v1::Reshape>(second, v0::Constant::create(ov::element::i64, {4}, {1, 1, 3, 8}), false);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{split->output(0), second}, ov::ParameterVector{x});
    };
    model = make_model(false);
    model_ref = make_model(true);
}

TEST_F(GGUFActivationShapes, RemovesOnlySymbolProvenIdentityReshapes) {
    const auto make_model = [](bool compact) {
        const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 8});
        ov::Output<ov::Node> value = x;
        if (!compact) {
            const auto expanded = std::make_shared<v0::Unsqueeze>(x, v0::Constant::create(ov::element::i64, {1}, {1}));
            const auto shape = std::make_shared<v3::ShapeOf>(expanded);
            const auto target = std::make_shared<v8::Gather>(shape,
                                                             v0::Constant::create(ov::element::i64, {3}, {0, 2, 3}),
                                                             v0::Constant::create(ov::element::i64, {}, {0}));
            value = std::make_shared<v1::Reshape>(expanded, target, false);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v4::Swish>(value)},
                                           ov::ParameterVector{x});
    };
    model = make_model(false);
    model_ref = make_model(true);
}

TEST_F(GGUFActivationShapes, LeavesIndependentDynamicDimensionsInReshape) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 8});
    const auto shape = std::make_shared<v0::Parameter>(ov::element::i64, ov::Shape{3});
    const auto reshape = std::make_shared<v1::Reshape>(x, shape, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{x, shape});
    comparator.disable(FunctionsComparator::ACCURACY);
}

TEST_F(GGUFActivationShapes, LeavesNonSingletonBroadcastAxisUnchanged) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, -1, 8});
    const auto bias = v0::Constant::create(ov::element::f32, {8}, {0.125f});
    const auto add = std::make_shared<v1::Add>(x, bias);
    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, LeavesSelectedTokenAxisDynamic) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 8});
    model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<v4::Swish>(x)}, ov::ParameterVector{x});
}

TEST_F(GGUFActivationShapes, LeavesUnprovenSingletonAxisWithoutBroadcasting) {
    const auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 3, 8});
    const auto y = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{2, 1, 3, 8});
    const auto add = std::make_shared<v1::Add>(x, y, ov::op::AutoBroadcastType::NONE);
    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x, y});
}

}  // namespace
