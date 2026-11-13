// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/collapse_unqdq.hpp"
#include "npuw_transformations/conv_to_matmul.hpp"
#include "npuw_transformations/drop_zp_subtract.hpp"

#include <algorithm>

#include <gtest/gtest.h>

#include <cmath>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

std::shared_ptr<ov::Model> build_unqdq_model(const ov::element::Type& input_type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 4});
    auto input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255.0f});
    auto fake_quantize =
        std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
    auto quantized_convert = std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16);
    auto dequantized_convert = std::make_shared<ov::op::v0::Convert>(quantized_convert, ov::element::f32);
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u16, ov::Shape{}, {128}),
        ov::element::f32);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(dequantized_convert, zero_point);
    auto scale = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.1f}),
        ov::element::f32);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    multiply->set_friendly_name("unqdq_out");
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(multiply)},
                                       ov::ParameterVector{input},
                                       "unqdq_model");
}

std::shared_ptr<ov::Model> build_zero_point_subtract_model(bool zero_point_is_zero) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u16, ov::Shape{4}, zero_point_is_zero ? std::vector<uint16_t>{0, 0, 0, 0}
                                                                                         : std::vector<uint16_t>{0, 1, 0, 0}),
        ov::element::f32);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(input, zero_point);
    subtract->set_friendly_name("zp_subtract");
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(subtract)},
                                       ov::ParameterVector{input},
                                       "zero_point_subtract_model");
}

std::shared_ptr<ov::Model> build_conv_to_matmul_model(bool per_output_channel_scale,
                                                      bool with_output_transpose,
                                                      bool with_input_transpose = true,
                                                      bool use_scale_parameter_chain = false) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        with_input_transpose ? ov::Shape{1, 1, 3, 4}
                                                                             : ov::Shape{1, 4, 1, 3});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{2, 4, 1, 1});
    std::shared_ptr<ov::Node> scale_source;
    std::shared_ptr<ov::op::v0::Parameter> scale_parameter;
    if (use_scale_parameter_chain) {
        scale_parameter = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32,
            per_output_channel_scale ? ov::Shape{2} : ov::Shape{8});
        auto scale_reshape_1 = std::make_shared<ov::op::v1::Reshape>(
            scale_parameter,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{per_output_channel_scale ? 2 : 8}),
            false);
        scale_source = std::make_shared<ov::op::v1::Reshape>(
            scale_reshape_1,
            ov::op::v0::Constant::create(ov::element::i32,
                                         ov::Shape{4},
                                         per_output_channel_scale ? std::vector<int32_t>{2, 1, 1, 1}
                                                                  : std::vector<int32_t>{2, 4, 1, 1}),
            false);
    } else {
        scale_source = std::make_shared<ov::op::v1::Reshape>(
            ov::op::v0::Constant::create(ov::element::f32,
                                         per_output_channel_scale ? ov::Shape{2} : ov::Shape{8},
                                         per_output_channel_scale ? std::vector<float>{0.25f, 0.5f}
                                                                  : std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f,
                                                                                       1.25f, 1.5f, 1.75f, 2.0f}),
            ov::op::v0::Constant::create(ov::element::i32,
                                         ov::Shape{4},
                                         per_output_channel_scale ? std::vector<int32_t>{2, 1, 1, 1}
                                                                  : std::vector<int32_t>{2, 4, 1, 1}),
            false);
    }

    std::shared_ptr<ov::Node> conv_input = data;
    if (with_input_transpose) {
        conv_input = std::make_shared<ov::op::v1::Transpose>(
            data,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2}));
    }
    auto converted_weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(converted_weights, scale_source);
    auto convolution = std::make_shared<ov::op::v1::Convolution>(conv_input,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    convolution->set_friendly_name("conv_to_matmul_out");

    std::shared_ptr<ov::Node> output = convolution;
    if (with_output_transpose) {
        output = std::make_shared<ov::op::v1::Transpose>(
            convolution,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1}));
        output->set_friendly_name("conv_to_matmul_output");
    }

    ov::ParameterVector parameters{data, weights};
    if (scale_parameter != nullptr) {
        parameters.push_back(scale_parameter);
    }

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                       parameters,
                                       "conv_to_matmul_model");
}

std::shared_ptr<ov::Model> build_conv_to_matmul_and_unqdq_model() {
    const auto conv_model = build_conv_to_matmul_model(true, true);
    const auto unqdq_model = build_unqdq_model();

    ov::ResultVector results;
    for (const auto& result : conv_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }
    for (const auto& result : unqdq_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }

    ov::ParameterVector parameters = conv_model->get_parameters();
    const auto unqdq_parameters = unqdq_model->get_parameters();
    parameters.insert(parameters.end(), unqdq_parameters.begin(), unqdq_parameters.end());

    return std::make_shared<ov::Model>(results, parameters, "conv_to_matmul_and_unqdq_model");
}

void expect_conv_to_matmul_and_unqdq_rewritten(const std::shared_ptr<ov::Model>& model) {
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
}

TEST(CollapseUNQDQPassTest, DropsEntireUnqdqChainFromModel) {
    const auto model = build_unqdq_model();
    ASSERT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 1u);
    ASSERT_EQ(count_ops<ov::op::v1::Multiply>(model), 1u);

    ov::npuw::CollapseUNQDQ pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 0u);

    const auto output_node = model->get_results().front()->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(output_node));
}

TEST(CollapseUNQDQPassTest, PreservesOriginalOutputElementType) {
    const auto model = build_unqdq_model(ov::element::f16);

    ov::npuw::CollapseUNQDQ pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_element_type(), ov::element::f32);
}

TEST(DropZPSubtractPassTest, DropsSubtractWhenZeroPointIsAllZeros) {
    const auto model = build_zero_point_subtract_model(true);

    ov::npuw::DropZPSubtract pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 0u);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(model->get_results().front()->input_value(0).get_node_shared_ptr()));
}

TEST(DropZPSubtractPassTest, KeepsSubtractWhenZeroPointIsNotAllZeros) {
    const auto model = build_zero_point_subtract_model(false);

    ov::npuw::DropZPSubtract pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 1u);
}

TEST(CollapseUNQDQPassTest, ConvToMatMulAndUNQDQOrderDoesNotMatter) {
    {
        const auto model = build_conv_to_matmul_and_unqdq_model();
        ov::npuw::ConvToMatMul conv_to_matmul;
        ov::npuw::CollapseUNQDQ collapse_unqdq;

        conv_to_matmul.run_on_model(model);
        collapse_unqdq.run_on_model(model);

        expect_conv_to_matmul_and_unqdq_rewritten(model);
    }

    {
        const auto model = build_conv_to_matmul_and_unqdq_model();
        ov::npuw::ConvToMatMul conv_to_matmul;
        ov::npuw::CollapseUNQDQ collapse_unqdq;

        collapse_unqdq.run_on_model(model);
        conv_to_matmul.run_on_model(model);

        expect_conv_to_matmul_and_unqdq_rewritten(model);
    }
}

constexpr float kOverflowRange = 950796.0f;
constexpr float kEps = 1e-7f;

ov::Output<ov::Node> add_unqdq(const ov::Output<ov::Node>& input, float range, bool swapped_multiply = false) {
    auto fake_quantize = std::make_shared<ov::op::v0::FakeQuantize>(
        input,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-range}),
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {range}),
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}),
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {65535.0f}),
        65536);
    auto quantized_convert = std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16);
    auto dequantized_convert = std::make_shared<ov::op::v0::Convert>(quantized_convert, ov::element::f32);
    auto subtract =
        std::make_shared<ov::op::v1::Subtract>(dequantized_convert,
                                               ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {32768.0f}));
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {range / 32768.0f});
    return swapped_multiply ? std::make_shared<ov::op::v1::Multiply>(scale, subtract)
                            : std::make_shared<ov::op::v1::Multiply>(subtract, scale);
}

ov::Output<ov::Node> rescale(const ov::Output<ov::Node>& input, int k) {
    return std::make_shared<ov::op::v1::Multiply>(
        input,
        ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {std::ldexp(1.0f, -k)}));
}

std::shared_ptr<ov::Node> add_l2_normalize(const ov::Output<ov::Node>& x, bool decomposed, int k) {
    const float eps = kEps * std::ldexp(1.0f, -2 * k);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    if (!decomposed) {
        return std::make_shared<ov::op::v0::NormalizeL2>(x, axes, eps, ov::op::EpsMode::ADD);
    }
    auto square =
        std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f}));
    auto sum = std::make_shared<ov::op::v1::ReduceSum>(square, axes, true);
    auto add =
        std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {eps}));
    return std::make_shared<ov::op::v1::Divide>(x, std::make_shared<ov::op::v0::Sqrt>(add));
}

// Mirrors an MLP feeding a post-norm: Mul_1 -> FQ -> down_proj -> FQ -> L2 normalization
struct MlpOptions {
    float mul_range = kOverflowRange / 4;
    float down_range = kOverflowRange;
    bool decomposed_l2 = false;
    bool residual_add = false;
    bool reshape_before_norm = false;
    bool swapped_multiply = false;
    ov::element::Type type = ov::element::f32;
    // Reference graph state
    int k = 0;               // exponent RescaleFP16OverflowQDQ applied, 0 if it left the model alone
    bool collapsed = false;  // CollapseUNQDQ ran; RescaleFP16OverflowQDQ collapses the region itself when k != 0
};

std::shared_ptr<ov::Model> build_overflow_mlp_model(const MlpOptions& o) {
    auto up = std::make_shared<ov::op::v0::Parameter>(o.type, ov::Shape{1, 4});
    auto gate = std::make_shared<ov::op::v0::Parameter>(o.type, ov::Shape{1, 4});
    ov::ParameterVector params{up, gate};
    // The first overflowing FQ in topological order decides where the factor goes
    const bool scale_mul = o.k && o.mul_range > 65504.0f;
    const bool scale_down = o.k && !scale_mul;
    const auto qdq = [&](const ov::Output<ov::Node>& input, float range) -> ov::Output<ov::Node> {
        if (!o.collapsed && !o.k) {
            return add_unqdq(input, range, o.swapped_multiply);
        }
        if (input.get_element_type() != ov::element::f32) {
            return std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
        }
        return input;
    };

    auto mul = std::make_shared<ov::op::v1::Multiply>(scale_mul ? rescale(up, o.k) : up, gate);
    ov::Output<ov::Node> down_input = qdq(mul, o.mul_range);
    auto down = std::make_shared<ov::op::v0::MatMul>(
        scale_down ? rescale(down_input, o.k) : down_input,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 1.0f)));
    ov::Output<ov::Node> x = qdq(down, o.down_range);
    if (o.residual_add) {
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
        params.push_back(residual);
        x = std::make_shared<ov::op::v1::Add>(x, residual);
    }
    if (o.reshape_before_norm) {
        x = std::make_shared<ov::op::v1::Reshape>(x,
                                                  ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 4}),
                                                  false);
    }
    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(add_l2_normalize(x, o.decomposed_l2, o.k))},
        params,
        "overflow_mlp_model");
}

template <class F>
std::shared_ptr<ov::Model> build_mlp(F&& set) {
    MlpOptions options;
    set(options);
    return build_overflow_mlp_model(options);
}

class RescaleFP16OverflowQDQTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    }
};

TEST_F(RescaleFP16OverflowQDQTest, ScalesRegionEndingInNormalizeL2) {
    model = build_overflow_mlp_model(MlpOptions{});
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    // 950796 / 2^5 = 29712 is the first value under the 32768 target; eps follows with 2^-10
    model_ref = build_mlp([](MlpOptions& o) {
        o.k = 5;
    });
}

TEST_F(RescaleFP16OverflowQDQTest, ScalesRegionEndingInDecomposedL2Normalize) {
    model = build_mlp([](MlpOptions& o) {
        o.decomposed_l2 = true;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    model_ref = build_mlp([](MlpOptions& o) {
        o.decomposed_l2 = true;
        o.k = 5;
    });
}

TEST_F(RescaleFP16OverflowQDQTest, ScalesSwappedDequantizeMultiply) {
    model = build_mlp([](MlpOptions& o) {
        o.swapped_multiply = true;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    model_ref = build_mlp([](MlpOptions& o) {
        o.swapped_multiply = true;
        o.k = 5;
    });
}

TEST_F(RescaleFP16OverflowQDQTest, SecondRunKeepsModel) {
    model = build_overflow_mlp_model(MlpOptions{});
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    model_ref = build_mlp([](MlpOptions& o) {
        o.k = 5;
    });
}

TEST_F(RescaleFP16OverflowQDQTest, ScalesMatMulInputThenCollapses) {
    model = build_mlp([](MlpOptions& o) {
        o.mul_range = 100.0f;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    manager.register_pass<ov::npuw::CollapseUNQDQ>();
    model_ref = build_mlp([](MlpOptions& o) {
        o.mul_range = 100.0f;
        o.k = 5;
        o.collapsed = true;
    });
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsRegionWithResidualAdd) {
    model = build_mlp([](MlpOptions& o) {
        o.residual_add = true;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsRegionWithReshapeBeforeNorm) {
    model = build_mlp([](MlpOptions& o) {
        o.reshape_before_norm = true;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsRegionWithoutMatMul) {
    auto up = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto gate = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto mul = std::make_shared<ov::op::v1::Multiply>(up, gate);
    auto norm = add_l2_normalize(add_unqdq(mul, kOverflowRange), false, 0);
    model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(norm)},
                                        ov::ParameterVector{up, gate},
                                        "no_matmul_model");
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsRangesWithinF16) {
    model = build_mlp([](MlpOptions& o) {
        o.mul_range = 15000.0f;
        o.down_range = 60000.0f;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsFactorThatRoundsToZero) {
    // 2^40 needs 2^-25, which is below the smallest f16 subnormal
    model = build_mlp([](MlpOptions& o) {
        o.mul_range = 0x1p40f;
        o.down_range = 60000.0f;
        o.type = ov::element::f16;
    });
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsRequantizedDequantizeOutput) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto norm = add_l2_normalize(add_unqdq(add_unqdq(input, 100.0f), kOverflowRange), false, 0);
    model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(norm)},
                                        ov::ParameterVector{input},
                                        "requantize_model");
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsScaledPowerExponent) {
    // The producer also feeds the exponent of a sum of squares, so scaling it would change the power
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto two =
        std::make_shared<ov::op::v1::Multiply>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}),
                                               ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f}));
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto sum = std::make_shared<ov::op::v1::ReduceSum>(std::make_shared<ov::op::v1::Power>(y, two), axes, true);
    auto add =
        std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {kEps}));
    auto norm_y = std::make_shared<ov::op::v1::Divide>(y, std::make_shared<ov::op::v0::Sqrt>(add));
    auto two_2d =
        std::make_shared<ov::op::v1::Reshape>(add_unqdq(two, kOverflowRange),
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}),
                                              false);
    auto norm_two = add_l2_normalize(two_2d, false, 0);
    model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(norm_y), std::make_shared<ov::op::v0::Result>(norm_two)},
        ov::ParameterVector{y},
        "power_exponent_model");
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    manager.register_pass<ov::npuw::CollapseUNQDQ>();
    model_ref = model->clone();
    ov::pass::Manager collapse;
    collapse.register_pass<ov::npuw::CollapseUNQDQ>();
    collapse.run_passes(model_ref);
}

TEST_F(RescaleFP16OverflowQDQTest, KeepsMatMulWithDequantizedInput) {
    // A factor on the MatMul input would complete the input chain, and CollapseUNQDQ would drop it
    const auto build = [](bool collapsed) {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
        auto fake_quantize = std::make_shared<ov::op::v0::FakeQuantize>(
            input,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-100.0f}),
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {100.0f}),
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}),
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {65535.0f}),
            65536);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(
            std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16),
            ov::element::f32);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(
            dequantized,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {32768.0f}));
        ov::Output<ov::Node> down = std::make_shared<ov::op::v0::MatMul>(
            subtract,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 4}, std::vector<float>(16, 1.0f)));
        if (!collapsed) {
            down = add_unqdq(down, kOverflowRange);
        }
        return std::make_shared<ov::Model>(
            ov::ResultVector{std::make_shared<ov::op::v0::Result>(add_l2_normalize(down, false, 0))},
            ov::ParameterVector{input},
            "dequantized_matmul_model");
    };
    model = build(false);
    manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
    manager.register_pass<ov::npuw::CollapseUNQDQ>();
    model_ref = build(true);
}

// After CollapseUNQDQ the factor must cancel in the normalization, including for activations small
// enough that eps matters, and must not create inf/NaN for f16 inputs that become subnormal
TEST(RescaleFP16OverflowQDQNumericTest, MatchesCollapseOnlyOutput) {
    const auto run = [](bool rescale, const ov::element::Type& type, float value) {
        const auto model = build_mlp([&](MlpOptions& o) {
            o.decomposed_l2 = true;
            o.type = type;
        });
        ov::pass::Manager manager;
        if (rescale) {
            manager.register_pass<ov::npuw::RescaleFP16OverflowQDQ>();
        }
        manager.register_pass<ov::npuw::CollapseUNQDQ>();
        manager.run_passes(model);
        ov::TensorVector inputs;
        for (auto&& param : model->get_parameters()) {
            inputs.emplace_back(param->get_element_type(), param->get_shape());
            if (param->get_element_type() == ov::element::f16) {
                std::fill_n(inputs.back().data<ov::float16>(), inputs.back().get_size(), ov::float16(value));
            } else {
                std::fill_n(inputs.back().data<float>(), inputs.back().get_size(), value);
            }
        }
        ov::TensorVector outputs{ov::Tensor(ov::element::f32, ov::Shape{1, 4})};
        EXPECT_TRUE(model->evaluate(outputs, inputs));
        return std::vector<float>(outputs[0].data<float>(), outputs[0].data<float>() + outputs[0].get_size());
    };
    // In f16 the factor raises the subnormal threshold by 2^k, so tiny products lose precision; they must
    // still stay finite. Values that remain normal after scaling must match.
    for (const auto& type : {ov::element::f32, ov::element::f16}) {
        for (float value : {0.0f, 1e-3f, 1e-2f, 1.0f, 200.0f}) {
            const auto expected = run(false, type, value);
            const auto actual = run(true, type, value);
            const bool exact = type == ov::element::f32 || value == 0.0f || value >= 1.0f;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                ASSERT_TRUE(std::isfinite(actual[i])) << type << " value=" << value;
                if (exact) {
                    EXPECT_NEAR(actual[i], expected[i], 1e-5f * std::fabs(expected[i])) << type << " value=" << value;
                }
            }
        }
    }
}

}  // namespace
