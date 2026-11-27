// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset3_decl.hpp"
#include "openvino/opsets/opset7_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::test {

using namespace testing;

namespace {
struct Conv1x1ToMatmulTestParams {
    bool with_group_quant;
    bool with_zp;
    bool with_bias;
    bool with_convert;
    bool with_param_weight;
    bool with_act_new_reshape;
    bool with_batched_input;
    std::string activation_op_type;
    ov::PartialShape input_shape = ov::PartialShape::dynamic();
    bool special_zero = false;
    ov::element::Type reshape_pattern_type = ov::element::i32;
};

using Conv1x1ToMatmulParams =
    std::tuple<bool, bool, bool, bool, bool, bool, bool, std::string, ov::PartialShape, bool, ov::element::Type>;

std::shared_ptr<ov::Model> gen_model(const Conv1x1ToMatmulTestParams& p) {
    size_t input_batch = p.with_batched_input ? 4 : 1;
    size_t input_h =
        ((p.activation_op_type == "Reshape" || p.activation_op_type == "Reshape_None") && p.with_act_new_reshape) ? 2
                                                                                                                  : 1;
    size_t input_w =
        ((p.activation_op_type == "Reshape" || p.activation_op_type == "Reshape_None") && p.with_act_new_reshape) ? 5
                                                                                                                  : 10;
    const auto input_shape =
        p.input_shape.rank().is_static()
            ? p.input_shape
            : ov::PartialShape((p.activation_op_type == "None_Transpose" || p.activation_op_type == "None_Reshape" ||
                                p.activation_op_type == "None")
                                   ? ov::Shape{input_batch, input_w, 1, input_h}
                                   : ov::Shape{input_batch, 1, input_h, input_w});
    const int64_t reshape_batch =
        p.input_shape.rank().is_static() ? (p.special_zero ? 0 : -1) : static_cast<int64_t>(input_batch);
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, input_shape);

    std::shared_ptr<ov::Node> act_node = input;
    if ((p.activation_op_type == "Transpose" || p.activation_op_type == "Transpose_None")) {
        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        act_node = std::make_shared<ov::opset1::Transpose>(input, transpose_const);
    } else if ((p.activation_op_type == "Reshape" || p.activation_op_type == "Reshape_None")) {
        auto reshape_const = ov::opset1::Constant::create(p.reshape_pattern_type,
                                                          ov::Shape{4},
                                                          std::vector<int64_t>{reshape_batch, 10, 1, 1});
        act_node = std::make_shared<ov::opset1::Reshape>(input, reshape_const, p.special_zero);
    }

    std::shared_ptr<ov::Node> weights_node;
    ov::ParameterVector params = {input};
    if (p.with_param_weight) {
        auto weights_param = std::make_shared<ov::opset1::Parameter>(
            ov::element::i4,
            p.with_group_quant ? ov::Shape{15, 5, 2, 1, 1} : ov::Shape{15, 10, 1, 1});
        weights_node = weights_param;
        params.push_back(weights_param);
    } else {
        weights_node =
            ov::opset1::Constant::create(ov::element::i4,
                                         p.with_group_quant ? ov::Shape{15, 5, 2, 1, 1} : ov::Shape{15, 10, 1, 1},
                                         {1});
    }

    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    std::shared_ptr<ov::Node> current_node = weights_convert;

    if (p.with_zp) {
        auto zp_const =
            ov::opset1::Constant::create(ov::element::i4,
                                         p.with_group_quant ? ov::Shape{15, 5, 1, 1, 1} : ov::Shape{15, 1, 1, 1},
                                         {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        current_node = std::make_shared<ov::opset1::Subtract>(weights_convert, zp_convert);
    }

    auto scale_const =
        ov::opset1::Constant::create(ov::element::f16,
                                     p.with_group_quant ? ov::Shape{15, 5, 1, 1, 1} : ov::Shape{15, 1, 1, 1},
                                     {1});
    std::shared_ptr<ov::Node> mul = std::make_shared<ov::opset1::Multiply>(current_node, scale_const);

    if (p.with_group_quant) {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {15, 10, 1, 1});
        mul = std::make_shared<ov::opset1::Reshape>(mul, reshape_const, false);
    }

    auto conv = std::make_shared<ov::opset1::Convolution>(act_node,
                                                          mul,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1},
                                                          ov::op::PadType::EXPLICIT);
    current_node = conv;

    if (p.with_bias) {
        auto bias_const = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 15, 1, 1}, {1});
        current_node = std::make_shared<ov::opset1::Add>(conv, bias_const);
    }
    if (p.with_convert) {
        current_node = std::make_shared<ov::op::v0::Convert>(current_node, ov::element::f32);
    }

    std::shared_ptr<ov::Node> out_node = current_node;
    if (p.activation_op_type == "Transpose" || p.activation_op_type == "None_Transpose") {
        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        out_node = std::make_shared<ov::opset1::Transpose>(current_node, transpose_const);
    } else if (p.activation_op_type == "Reshape" || p.activation_op_type == "None_Reshape") {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {(int)input_batch, 1, 1, 15});
        out_node = std::make_shared<ov::opset1::Reshape>(current_node, reshape_const, false);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{out_node}, params);
}

std::shared_ptr<ov::Model> gen_model_ref(const Conv1x1ToMatmulTestParams& p) {
    size_t input_batch = p.with_batched_input ? 4 : 1;
    size_t input_h =
        ((p.activation_op_type == "Reshape" || p.activation_op_type == "Reshape_None") && p.with_act_new_reshape) ? 2
                                                                                                                  : 1;
    size_t input_w =
        ((p.activation_op_type == "Reshape" || p.activation_op_type == "Reshape_None") && p.with_act_new_reshape) ? 5
                                                                                                                  : 10;
    const auto input_shape =
        p.input_shape.rank().is_static()
            ? p.input_shape
            : ov::PartialShape((p.activation_op_type == "None_Transpose" || p.activation_op_type == "None_Reshape")
                                   ? ov::Shape{input_batch, input_w, 1, input_h}
                                   : ov::Shape{input_batch, 1, input_h, input_w});
    const int64_t reshape_batch =
        p.input_shape.rank().is_static() ? (p.special_zero ? 0 : -1) : static_cast<int64_t>(input_batch);
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, input_shape);

    std::shared_ptr<ov::Node> weights_node;
    ov::ParameterVector params = {input};
    if (p.with_param_weight) {
        auto weights_param = std::make_shared<ov::opset1::Parameter>(
            ov::element::i4,
            p.with_group_quant ? ov::Shape{15, 5, 2, 1, 1} : ov::Shape{15, 10, 1, 1});
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32,
                                                          p.with_group_quant ? ov::Shape{3} : ov::Shape{2},
                                                          p.with_group_quant ? ov::Shape{15, 5, 2} : ov::Shape{15, 10});
        weights_node = std::make_shared<ov::opset1::Reshape>(weights_param, reshape_const, false);
        params.push_back(weights_param);
    } else {
        weights_node = ov::opset1::Constant::create(ov::element::i4,
                                                    p.with_group_quant ? ov::Shape{15, 5, 2} : ov::Shape{15, 10},
                                                    {1});
    }

    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    std::shared_ptr<ov::Node> current_node = weights_convert;

    if (p.with_zp) {
        auto zp_const = ov::opset1::Constant::create(ov::element::i4,
                                                     p.with_group_quant ? ov::Shape{15, 5, 1} : ov::Shape{15, 1},
                                                     {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f16);
        current_node = std::make_shared<ov::opset1::Subtract>(weights_convert, zp_convert);
    }

    auto scale_const = ov::opset1::Constant::create(ov::element::f16,
                                                    p.with_group_quant ? ov::Shape{15, 5, 1} : ov::Shape{15, 1},
                                                    {1});
    std::shared_ptr<ov::Node> mul = std::make_shared<ov::opset1::Multiply>(current_node, scale_const);

    if (p.with_group_quant) {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{2}, {15, 10});
        mul = std::make_shared<ov::opset1::Reshape>(mul, reshape_const, false);
    }

    std::shared_ptr<ov::Node> act_node = input;
    if (p.activation_op_type == "None_Transpose" || p.activation_op_type == "None_Reshape") {
        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        act_node = std::make_shared<ov::opset1::Transpose>(act_node, transpose_const);
    }
    if ((p.activation_op_type == "Reshape" || p.activation_op_type == "Reshape_None") && p.with_act_new_reshape) {
        auto reshape_const = ov::opset1::Constant::create(p.reshape_pattern_type,
                                                          ov::Shape{4},
                                                          std::vector<int64_t>{reshape_batch, 1, 1, 10});
        act_node = std::make_shared<ov::opset1::Reshape>(act_node, reshape_const, p.special_zero);
    }
    auto matmul = std::make_shared<ov::op::v0::MatMul>(act_node, mul, false, true);
    current_node = matmul;

    if (p.with_bias) {
        auto bias_const = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 1, 1, 15}, {1});
        current_node = std::make_shared<ov::opset1::Add>(current_node, bias_const);
    }
    if (p.with_convert) {
        current_node = std::make_shared<ov::op::v0::Convert>(current_node, ov::element::f32);
    }

    std::shared_ptr<ov::Node> out_node = current_node;
    if (p.activation_op_type == "Reshape" || p.activation_op_type == "None_Reshape") {
        // Here H = W = 1, so the NHWC MatMul output [N, 1, 1, Cout] and the NCHW order the matched
        // Reshape was built for [N, Cout, 1, 1] have identical element order. The transformation
        // therefore feeds the MatMul output directly into the matched Reshape without inserting any
        // NCHW-restoring Reshape/Transpose (those are only needed when H*W > 1).
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {(int)input_batch, 1, 1, 15});
        out_node = std::make_shared<ov::opset1::Reshape>(out_node, reshape_const, false);
    }
    if (p.activation_op_type == "Transpose_None" || p.activation_op_type == "Reshape_None") {
        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        out_node = std::make_shared<ov::opset1::Transpose>(out_node, transpose_const);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{out_node}, params);
}

std::vector<Conv1x1ToMatmulParams> get_reshape_input_test_params() {
    const auto make_params = [](const ov::PartialShape& input_shape, bool special_zero, bool with_act_new_reshape) {
        return Conv1x1ToMatmulParams{false,
                                     false,
                                     false,
                                     false,
                                     false,
                                     with_act_new_reshape,
                                     false,
                                     "Reshape_None",
                                     input_shape,
                                     special_zero,
                                     ov::element::i64};
    };
    return {make_params(ov::PartialShape{2, 10}, false, true),
            make_params(ov::PartialShape{-1, 10}, false, true),
            make_params(ov::PartialShape{2, 1, 10}, false, true),
            make_params(ov::PartialShape{-1, 1, 10}, false, true),
            make_params(ov::PartialShape{2, 10, 1, 1}, false, true),
            make_params(ov::PartialShape{-1, 10, 1, 1}, false, true),
            make_params(ov::PartialShape{2, 1, 1, 1, 10}, false, true),
            make_params(ov::PartialShape{20}, false, true),
            make_params(ov::PartialShape{-1}, false, true),
            make_params(ov::PartialShape{2, 1, 1, 10}, false, false),
            make_params(ov::PartialShape{2, 10}, true, true),
            make_params(ov::PartialShape{-1, 10}, true, true),
            make_params(ov::PartialShape{2, 1, 10}, true, true),
            make_params(ov::PartialShape{-1, 1, 10}, true, true),
            make_params(ov::PartialShape{2, 1, 1, 10}, true, false)};
}
}  // namespace

class ConvertWeightCompressedConv1x1ToMatmulTest : public TransformationTestsF,
                                                   public WithParamInterface<Conv1x1ToMatmulParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<Conv1x1ToMatmulParams>& obj) {
        const auto& [with_group_quant,
                     with_zp,
                     with_bias,
                     with_convert,
                     with_param_weight,
                     with_act_new_reshape,
                     with_batched_input,
                     activation_op_type,
                     input_shape,
                     special_zero,
                     reshape_pattern_type] = obj.param;

        std::ostringstream result;
        result << "with_group_quant=" << with_group_quant << "_";
        result << "with_zp=" << with_zp << "_";
        result << "with_bias=" << with_bias << "_";
        result << "with_convert=" << with_convert << "_";
        result << "with_param_weight=" << with_param_weight << "_";
        result << "with_act_new_reshape=" << with_act_new_reshape << "_";
        result << "with_batched_input=" << with_batched_input << "_";
        result << "activation_op_type=" << activation_op_type;
        if (input_shape.rank().is_static()) {
            result << "_input_shape";
            for (const auto& dimension : input_shape) {
                result << "_" << (dimension.is_dynamic() ? "Dynamic" : std::to_string(dimension.get_length()));
            }
            result << "_special_zero=" << special_zero << "_reshape_pattern_type=" << reshape_pattern_type;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [with_group_quant,
                     with_zp,
                     with_bias,
                     with_convert,
                     with_param_weight,
                     with_act_new_reshape,
                     with_batched_input,
                     activation_op_type,
                     input_shape,
                     special_zero,
                     reshape_pattern_type] = GetParam();
        Conv1x1ToMatmulTestParams params{with_group_quant,
                                         with_zp,
                                         with_bias,
                                         with_convert,
                                         with_param_weight,
                                         with_act_new_reshape,
                                         with_batched_input,
                                         activation_op_type,
                                         input_shape,
                                         special_zero,
                                         reshape_pattern_type};
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        model = gen_model(params);
        model_ref = (activation_op_type == "None") ? model->clone() : gen_model_ref(params);
        manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul>();
    }
};

TEST_P(ConvertWeightCompressedConv1x1ToMatmulTest, CompareFunctions) {}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         ConvertWeightCompressedConv1x1ToMatmulTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values("Transpose",
                                                              "Reshape",
                                                              "Transpose_None",
                                                              "Reshape_None",
                                                              "None_Transpose",
                                                              "None_Reshape",
                                                              "None"),
                                            ::testing::Values(ov::PartialShape::dynamic()),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::element::i32)),
                         ConvertWeightCompressedConv1x1ToMatmulTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(TransformationTests_ReshapeInput,
                         ConvertWeightCompressedConv1x1ToMatmulTest,
                         ::testing::ValuesIn(get_reshape_input_test_params()),
                         ConvertWeightCompressedConv1x1ToMatmulTest::get_test_case_name);

// Checked blocked cases
class ConvertWeightCompressedConv1x1ToMatmulTest_Blocked : public TransformationTestsF {
public:
    ConvertWeightCompressedConv1x1ToMatmulTest_Blocked() {
        manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul>();
    }
};

TEST_F(ConvertWeightCompressedConv1x1ToMatmulTest_Blocked, conv3x3) {
    auto CreateConv3x3 = [&]() {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{1, 1};
        ov::CoordinateDiff pads_end{1, 1};
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1, 1, 2, 1});
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i4, ov::Shape{1, 1, 3, 3}, {1});
        auto input2_convert = std::make_shared<ov::op::v0::Convert>(input2, ov::element::f16);
        auto input2_scale = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 1, 3, 3}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(input2_convert, input2_scale);
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);
        auto conv3x3 = std::make_shared<ov::opset1::Convolution>(transpose1,
                                                                 mul,
                                                                 strides,
                                                                 pads_begin,
                                                                 pads_end,
                                                                 dilations,
                                                                 ov::op::PadType::EXPLICIT);
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(conv3x3, transpose_constant2);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{transpose2}, ov::ParameterVector{input1});
        return model;
    };
    model = CreateConv3x3();
}

TEST_F(ConvertWeightCompressedConv1x1ToMatmulTest_Blocked, converted_scale) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1, 2, 2, 3});
    auto order = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose = std::make_shared<ov::opset1::Transpose>(input, order);
    auto weights = ov::opset1::Constant::create(ov::element::i8, ov::Shape{4, 3, 1, 1}, {1});
    auto weights_convert = std::make_shared<ov::opset1::Convert>(weights, ov::element::f16);
    auto scale = ov::opset1::Constant::create(ov::element::f32, ov::Shape{4, 1, 1, 1}, {0.25f});
    auto scale_convert = std::make_shared<ov::opset1::Convert>(scale, ov::element::f16);
    auto multiply = std::make_shared<ov::opset1::Multiply>(weights_convert, scale_convert);
    auto conv = std::make_shared<ov::opset1::Convolution>(transpose,
                                                          multiply,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1});
    model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
}

TEST_F(ConvertWeightCompressedConv1x1ToMatmulTest_Blocked, parameter_zero_point) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1, 2, 2, 3});
    auto order = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose = std::make_shared<ov::opset1::Transpose>(input, order);
    auto weights = ov::opset1::Constant::create(ov::element::i8, ov::Shape{4, 3, 1, 1}, {1});
    auto weights_convert = std::make_shared<ov::opset1::Convert>(weights, ov::element::f16);
    auto zero_point = std::make_shared<ov::opset1::Parameter>(ov::element::i8, ov::Shape{4, 1, 1, 1});
    auto zero_point_convert = std::make_shared<ov::opset1::Convert>(zero_point, ov::element::f16);
    auto subtract = std::make_shared<ov::opset1::Subtract>(weights_convert, zero_point_convert);
    auto scale = ov::opset1::Constant::create(ov::element::f16, ov::Shape{4, 1, 1, 1}, {0.25f});
    auto multiply = std::make_shared<ov::opset1::Multiply>(subtract, scale);
    auto conv = std::make_shared<ov::opset1::Convolution>(transpose,
                                                          multiply,
                                                          ov::Strides{1, 1},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::CoordinateDiff{0, 0},
                                                          ov::Strides{1, 1});
    model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input, zero_point});
}

// Regression test for the NCHW/NHWC layout mismatch when the Convolution output is consumed by a
// Reshape and the spatial size H*W > 1 (see PR #37133). This mirrors the self-attention projections
// of the machine-translation Transformer model that regressed: a weight-compressed 1x1 Convolution
// producing NCHW [1, Cout, 1, W] (W > 1) feeding a Reshape to [Cout, W].
//
// MatMul(transpose_b=true) produces channel-last output [N, H, W, Cout], while the Convolution it
// replaces produces channel-second NCHW [N, Cout, H, W]. Reshape never reorders elements, so the
// matched Reshape (built for the Convolution's NCHW output) must receive NCHW data.
class ConvertWeightCompressedConv1x1ToMatmulTest_ReshapeConsumerSpatial : public TransformationTestsF {
public:
    ConvertWeightCompressedConv1x1ToMatmulTest_ReshapeConsumerSpatial() {
        manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul>();
    }
};

TEST_F(ConvertWeightCompressedConv1x1ToMatmulTest_ReshapeConsumerSpatial, conv1x1_reshape_consumer_spatial_gt_1) {
    const int Cin = 10;
    const int Cout = 15;
    const int W = 5;  // spatial size > 1 - exposes the NHWC/NCHW mismatch

    {
        // input [1, 1, W, Cin] -> Transpose[0,3,1,2] -> conv input [1, Cin, 1, W]
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, W, Cin});
        auto in_transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto act_node = std::make_shared<ov::op::v1::Transpose>(input, in_transpose_const);

        auto weights = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{Cout, Cin, 1, 1}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{Cout, 1, 1, 1}, {1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);

        auto conv = std::make_shared<ov::op::v1::Convolution>(act_node,
                                                              mul,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        // conv output NCHW [1, Cout, 1, W] -> Reshape to [Cout, W]
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {Cout, W});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(conv, reshape_const, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, W, Cin});

        // weights squeezed to 2D [Cout, Cin]
        auto weights = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{Cout, Cin}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{Cout, 1}, {1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);

        // activation is used directly (input transpose is absorbed); MatMul(transpose_b=true)
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, mul, false, true);  // -> [1, 1, W, Cout] NHWC

        // MatMul output is already NHWC [N, H, W, Cout], so a single Transpose [0, 3, 1, 2] restores NCHW
        auto nhwc_to_nchw_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto nchw = std::make_shared<ov::op::v1::Transpose>(matmul, nhwc_to_nchw_const);  // -> [1, Cout, 1, W]

        // matched Reshape to [Cout, W]
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {Cout, W});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(nchw, reshape_const, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
}

// Companion to conv1x1_reshape_consumer_spatial_gt_1 for the spatial size H*W == 1 case, which
// mirrors the gpt_oss Transformer self-attention projections: a weight-compressed 1x1 Convolution
// producing NCHW [1, Cout, 1, 1] whose output is consumed by a Reshape.
//
// When H*W == 1 the channel-last MatMul output [N, 1, 1, Cout] and the NCHW order the matched
// Reshape was built for [N, Cout, 1, 1] have identical element order, so no NCHW-restoring
// Reshape/Transpose is needed - the MatMul output must feed the matched Reshape directly. This
// guards against inserting the redundant no-op Reshape/Transpose pair (which is only required when
// H*W > 1, see conv1x1_reshape_consumer_spatial_gt_1).
TEST_F(ConvertWeightCompressedConv1x1ToMatmulTest_ReshapeConsumerSpatial, conv1x1_reshape_consumer_spatial_eq_1) {
    const int Cin = 10;
    const int Cout = 15;

    {
        // input [1, 1, 1, Cin] -> Transpose[0,3,1,2] -> conv input [1, Cin, 1, 1]
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 1, Cin});
        auto in_transpose_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto act_node = std::make_shared<ov::op::v1::Transpose>(input, in_transpose_const);

        auto weights = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{Cout, Cin, 1, 1}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{Cout, 1, 1, 1}, {1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);

        auto conv = std::make_shared<ov::op::v1::Convolution>(act_node,
                                                              mul,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        // conv output NCHW [1, Cout, 1, 1] -> Reshape to [1, Cout]
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, Cout});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(conv, reshape_const, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 1, Cin});

        // weights squeezed to 2D [Cout, Cin]
        auto weights = ov::op::v0::Constant::create(ov::element::i4, ov::Shape{Cout, Cin}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{Cout, 1}, {1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);

        // activation is used directly (input transpose is absorbed); MatMul(transpose_b=true)
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, mul, false, true);  // -> [1, 1, 1, Cout] NHWC

        // H*W == 1: the MatMul output feeds the matched Reshape directly, no Reshape/Transpose pair.
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, Cout});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_const, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
}
}  // namespace ov::test
