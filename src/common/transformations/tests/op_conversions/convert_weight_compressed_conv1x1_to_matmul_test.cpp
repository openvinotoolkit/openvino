// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

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
    std::string activation_op_type;
};

std::shared_ptr<ov::Model> gen_model(const Conv1x1ToMatmulTestParams& p) {
    auto input = std::make_shared<ov::opset1::Parameter>(
        ov::element::f16,
        (p.activation_op_type == "Reshape" && p.with_act_new_reshape) ? ov::Shape{1, 1, 2, 5} : ov::Shape{1, 1, 1, 10});

    std::shared_ptr<ov::Node> act_node;
    if (p.activation_op_type == "Transpose") {
        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        act_node = std::make_shared<ov::opset1::Transpose>(input, transpose_const);
    } else {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {1, 10, 1, 1});
        act_node = std::make_shared<ov::opset1::Reshape>(input, reshape_const, false);
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

    std::shared_ptr<ov::Node> out_node;
    if (p.activation_op_type == "Transpose") {
        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        out_node = std::make_shared<ov::opset1::Transpose>(current_node, transpose_const);
    } else {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {1, 1, 1, 15});
        out_node = std::make_shared<ov::opset1::Reshape>(current_node, reshape_const, false);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{out_node}, params);
}

std::shared_ptr<ov::Model> gen_model_ref(const Conv1x1ToMatmulTestParams& p) {
    auto input = std::make_shared<ov::opset1::Parameter>(
        ov::element::f16,
        (p.activation_op_type == "Reshape" && p.with_act_new_reshape) ? ov::Shape{1, 1, 2, 5} : ov::Shape{1, 1, 1, 10});

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
        auto reshape_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {15, 10});
        mul = std::make_shared<ov::opset1::Reshape>(mul, reshape_const, false);
    }

    std::shared_ptr<ov::Node> act_node = input;
    if (p.activation_op_type == "Reshape" && p.with_act_new_reshape) {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 10});
        act_node = std::make_shared<ov::opset1::Reshape>(input, reshape_const, false);
    }
    auto matmul = std::make_shared<ov::op::v0::MatMul>(act_node, mul, false, true);
    current_node = matmul;

    if (p.with_bias) {
        auto bias_const = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 1, 1, 15}, {1});
        current_node = std::make_shared<ov::opset1::Add>(matmul, bias_const);
    }
    if (p.with_convert) {
        current_node = std::make_shared<ov::op::v0::Convert>(current_node, ov::element::f32);
    }

    std::shared_ptr<ov::Node> out_node;
    if (p.activation_op_type == "Reshape") {
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {1, 1, 1, 15});
        out_node = std::make_shared<ov::opset1::Reshape>(current_node, reshape_const, false);
    } else {
        out_node = current_node;
    }

    return std::make_shared<ov::Model>(ov::OutputVector{out_node}, params);
}
}  // namespace

class ConvertWeightCompressedConv1x1ToMatmulTest
    : public TransformationTestsF,
      public WithParamInterface<std::tuple<bool, bool, bool, bool, bool, bool, std::string>> {
public:
    static std::string get_test_case_name(
        const testing::TestParamInfo<std::tuple<bool, bool, bool, bool, bool, bool, std::string>>& obj) {
        const auto& [with_group_quant,
                     with_zp,
                     with_bias,
                     with_convert,
                     with_param_weight,
                     with_act_new_reshape,
                     activation_op_type] = obj.param;

        std::ostringstream result;
        result << "with_group_quant=" << with_group_quant << "_";
        result << "with_zp=" << with_zp << "_";
        result << "with_bias=" << with_bias << "_";
        result << "with_convert=" << with_convert << "_";
        result << "with_param_weight=" << with_param_weight << "_";
        result << "with_act_new_reshape=" << with_act_new_reshape << "_";
        result << "activation_op_type=" << activation_op_type;
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
                     activation_op_type] = GetParam();
        Conv1x1ToMatmulTestParams params{with_group_quant,
                                         with_zp,
                                         with_bias,
                                         with_convert,
                                         with_param_weight,
                                         with_act_new_reshape,
                                         activation_op_type};
        model = gen_model(params);
        model_ref = gen_model_ref(params);
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
                                            ::testing::Values("Transpose", "Reshape")),
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
}  // namespace ov::test
