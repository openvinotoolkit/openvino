// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.hpp"

using namespace ov;
using namespace testing;

namespace {
struct Conv1x1ToMatmulActNotTranParams {
    PartialShape input_shape;
    std::vector<int64_t> input_transpose_order;
    std::vector<int64_t> output_transpose_order;
    bool with_zp;
};

std::shared_ptr<ov::Model> create_model(const Conv1x1ToMatmulActNotTranParams& p) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, p.input_shape);

    const size_t hidden_out = 4;
    const size_t hidden_in = p.input_shape[1].get_length();
    const size_t block_size = 2;

    auto weights =
        ov::op::v0::Constant::create(element::i4, {hidden_out, hidden_in / block_size, block_size, 1, 1}, {1});
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, element::f32);

    Output<Node> sub_input = weights_convert;
    if (p.with_zp) {
        auto zp = ov::op::v0::Constant::create(element::i4, {hidden_out, hidden_in / block_size, 1, 1, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp, element::f32);
        sub_input = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
    }

    auto scale = ov::op::v0::Constant::create(element::f32, {hidden_out, hidden_in / block_size, 1, 1, 1}, {1});
    auto weights_mult = std::make_shared<ov::op::v1::Multiply>(sub_input, scale);

    auto reshape_const = ov::op::v0::Constant::create(
        element::i64,
        {4},
        std::vector<int64_t>{static_cast<int64_t>(hidden_out), static_cast<int64_t>(hidden_in), 1, 1});
    auto weights_reshape = std::make_shared<ov::op::v1::Reshape>(weights_mult, reshape_const, false);

    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weights_reshape,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});

    return std::make_shared<Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_ref_model(const Conv1x1ToMatmulActNotTranParams& p) {
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, p.input_shape);

    const size_t hidden_out = 4;
    const size_t hidden_in = p.input_shape[1].get_length();
    const size_t block_size = 2;

    auto input_transpose_const = ov::op::v0::Constant::create(element::i64, {4}, p.input_transpose_order);
    auto transpose_input = std::make_shared<ov::op::v1::Transpose>(input, input_transpose_const);

    auto weights = ov::op::v0::Constant::create(element::i4, {hidden_out, hidden_in / block_size, block_size}, {1});
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, element::f32);

    Output<Node> sub_input = weights_convert;
    if (p.with_zp) {
        auto zp = ov::op::v0::Constant::create(element::i4, {hidden_out, hidden_in / block_size, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp, element::f32);
        sub_input = std::make_shared<ov::op::v1::Subtract>(weights_convert, zp_convert);
    }

    auto scale = ov::op::v0::Constant::create(element::f32, {hidden_out, hidden_in / block_size, 1}, {1});
    auto weights_mult = std::make_shared<ov::op::v1::Multiply>(sub_input, scale);

    auto reshape_const = ov::op::v0::Constant::create(
        element::i64,
        {2},
        std::vector<int64_t>{static_cast<int64_t>(hidden_out), static_cast<int64_t>(hidden_in)});
    auto weights_reshape = std::make_shared<ov::op::v1::Reshape>(weights_mult, reshape_const, false);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_input, weights_reshape, false, true);

    auto output_transpose_const = ov::op::v0::Constant::create(element::i64, {4}, p.output_transpose_order);
    auto final_node = std::make_shared<ov::op::v1::Transpose>(matmul, output_transpose_const);

    return std::make_shared<Model>(ov::OutputVector{final_node}, ov::ParameterVector{input});
}
}  // namespace

class ConvertWeightCompressedConv1x1ToMatmulActNotTranTest
    : public TransformationTestsF,
      public WithParamInterface<Conv1x1ToMatmulActNotTranParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<Conv1x1ToMatmulActNotTranParams>& obj) {
        Conv1x1ToMatmulActNotTranParams p = obj.param;
        std::ostringstream result;
        result << "input_shape=" << p.input_shape << "_";
        result << "with_zp=" << p.with_zp;
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        auto p = GetParam();
        model = create_model(p);
        model_ref = create_ref_model(p);
        manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul_ActNotTran>(
            element::TypeVector{element::u4, element::i4},
            element::TypeVector{});
    }
};

TEST_P(ConvertWeightCompressedConv1x1ToMatmulActNotTranTest, CompareFunctions) {}

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    ConvertWeightCompressedConv1x1ToMatmulActNotTranTest,
    ::testing::Values(
        Conv1x1ToMatmulActNotTranParams{{1, 8, 1, 10}, {0, 2, 3, 1}, {0, 3, 1, 2}, true},
        Conv1x1ToMatmulActNotTranParams{{1, 8, 1, 10}, {0, 2, 3, 1}, {0, 3, 1, 2}, false},
        Conv1x1ToMatmulActNotTranParams{{10, 8, 1, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, true},
        Conv1x1ToMatmulActNotTranParams{{10, 8, 1, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, false},
        Conv1x1ToMatmulActNotTranParams{{1, 8, 10, 1}, {0, 3, 2, 1}, {0, 3, 2, 1}, true},
        Conv1x1ToMatmulActNotTranParams{{1, 8, 10, 1}, {0, 3, 2, 1}, {0, 3, 2, 1}, false},
        Conv1x1ToMatmulActNotTranParams{{1, 8, 1, Dimension::dynamic()}, {0, 2, 3, 1}, {0, 3, 1, 2}, true},
        Conv1x1ToMatmulActNotTranParams{{1, 8, 1, Dimension::dynamic()}, {0, 2, 3, 1}, {0, 3, 1, 2}, false},
        Conv1x1ToMatmulActNotTranParams{{Dimension::dynamic(), 8, 1, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, true},
        Conv1x1ToMatmulActNotTranParams{{Dimension::dynamic(), 8, 1, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, false},
        Conv1x1ToMatmulActNotTranParams{{1, 8, Dimension::dynamic(), 1}, {0, 3, 2, 1}, {0, 3, 2, 1}, true},
        Conv1x1ToMatmulActNotTranParams{{1, 8, Dimension::dynamic(), 1}, {0, 3, 2, 1}, {0, 3, 2, 1}, false}),
    ConvertWeightCompressedConv1x1ToMatmulActNotTranTest::get_test_case_name);

// Checked blocked cases
TEST(TransformationTests, ConvertWeightCompressedConv1x1ToMatmulActNotTranExceptionTest_conv3x3) {
    auto create_conv_3x3 = []() {
        auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4, 2, 1});
        auto weights = ov::op::v0::Constant::create(element::i4, {8, 2, 2, 3, 3}, {1});
        auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, element::f32);
        auto scale = ov::op::v0::Constant::create(element::f32, {8, 2, 1, 3, 3}, {1});
        auto weights_mult = std::make_shared<ov::op::v1::Multiply>(weights_convert, scale);
        auto reshape_const = ov::op::v0::Constant::create(element::i64, {4}, {8, 4, 3, 3});
        auto weights_reshape = std::make_shared<ov::op::v1::Reshape>(weights_mult, reshape_const, false);

        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              weights_reshape,
                                                              Strides{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              Strides{1, 1});
        return std::make_shared<Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    };

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertWeightCompressedConv1x1ToMatmul_ActNotTran>(element::TypeVector{element::i4},
                                                                                       element::TypeVector{});

    auto model = create_conv_3x3();
    manager.run_passes(model);

    bool converted = false;
    for (const auto& op : model->get_ops()) {
        if (std::string(op->get_type_name()) == "MatMul") {
            converted = true;
            break;
        }
    }
    ASSERT_FALSE(converted);
}
