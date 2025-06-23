// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moc_transformations.hpp"

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset12_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

using namespace testing;
using namespace ov;
using namespace ov::element;
using namespace ov::opset12;

namespace {

std::shared_ptr<ov::Node> make_dq_weights(const ov::element::Type& quant_type,
                                          const ov::Shape& w_shape,
                                          float scale,
                                          float zp,
                                          bool dq_markup = false) {
    auto w = ov::op::v0::Constant::create(quant_type, w_shape, {1.0f});
    auto w_f = std::make_shared<ov::op::v0::Convert>(w, ov::element::f32);

    auto zp_const = ov::op::v0::Constant::create(quant_type, {}, {zp});
    auto zp_f = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
    auto sub = std::make_shared<ov::op::v1::Subtract>(w_f, zp_f);

    std::vector<size_t> scale_shape(w_shape.size(), 1);
    auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_shape, {scale});
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, scale_const);

    if (dq_markup) {
        mark_as_dequantization_node(sub);
        mark_as_dequantization_node(mul);
        disable_constant_folding(zp_f);
    }
    return mul;
}

}  // namespace

TEST(TransformationTests, TestModelTensorsConsistencyUseShapesTrue) {
    auto input = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    auto const1 = opset12::Constant::create(element::f32, Shape{1}, {1});
    auto const2 = opset12::Constant::create(element::f32, Shape{1}, {2});
    auto const3 = opset12::Constant::create(element::f32, Shape{1}, {3});
    auto add1 = std::make_shared<opset12::Add>(input, const1);
    auto add2 = std::make_shared<opset12::Add>(add1, const2);
    auto add3 = std::make_shared<opset12::Add>(add2, const3);

    auto model = std::make_shared<Model>(OutputVector{add3}, ParameterVector{input});
    ov::pass::Manager m;
    m.register_pass<ov::pass::MOCTransformations>(true);
    m.run_passes(model);

    std::unordered_set<std::string> new_tensors = {"new_name"};
    model->outputs()[0].set_names(new_tensors);
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);

    model->validate_nodes_and_infer_types();
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);
}

TEST(TransformationTests, MOCConvertElimination) {
    auto input = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    auto const_val = opset12::Constant::create(element::f32, Shape{1}, {2});

    auto add1 = std::make_shared<opset12::Add>(input, const_val);
    auto convert_fp32 = std::make_shared<opset12::Convert>(const_val, element::f32);
    auto mul = std::make_shared<opset12::MatMul>(add1, convert_fp32);

    auto model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{input});
    ov::pass::Manager m;
    m.register_pass<ov::pass::MOCTransformations>(false);
    m.run_passes(model);

    EXPECT_EQ(count_ops_of_type<opset12::Constant>(model), 1);
}

TEST(TransformationTests, TestModelTensorsConsistencyUseShapesFalse) {
    auto input = std::make_shared<opset12::Parameter>(element::f32, Shape{1});
    auto const1 = opset12::Constant::create(element::f32, Shape{1}, {1});
    auto const2 = opset12::Constant::create(element::f32, Shape{1}, {2});
    auto const3 = opset12::Constant::create(element::f32, Shape{1}, {3});
    auto add1 = std::make_shared<opset12::Add>(input, const1);
    auto add2 = std::make_shared<opset12::Add>(add1, const2);
    auto add3 = std::make_shared<opset12::Add>(add2, const3);

    auto model = std::make_shared<Model>(OutputVector{add3}, ParameterVector{input});
    ov::pass::Manager m;
    m.register_pass<ov::pass::MOCTransformations>(false);
    m.run_passes(model);

    std::unordered_set<std::string> new_tensors = {"new_name"};
    model->outputs()[0].set_names(new_tensors);
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);

    model->validate_nodes_and_infer_types();
    EXPECT_TRUE(model->outputs()[0].get_names() == new_tensors);
}

TEST_F(TransformationTestsF, SqueezeRemainsSqueezeAfterMOC) {
    {
        using namespace ov::op;
        auto input = std::make_shared<v0::Parameter>(element::f32, Shape{30});
        auto shape = v0::Constant::create(element::i64, Shape{5}, {2, 3, 1, 5, 1});
        auto reshape = std::make_shared<v1::Reshape>(input, shape, false);
        auto unsqueeze_axes = v0::Constant::create(element::i64, Shape{1}, {0});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(reshape, unsqueeze_axes);

        auto squeeze_axes = v0::Constant::create(element::i64, Shape{2}, {3, 5});
        auto squeeze = std::make_shared<v0::Squeeze>(unsqueeze, squeeze_axes);

        auto res = std::make_shared<v0::Result>(squeeze);
        model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MOCTransformations>(false);
    }
}

TEST_F(TransformationTestsF, MOCTest) {
    std::shared_ptr<ov::Node> weights;
    std::shared_ptr<ov::Node> weights_ref;
    {
        using namespace ov::op;
        auto data = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 5});
        auto data1 = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 5});
        auto a_mul = std::make_shared<v1::Multiply>(data, data1);
        weights = std::make_shared<v0::Constant>(element::f32, ov::Shape{3, 5});
        auto scale = std::make_shared<v0::Constant>(element::f32, ov::Shape{}, 0.194145);
        auto matmul = std::make_shared<v0::MatMul>(a_mul, weights, false, true);
        auto mul = std::make_shared<v1::Multiply>(matmul, scale);
        auto res = std::make_shared<v0::Result>(mul);

        weights->set_friendly_name("self.model.layers.50.mlp.down_proj.weight");

        model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data, data1});
        manager.register_pass<ov::pass::MOCTransformations>(false);
    }
    {
        using namespace ov::op;
        auto data = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 5});
        auto data1 = std::make_shared<v0::Parameter>(element::f32, ov::PartialShape{-1, -1, 5});
        auto a_mul = std::make_shared<v1::Multiply>(data, data1);
        weights_ref = std::make_shared<v0::Constant>(element::f32, ov::Shape{3, 5});
        auto scale = std::make_shared<v0::Constant>(element::f32, ov::Shape{1, 1, 1}, 0.194145);
        auto matmul = std::make_shared<v0::MatMul>(a_mul, weights_ref, false, true);
        auto mul = std::make_shared<v1::Multiply>(matmul, scale);
        auto res = std::make_shared<v0::Result>(mul);

        weights_ref->set_friendly_name("self.model.layers.50.mlp.down_proj.weight");

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data, data1});

        EXPECT_EQ(weights->get_friendly_name(), weights_ref->get_friendly_name());
    }
}

class QuantWeightsTestP : public TransformationTestsF, public ::testing::WithParamInterface<ov::element::Type> {
protected:
    ov::element::Type qtype;

    void SetUp() override {
        TransformationTestsF::SetUp();
        qtype = GetParam();
    }
};

/**
 *  Model structure for MatMul_Conv_QuantWeights test
 *
 *   Input
 *     │
 *   MatMul  ◀── DQ (MatMul weights)
 *     │
 *  Reshape
 *     │
 *   Conv    ◀── DQ (Conv weights)
 *     │
 *   Output
 *
 *  Test checks that DQ subgraphs for MatMul and Conv weights
 *  are not modified by MOC and PrePostProcessing passes.
 *
 */
TEST_P(QuantWeightsTestP, MatMul_Conv_QuantWeights) {
    {
        const float matmul_scale = 0.02f, matmul_zp = 1.f;
        const float conv_scale = 0.05f, conv_zp = 2.f;

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128});
        auto matmul_weight = make_dq_weights(qtype, ov::Shape{128, 64}, matmul_scale, matmul_zp);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, matmul_weight, false, false);

        auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 64, 1, 1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_const, false);
        auto conv_weight = make_dq_weights(qtype, ov::Shape{8, 64, 1, 1}, conv_scale, conv_zp);
        auto conv = std::make_shared<ov::op::v1::Convolution>(reshape,
                                                              conv_weight,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});

        auto prep = ov::preprocess::PrePostProcessor(model);

        model = prep.build();

        manager.register_pass<ov::pass::MOCTransformations>(false);
    }

    {
        const float matmul_scale = 0.02f, matmul_zp = 1.f;
        const float conv_scale = 0.05f, conv_zp = 2.f;

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128});
        auto matmul_weight = make_dq_weights(qtype, ov::Shape{128, 64}, matmul_scale, matmul_zp, true);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input, matmul_weight, false, false);

        auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 64, 1, 1});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_const, false);
        auto conv_weight = make_dq_weights(qtype, ov::Shape{8, 64, 1, 1}, conv_scale, conv_zp, true);
        auto conv = std::make_shared<ov::op::v1::Convolution>(reshape,
                                                              conv_weight,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(
    QuantWeightTypes,
    QuantWeightsTestP,
    ::testing::Values(i32, u32, i16, u16, i8, u8, u6, i4, u4, nf4, u3, u2, u1, f4e2m1, f8e4m3, f8e5m2, f8e8m0));
