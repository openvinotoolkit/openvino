// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/sdpa.hpp"

#include "plugin/transformations/transpose_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, TransposeSDPAFusion1) {
    const bool is_causal = true;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(input_a, input_b, input_c, is_causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 1, 2, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransformationTestsF) {
    const bool is_causal = true;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, input_b, input_c, is_causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeSDPAFusion3) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 2, 0, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, input_c, is_causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {1, 2, 0, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeSDPAFusion4) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, tranpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, tranpose_c, is_causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 2, 1, 3};
        std::vector<int64_t> order_c = {0, 2, 1, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, TransposeSDPAFusion5) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {3, 2, 1, 0});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, tranpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, tranpose_c, is_causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, tranpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto tranpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, tranpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto tranpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {3, 2, 1, 0});
        auto tranpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, tranpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(tranpose_a, tranpose_b, tranpose_c, is_causal);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// Existing Q order {0,2,1,3} composes with input Transpose {0,2,1,3} to identity.
TEST_F(TransformationTestsF, TransposeSDPAFusion_QComposed) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto transpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{transpose_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        std::vector<int64_t> order_a = {0, 1, 2, 3};
        std::vector<int64_t> order_b = {0, 1, 2, 3};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// Existing K order {0,2,3,1} composes with input Transpose {0,2,1,3} to {0,1,3,2}; output order kept.
TEST_F(TransformationTestsF, TransposeSDPAFusion_KComposed) {
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto transpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, transpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        std::vector<int64_t> order_a = {0, 1, 2, 3};
        std::vector<int64_t> order_b = {0, 2, 3, 1};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 2, 1, 3};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, transpose_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        std::vector<int64_t> order_a = {0, 1, 2, 3};
        std::vector<int64_t> order_b = {0, 1, 3, 2};
        std::vector<int64_t> order_c = {0, 1, 2, 3};
        std::vector<int64_t> order_output = {0, 2, 1, 3};
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

class TransposeSDPAFusionPrecisionTest : public TransformationTestsF,
                                         public ::testing::WithParamInterface<ov::element::Type> {};

TEST_P(TransposeSDPAFusionPrecisionTest, AllInputsTransposed) {
    const auto input_type = GetParam();
    const bool is_causal = false;
    {
        auto input_a = std::make_shared<ov::op::v0::Parameter>(input_type, ov::PartialShape::dynamic(4));
        auto transpose_a_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose_a = std::make_shared<ov::op::v1::Transpose>(input_a, transpose_a_const);
        auto input_b = std::make_shared<ov::op::v0::Parameter>(input_type, ov::PartialShape::dynamic(4));
        auto transpose_b_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose_b = std::make_shared<ov::op::v1::Transpose>(input_b, transpose_b_const);
        auto input_c = std::make_shared<ov::op::v0::Parameter>(input_type, ov::PartialShape::dynamic(4));
        auto transpose_c_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose_c = std::make_shared<ov::op::v1::Transpose>(input_c, transpose_c_const);

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transpose_a, transpose_b, transpose_c, is_causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        manager.register_pass<TransposeFusion>();
    }
    {
        std::vector<int64_t> order_a = {0, 2, 1, 3};
        std::vector<int64_t> order_b = {0, 2, 1, 3};
        std::vector<int64_t> order_c = {0, 2, 1, 3};
        std::vector<int64_t> order_output = {0, 1, 2, 3};
        auto input_a = std::make_shared<ov::op::v0::Parameter>(input_type, ov::PartialShape::dynamic(4));
        auto input_b = std::make_shared<ov::op::v0::Parameter>(input_type, ov::PartialShape::dynamic(4));
        auto input_c = std::make_shared<ov::op::v0::Parameter>(input_type, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{input_a, input_b, input_c},
                                                              is_causal,
                                                              order_a,
                                                              order_b,
                                                              order_c,
                                                              order_output,
                                                              ov::element::dynamic);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{input_a, input_b, input_c});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

INSTANTIATE_TEST_SUITE_P(TransposeSDPAFusionInputPrecision,
                         TransposeSDPAFusionPrecisionTest,
                         ::testing::ValuesIn(ov::intel_gpu::op::SDPA::get_supported_precisions()),
                         [](const ::testing::TestParamInfo<ov::element::Type>& info) {
                             return info.param.get_type_name();
                         });

namespace {

using ov::op::v0::Constant;

std::shared_ptr<ov::Node> i64_const(const std::vector<int64_t>& values, bool scalar = false) {
    return Constant::create(ov::element::i64, scalar ? ov::Shape{} : ov::Shape{values.size()}, values);
}

// Fused QKV [N, S, 3*H*D] -> Reshape [N, S, 3, H, D], as produced by attention blocks.
std::shared_ptr<ov::Node> qkv_5d(const std::shared_ptr<ov::op::v0::Parameter>& qkv) {
    return std::make_shared<ov::op::v1::Reshape>(qkv, i64_const({0, 0, 3, 6, 32}), true);
}

// Expected rewrite: [N, S, 3, H, D] -> Reshape [N, S, 3*H, D] -> Split(axis 2) -> Transpose(0, 2, 1, 3)
ov::OutputVector merged_split_ref(const std::shared_ptr<ov::Node>& qkv_5d_node) {
    auto merged = std::make_shared<ov::op::v1::Reshape>(qkv_5d_node, i64_const({0, 0, 18, 32}), true);
    auto split = std::make_shared<ov::op::v1::Split>(merged, i64_const({2}, true), 3);
    ov::OutputVector res;
    for (size_t i = 0; i < 3; ++i) {
        res.push_back(std::make_shared<ov::op::v1::Transpose>(split->output(i), i64_const({0, 2, 1, 3})));
    }
    return res;
}

std::shared_ptr<ov::Node> q_scaled(const ov::Output<ov::Node>& q) {
    return std::make_shared<ov::op::v1::Multiply>(q, Constant::create(ov::element::f16, ov::Shape{}, {0.25f}));
}

}  // namespace

// [N,S,3,H,D] -> Transpose(0,3,2,1,4) -> VariadicSplit(axis 2) -> Squeeze(2) -> [Q scale] -> SDPA
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_VariadicSplitSqueeze) {
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv_5d(qkv), i64_const({0, 3, 2, 1, 4}));
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(transpose, i64_const({2}, true), i64_const({1, 1, 1}));
        auto q = std::make_shared<ov::op::v0::Squeeze>(split->output(0), i64_const({2}));
        auto k = std::make_shared<ov::op::v0::Squeeze>(split->output(1), i64_const({2}));
        auto v = std::make_shared<ov::op::v0::Squeeze>(split->output(2), i64_const({2}));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled(q), k, v, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
        manager.register_pass<SinkTransposeThroughSplitForSDPA>();
    }
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
        auto qkv_inputs = merged_split_ref(qkv_5d(qkv));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_scaled(qkv_inputs[0]), qkv_inputs[1], qkv_inputs[2], false);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// Same pattern after CommonOptimizations has turned the Squeeze ops into unit-axis-dropping Reshapes.
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_SplitReshape) {
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv_5d(qkv), i64_const({0, 3, 2, 1, 4}));
        auto split = std::make_shared<ov::op::v1::Split>(transpose, i64_const({2}, true), 3);
        ov::OutputVector qkv_inputs;
        for (size_t i = 0; i < 3; ++i) {
            qkv_inputs.push_back(std::make_shared<ov::op::v1::Reshape>(split->output(i), i64_const({64, 6, 256, 32}), false));
        }
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(qkv_inputs[0], qkv_inputs[1], qkv_inputs[2], false);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
        manager.register_pass<SinkTransposeThroughSplitForSDPA>();
    }
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
        auto qkv_inputs = merged_split_ref(qkv_5d(qkv));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(qkv_inputs[0], qkv_inputs[1], qkv_inputs[2], false);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// timm-style qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4); q, k, v = qkv[0], qkv[1], qkv[2]
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_Gather) {
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv_5d(qkv), i64_const({2, 0, 3, 1, 4}));
        ov::OutputVector qkv_inputs;
        for (int64_t i = 0; i < 3; ++i) {
            qkv_inputs.push_back(std::make_shared<ov::op::v8::Gather>(transpose, i64_const({i}, true), i64_const({0}, true)));
        }
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(qkv_inputs[0], qkv_inputs[1], qkv_inputs[2], false);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
        manager.register_pass<SinkTransposeThroughSplitForSDPA>();
    }
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
        auto qkv_inputs = merged_split_ref(qkv_5d(qkv));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(qkv_inputs[0], qkv_inputs[1], qkv_inputs[2], false);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// A dynamic head count prevents merging the split axis, so the slices are squeezed instead.
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_DynamicHeadsSqueezeFallback) {
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 256, 3, -1, 32});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv, i64_const({0, 3, 2, 1, 4}));
        auto split = std::make_shared<ov::op::v1::Split>(transpose, i64_const({2}, true), 3);
        auto q = std::make_shared<ov::op::v0::Squeeze>(split->output(0), i64_const({2}));
        auto k = std::make_shared<ov::op::v0::Squeeze>(split->output(1), i64_const({2}));
        auto v = std::make_shared<ov::op::v0::Squeeze>(split->output(2), i64_const({2}));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
        manager.register_pass<SinkTransposeThroughSplitForSDPA>();
    }
    {
        auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 256, 3, -1, 32});
        auto split = std::make_shared<ov::op::v1::Split>(qkv, i64_const({2}, true), 3);
        ov::OutputVector qkv_inputs;
        for (size_t i = 0; i < 3; ++i) {
            auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(i), i64_const({2}, true));
            qkv_inputs.push_back(std::make_shared<ov::op::v1::Transpose>(squeeze, i64_const({0, 2, 1, 3})));
        }
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(qkv_inputs[0], qkv_inputs[1], qkv_inputs[2], false);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// A split output that is also consumed outside SDPA keeps the original Transpose.
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_NonSDPAConsumer) {
    auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv_5d(qkv), i64_const({0, 3, 2, 1, 4}));
    auto split = std::make_shared<ov::op::v1::Split>(transpose, i64_const({2}, true), 3);
    auto q = std::make_shared<ov::op::v0::Squeeze>(split->output(0), i64_const({2}));
    auto k = std::make_shared<ov::op::v0::Squeeze>(split->output(1), i64_const({2}));
    auto v = std::make_shared<ov::op::v0::Squeeze>(split->output(2), i64_const({2}));
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);
    auto extra = std::make_shared<ov::op::v0::Result>(v);

    model = std::make_shared<ov::Model>(ov::OutputVector{sdpa, extra}, ov::ParameterVector{qkv});
    manager.register_pass<SinkTransposeThroughSplitForSDPA>();
}

// The remaining transpose would move the head-size axis, which SDPA cannot absorb.
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_HeadAxisMoved) {
    auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{64, 256, 576});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv_5d(qkv), i64_const({0, 4, 2, 3, 1}));
    auto split = std::make_shared<ov::op::v1::Split>(transpose, i64_const({2}, true), 3);
    auto q = std::make_shared<ov::op::v0::Squeeze>(split->output(0), i64_const({2}));
    auto k = std::make_shared<ov::op::v0::Squeeze>(split->output(1), i64_const({2}));
    auto v = std::make_shared<ov::op::v0::Squeeze>(split->output(2), i64_const({2}));
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);

    model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
    manager.register_pass<SinkTransposeThroughSplitForSDPA>();
}

// Squeeze keeps a selected axis whose size is not 1: the split slices here have 2 elements on the
// squeezed axis, so Q/K/V stay 4D with 2 heads and the transpose must not be sunk.
TEST_F(TransformationTestsF, SinkTransposeThroughSplitForSDPA_NonUnitSliceKept) {
    auto qkv = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 256, 6, 32});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(qkv, i64_const({0, 2, 1, 3}));
    auto split = std::make_shared<ov::op::v1::Split>(transpose, i64_const({1}, true), 3);
    auto q = std::make_shared<ov::op::v0::Squeeze>(split->output(0), i64_const({1}));
    auto k = std::make_shared<ov::op::v0::Squeeze>(split->output(1), i64_const({1}));
    auto v = std::make_shared<ov::op::v0::Squeeze>(split->output(2), i64_const({1}));
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);

    model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{qkv});
    manager.register_pass<SinkTransposeThroughSplitForSDPA>();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
