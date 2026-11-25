// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/l2_norm_to_rms.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr size_t hidden = 64;
constexpr float eps = 1e-6f;

// x / sqrt(ReduceSum(x^2, axis) + eps) * gamma, as an ONNX LpNormalization(p=2) followed by a scale exports it.
std::shared_ptr<ov::Node> l2_norm(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, int64_t axis = -1, bool keep_dims = true) {
    auto power = std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(x.get_element_type(), {}, {2.f}));
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, axes, keep_dims);
    auto add = std::make_shared<ov::op::v1::Add>(sum, ov::op::v0::Constant::create(x.get_element_type(), {}, {eps}));
    auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add);
    auto div = std::make_shared<ov::op::v1::Divide>(x, sqrt);
    return std::make_shared<ov::op::v1::Multiply>(div, gamma);
}

std::shared_ptr<ov::Node> rms(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gamma, ov::element::Type out_type) {
    auto inv_sqrt_n = ov::op::v0::Constant::create(gamma.get_element_type(), {}, {1.0 / std::sqrt(static_cast<double>(hidden))});
    auto scaled_gamma = std::make_shared<ov::op::v1::Multiply>(gamma, inv_sqrt_n);
    ov::disable_conversion(scaled_gamma, ov::element::f16);
    return std::make_shared<ov::op::internal::RMS>(x, scaled_gamma, eps / static_cast<float>(hidden), out_type);
}

}  // namespace

TEST_F(TransformationTestsF, L2NormToRMS) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto norm = l2_norm(x, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// f16 activations normalized in f32 and converted back: RMS takes the f16 input and writes f16 directly.
TEST_F(TransformationTestsF, L2NormToRMSWithConverts) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        auto x_f32 = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        auto norm = l2_norm(x_f32, gamma);
        auto out = std::make_shared<ov::op::v0::Convert>(norm, ov::element::f16);
        model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f16)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Not normalized over the last axis: left alone.
TEST_F(TransformationTestsF, L2NormToRMSNotLastAxis) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    auto norm = l2_norm(x, gamma, 1);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// Without keep_dims the reduced tensor does not broadcast against x the way RMS needs: left alone.
TEST_F(TransformationTestsF, L2NormToRMSNoKeepDims) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma, -1, false);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// A row too long for one work group of the RMS kernel (hidden / 8 > max work group size): left alone.
TEST_F(TransformationTestsF, L2NormToRMSRowTooLong) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(4);
}

// gamma {1, 1, n} still scales the normalized axis only: fused.
TEST_F(TransformationTestsF, L2NormToRMSGammaWithLeadingOnes) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
        auto norm = l2_norm(x, gamma);
        model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
        manager.register_pass<L2NormToRMS>(1024);
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
        auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, hidden});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms(x, gamma, ov::element::f32)}, ov::ParameterVector{x, gamma});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// gamma {n, 1} has n elements but scales each row, not the normalized axis: left alone.
TEST_F(TransformationTestsF, L2NormToRMSGammaColumn) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, hidden, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden, 1});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// gamma with more dims than x would change the output rank: left alone.
TEST_F(TransformationTestsF, L2NormToRMSGammaHigherRank) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

// Rank 6 has no RMS implementation on GPU: left alone.
TEST_F(TransformationTestsF, L2NormToRMSRankSix) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 1, 4, hidden});
    auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{hidden});
    auto norm = l2_norm(x, gamma);
    model = std::make_shared<ov::Model>(ov::OutputVector{norm}, ov::ParameterVector{x, gamma});
    manager.register_pass<L2NormToRMS>(1024);
}

namespace {

// A model output together with the parameters it depends on, so a test builds model and model_ref separately.
struct NormInput {
    ov::Output<ov::Node> x;
    ov::ParameterVector params;
};

std::shared_ptr<ov::Node> add_bias(const ov::Output<ov::Node>& x) {
    return std::make_shared<ov::op::v1::Add>(x, ov::op::v0::Constant::create(ov::element::f32, {hidden}, {0.5f}));
}

std::shared_ptr<ov::Node> constant_gamma() {
    return ov::op::v0::Constant::create(ov::element::f32, {hidden}, {1.5f});
}

// (h + b) + (branch + bias): a residual stream, the final Add has two non-constant inputs
NormInput residual_stream() {
    const ov::Shape shape{1, 16, hidden};
    auto h = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto prev_residual = std::make_shared<ov::op::v1::Add>(h, b);
    auto residual = std::make_shared<ov::op::v1::Add>(prev_residual, add_bias(branch));
    return {residual, {h, b, branch}};
}

// branch + bias: an Add with a constant input is a bias, not a residual connection
NormInput bias_only() {
    auto branch = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
    return {add_bias(branch), {branch}};
}

// gamma = (Convert(u16 weight) - zp) * scale with weight up to 65535 (above the f16 range)
std::shared_ptr<ov::Node> u16_dequantized_gamma() {
    std::vector<uint16_t> q(hidden, 40000);
    q[0] = 65535;
    auto weight = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::u16, {hidden}, q), ov::element::f32);
    auto zp = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::u16, {}, {127}), ov::element::f32);
    auto shifted = std::make_shared<ov::op::v1::Subtract>(weight, zp);
    return std::make_shared<ov::op::v1::Multiply>(shifted, ov::op::v0::Constant::create(ov::element::f32, {}, {0.000855f}));
}

std::shared_ptr<ov::Model> make_model(const std::shared_ptr<ov::Node>& out, const ov::ParameterVector& params) {
    return std::make_shared<ov::Model>(ov::OutputVector{out}, params);
}

// What lies between the residual Add and the norm. The FP32 mark propagates up through each of these ops.
enum class ResidualLink {
    Direct,
    BiasAdd,    // Add of a constant bias
    Reshape,    // rank-changing Reshape {1, 16, n} -> {16, n}
    Abs,        // unary elementwise arithmetic
    Broadcast,  // {1, 16, n} -> {2, 16, n}
};

ov::Output<ov::Node> link_to_norm(const ov::Output<ov::Node>& residual, ResidualLink link) {
    switch (link) {
    case ResidualLink::Direct:
        return residual;
    case ResidualLink::BiasAdd:
        return add_bias(residual);
    case ResidualLink::Reshape:
        return std::make_shared<ov::op::v1::Reshape>(residual, ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{16, hidden}), false);
    case ResidualLink::Abs:
        return std::make_shared<ov::op::v0::Abs>(residual);
    case ResidualLink::Broadcast:
        return std::make_shared<ov::op::v3::Broadcast>(residual, ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{2, 16, hidden}));
    }
    OPENVINO_THROW("unknown ResidualLink");
}

// L2NormToRMS followed by the f32 -> f16 ConvertPrecision of the GPU pipeline (precision-sensitive subgraphs kept in
// FP32, model inputs and outputs keep their types).
void fuse_and_convert_to_f16(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<L2NormToRMS>(1024);
    manager.register_pass<ov::pass::ConvertPrecision>(ov::element::f32, ov::element::f16, type_to_fuse_map{}, true, false);
    manager.run_passes(model);
}

std::shared_ptr<ov::op::internal::RMS> single_rms(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::internal::RMS> found;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto rms = ov::as_type_ptr<ov::op::internal::RMS>(op)) {
            EXPECT_EQ(found, nullptr) << "expected a single RMS";
            found = rms;
        }
    }
    return found;
}

const std::vector<ResidualLink> residual_links{ResidualLink::Direct, ResidualLink::BiasAdd, ResidualLink::Reshape, ResidualLink::Abs, ResidualLink::Broadcast};

class L2NormToRMSResidualFed : public TransformationTestsF, public WithParamInterface<ResidualLink> {};

class L2NormToRMSResidualFedPipeline : public TestWithParam<ResidualLink> {};

}  // namespace

// A residual-fed norm is fused and its RMS is marked to stay in FP32, so ConvertPrecision keeps the residual stream
// in FP32 (as MarkDivWithEps did for the decomposed norm).
TEST_P(L2NormToRMSResidualFed, FusedAndMarkedFP32) {
    {
        auto in = residual_stream();
        model = make_model(l2_norm(link_to_norm(in.x, GetParam()), constant_gamma()), in.params);
    }
    {
        auto in = residual_stream();
        auto fused = rms(link_to_norm(in.x, GetParam()), constant_gamma(), ov::element::f32);
        ov::disable_conversion(fused, ov::element::f16);
        model_ref = make_model(fused, in.params);
    }
    manager.register_pass<L2NormToRMS>(1024);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// After ConvertPrecision the residual Adds and the RMS stay in FP32.
TEST_P(L2NormToRMSResidualFedPipeline, ResidualStreamStaysFP32AfterConvertPrecision) {
    auto in = residual_stream();
    auto m = make_model(l2_norm(link_to_norm(in.x, GetParam()), constant_gamma()), in.params);
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(fused->get_output_element_type(0), ov::element::f32);
    for (const auto& op : m->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Add>(op)) {
            EXPECT_EQ(op->get_output_element_type(0), ov::element::f32) << op->get_friendly_name();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, L2NormToRMSResidualFed, ValuesIn(residual_links));
INSTANTIATE_TEST_SUITE_P(smoke, L2NormToRMSResidualFedPipeline, ValuesIn(residual_links));

// A norm fed by a bias Add is not residual-fed: fused without the FP32 mark.
TEST_F(TransformationTestsF, L2NormToRMS_BiasFedNormFused) {
    {
        auto in = bias_only();
        model = make_model(l2_norm(in.x, constant_gamma()), in.params);
    }
    {
        auto in = bias_only();
        model_ref = make_model(rms(in.x, constant_gamma(), ov::element::f32), in.params);
    }
    manager.register_pass<L2NormToRMS>(1024);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// A q/k-norm-like L2 norm fed by a MatMul (not a residual connection) runs in f16 after ConvertPrecision.
TEST(L2NormToRMSPipeline, NonResidualNormRunsInF16) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
    auto q = std::make_shared<ov::op::v0::MatMul>(x, ov::op::v0::Constant::create(ov::element::f32, {hidden, hidden}, {0.01f}));
    auto m = make_model(l2_norm(q, constant_gamma()), {x});
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(fused->get_output_element_type(0), ov::element::f16);
}

// gamma = (Convert(u16 model input) - zp) * scale, as ONNX Runtime with the OpenVINO EP feeds a shared gamma
// initializer: after ConvertPrecision the Convert of the u16 input still converts to f32 (to f16, 65535 is inf), and
// the dequantization stays in FP32. (The RMS itself may run in f16; the scaled gamma fits in f16.)
TEST(L2NormToRMSPipeline, U16GammaModelInputConvertStaysFP32) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
    auto weight = std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{hidden});
    auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32);
    auto zp = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::u16, {}, {127}), ov::element::f32);
    auto shifted = std::make_shared<ov::op::v1::Subtract>(weight_convert, zp);
    auto gamma = std::make_shared<ov::op::v1::Multiply>(shifted, ov::op::v0::Constant::create(ov::element::f32, {}, {0.000855f}));
    auto m = make_model(l2_norm(x, gamma), {x, weight});
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    ASSERT_EQ(weight->get_output_target_inputs(0).size(), 1);
    const auto* convert = weight->get_output_target_inputs(0).begin()->get_node();
    ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(convert));
    EXPECT_EQ(convert->get_output_element_type(0), ov::element::f32);
    for (const ov::Node* node : std::initializer_list<const ov::Node*>{shifted.get(), gamma.get()}) {
        EXPECT_EQ(node->get_output_element_type(0), ov::element::f32) << node->get_friendly_name();
    }
}

// A norm fused before ConvertPrecision with gamma dequantized from u16: the scaled gamma is marked to stay in FP32
// (no Divide is left for MarkDivWithEps, and a u16 -> f16 conversion of 65535 gives inf).
TEST_F(TransformationTestsF, L2NormToRMS_U16DequantizedGammaKeptFP32) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
        model = make_model(l2_norm(x, u16_dequantized_gamma()), {x});
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
        model_ref = make_model(rms(x, u16_dequantized_gamma(), ov::element::f32), {x});
    }
    manager.register_pass<L2NormToRMS>(1024);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
