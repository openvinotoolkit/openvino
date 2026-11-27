// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/l2_norm_to_rms.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/variable.hpp"
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

// MatMul(x, W): not a residual stream (a model input itself counts as one)
NormInput matmul_output() {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
    return {std::make_shared<ov::op::v0::MatMul>(x, ov::op::v0::Constant::create(ov::element::f32, {hidden, hidden}, {0.01f})), {x}};
}

// (h + b) + (branch + bias) with h, b, branch MatMul outputs: a residual stream, the final Add has two non-constant
// inputs. The walk stops at the MatMuls, so only the residual Add makes the norm residual-fed (not a model input).
NormInput residual_stream() {
    auto h = matmul_output();
    auto b = matmul_output();
    auto branch = matmul_output();
    auto prev_residual = std::make_shared<ov::op::v1::Add>(h.x, b.x);
    auto residual = std::make_shared<ov::op::v1::Add>(prev_residual, add_bias(branch.x));
    return {residual, {h.params[0], b.params[0], branch.params[0]}};
}

// MatMul + bias: an Add with a constant input is a bias, not a residual connection
NormInput bias_only() {
    auto in = matmul_output();
    return {add_bias(in.x), in.params};
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

namespace {
// A node of the decomposed norm that is already marked to stay in FP32 before the pass, e.g. by the model's rt_info.
enum class MarkedNode {
    Divide,
    OutputConvert,         // the trailing Convert that the RMS absorbs
    DivideFromF32,         // disable_conversion(node, f32, f16): only for its own source type
    OutputConvertFromF32,  // disable_conversion(node, f32, f16): only for its own source type
};

bool on_output_convert(MarkedNode marked) {
    return marked == MarkedNode::OutputConvert || marked == MarkedNode::OutputConvertFromF32;
}

// Other rt_info of the absorbed output Convert, which the RMS takes over.
constexpr const char* output_convert_rt_key = "l2_norm_to_rms_test_output_convert";

// A norm fed by a MatMul (not a residual connection), with one node marked to stay in FP32.
std::shared_ptr<ov::Model> norm_with_marked_node(MarkedNode marked) {
    auto in = matmul_output();
    auto norm = l2_norm(in.x, constant_gamma());
    std::shared_ptr<ov::Node> out = norm;
    std::shared_ptr<ov::Node> node = norm->get_input_node_shared_ptr(0);  // the Divide
    if (on_output_convert(marked)) {
        out = std::make_shared<ov::op::v0::Convert>(norm, ov::element::f32);
        out->get_rt_info()[output_convert_rt_key] = std::string("kept");
        node = out;
    }
    if (marked == MarkedNode::DivideFromF32 || marked == MarkedNode::OutputConvertFromF32) {
        ov::disable_conversion(node, ov::element::f32, ov::element::f16);
    } else {
        ov::disable_conversion(node, ov::element::f16);
    }
    return make_model(out, in.params);
}

const std::vector<MarkedNode> marked_nodes{MarkedNode::Divide, MarkedNode::OutputConvert, MarkedNode::DivideFromF32, MarkedNode::OutputConvertFromF32};

class L2NormToRMSMarkedNode : public TransformationTestsF, public WithParamInterface<MarkedNode> {};

class L2NormToRMSMarkedNodePipeline : public TestWithParam<MarkedNode> {};
}  // namespace

// The fusion removes the marked node: the RMS gets its FP32 mark (and the rt_info of an absorbed output Convert).
TEST_P(L2NormToRMSMarkedNode, MarkKeptOnRMS) {
    model = norm_with_marked_node(GetParam());
    {
        auto in = matmul_output();
        auto fused = rms(in.x, constant_gamma(), ov::element::f32);
        ov::disable_conversion(fused, ov::element::f16);
        if (on_output_convert(GetParam())) {
            fused->get_rt_info()[output_convert_rt_key] = std::string("kept");
        }
        model_ref = make_model(fused, in.params);
    }
    manager.register_pass<L2NormToRMS>(1024);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// After ConvertPrecision the RMS still computes and writes f32. In f16 a large input (e.g. 70000) would give inf and
// then NaN.
TEST_P(L2NormToRMSMarkedNodePipeline, RMSStaysFP32AfterConvertPrecision) {
    auto m = norm_with_marked_node(GetParam());
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(fused->get_output_element_type(0), ov::element::f32);
}

INSTANTIATE_TEST_SUITE_P(smoke, L2NormToRMSMarkedNode, ValuesIn(marked_nodes));
INSTANTIATE_TEST_SUITE_P(smoke, L2NormToRMSMarkedNodePipeline, ValuesIn(marked_nodes));

// A residual Add of a MatMul output and a ReadValue with a Constant initializer: the state changes at run time, so the
// Add is a residual connection and stays in FP32 (in f16 the sum can overflow).
TEST(L2NormToRMSPipeline, ResidualWithReadValueStaysFP32) {
    const ov::Shape shape{1, 16, hidden};
    auto in = matmul_output();
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{shape, ov::element::f32, "residual_state"});
    auto state = std::make_shared<ov::op::v6::ReadValue>(ov::op::v0::Constant::create(ov::element::f32, shape, {0.f}), variable);
    auto residual = std::make_shared<ov::op::v1::Add>(in.x, state);
    auto assign = std::make_shared<ov::op::v6::Assign>(residual, variable);
    auto m = std::make_shared<ov::Model>(ov::OutputVector{l2_norm(residual, constant_gamma())}, ov::SinkVector{assign}, in.params);
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_TRUE(ov::is_conversion_disabled(fused, ov::element::f16));
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(residual->get_output_element_type(0), ov::element::f32);
}

namespace {
// The norm is not residual-fed, so after ConvertPrecision it runs in f16. The graph comparison in the tests above
// does not catch an extra FP32 mark: it checks only that the reference's rt_info keys are present.
void expect_norm_runs_in_f16(const NormInput& in) {
    auto m = make_model(l2_norm(in.x, constant_gamma()), in.params);
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_FALSE(ov::is_conversion_disabled(fused, ov::element::f16));
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(fused->get_output_element_type(0), ov::element::f16);
}
}  // namespace

TEST(L2NormToRMSPipeline, BiasFedNormRunsInF16) {
    expect_norm_runs_in_f16(bias_only());
}

// A q/k-norm-like L2 norm fed by a MatMul (not a residual connection) runs in f16 after ConvertPrecision.
TEST(L2NormToRMSPipeline, NonResidualNormRunsInF16) {
    expect_norm_runs_in_f16(matmul_output());
}

// A norm fed directly by a model input stays in FP32, as the decomposed norm did: the input may carry a residual
// stream, e.g. in a model split per layer.
TEST(L2NormToRMSPipeline, ModelInputFedNormStaysFP32) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, hidden});
    auto m = make_model(l2_norm(x, constant_gamma()), {x});
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_TRUE(ov::is_conversion_disabled(fused, ov::element::f16));
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(fused->get_output_element_type(0), ov::element::f32);
}

// gamma = (Convert(u16 model input) - zp) * scale, as ONNX Runtime with the OpenVINO EP feeds a shared gamma
// initializer: after ConvertPrecision the Convert of the u16 input still converts to f32 (to f16, 65535 is inf), and
// the dequantization and the scaled gamma stay in FP32. x is a MatMul output (a PSX q/k norm), so the RMS itself runs
// in f16 and nothing but the gamma marks keeps the gamma path in FP32.
TEST(L2NormToRMSPipeline, U16GammaModelInputConvertStaysFP32) {
    auto in = matmul_output();
    auto weight = std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{hidden});
    auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32);
    auto zp = std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(ov::element::u16, {}, {127}), ov::element::f32);
    auto shifted = std::make_shared<ov::op::v1::Subtract>(weight_convert, zp);
    auto gamma = std::make_shared<ov::op::v1::Multiply>(shifted, ov::op::v0::Constant::create(ov::element::f32, {}, {0.000855f}));
    auto m = make_model(l2_norm(in.x, gamma), {in.params[0], weight});
    fuse_and_convert_to_f16(m);

    const auto fused = single_rms(m);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_input_element_type(0), ov::element::f16);
    // The scaled gamma is computed in FP32 and converted to f16 for the f16 RMS (it fits in f16).
    const auto* gamma_to_rms = fused->get_input_node_ptr(1);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(gamma_to_rms));
    const auto* scaled_gamma = gamma_to_rms->get_input_node_ptr(0);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(scaled_gamma));
    EXPECT_EQ(scaled_gamma->get_input_node_ptr(0), gamma.get());
    EXPECT_EQ(scaled_gamma->get_output_element_type(0), ov::element::f32);
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
