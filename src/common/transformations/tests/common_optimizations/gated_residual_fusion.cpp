// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_residual_fusion.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"

using namespace testing;
using namespace ov;

namespace {

constexpr size_t C = 4;
constexpr size_t D = 8;
constexpr size_t L = 6;
constexpr size_t CD = C * D;
constexpr float EPS = 1e-6F;

using Gr = op::internal::GatedResidual;

// Reshape with special zero: a 0 keeps the (dynamic) dimension of the input.
std::shared_ptr<op::v1::Reshape> reshape_to(const Output<Node>& src, const std::vector<int64_t>& shape) {
    auto target = op::v0::Constant::create(element::i64, Shape{shape.size()}, shape);
    return std::make_shared<op::v1::Reshape>(src, target, true);
}

// Only the last dimension of the activations is fixed: the leading ones (rows) are dynamic.
PartialShape with_last(PartialShape rows, size_t last) {
    rows.push_back(Dimension(static_cast<int64_t>(last)));
    return rows;
}

// Reshape target that keeps `leading` dimensions and appends `tail`.
std::vector<int64_t> keep_rows(size_t leading, const std::vector<int64_t>& tail) {
    std::vector<int64_t> target(leading, 0);
    target.insert(target.end(), tail.begin(), tail.end());
    return target;
}

std::shared_ptr<op::v0::Constant> scalar(float v) {
    return op::v0::Constant::create(element::f32, Shape{}, {v});
}

std::shared_ptr<op::v0::Constant> random_f32(const Shape& shape, float value) {
    return op::v0::Constant::create(element::f32, shape, std::vector<float>(shape_size(shape), value));
}

// NNCF-style compressed weight [N, G, K/G] -> Convert -> Subtract(zp) -> Multiply(scale) -> Reshape [N, K]. The
// scale and zero point are [N, G, 1]; the pass turns them into the op's [N, G] layout.
Output<Node> compressed_weight(size_t n, size_t k, size_t groups) {
    const size_t gs = k / groups;
    Tensor w_tensor(element::u4, Shape{n, groups, gs});
    std::memset(w_tensor.data(), 0x5A, w_tensor.get_byte_size());
    Tensor zp_tensor(element::u4, Shape{n, groups, 1});
    std::memset(zp_tensor.data(), 0x33, zp_tensor.get_byte_size());
    auto weights = std::make_shared<op::v0::Constant>(w_tensor);
    auto zp = std::make_shared<op::v0::Constant>(zp_tensor);
    auto scale = random_f32(Shape{n, groups, 1}, 0.02F);
    auto w_f32 = std::make_shared<op::v0::Convert>(weights, element::f32);
    auto zp_f32 = std::make_shared<op::v0::Convert>(zp, element::f32);
    auto sub = std::make_shared<op::v1::Subtract>(w_f32, zp_f32);
    auto mul = std::make_shared<op::v1::Multiply>(sub, scale);
    return reshape_to(mul, {static_cast<int64_t>(n), static_cast<int64_t>(k)})->output(0);
}

// The projection weight of a GEMM: [N, K] stored with transpose_b, either as a plain f32 constant or compressed.
Output<Node> weight(size_t n, size_t k, bool compressed) {
    if (compressed) {
        return compressed_weight(n, k, 2);
    }
    return random_f32(Shape{n, k}, 0.01F);
}

struct ReadOptions {
    bool with_gate = false;
    bool compressed = false;
    bool rms_fused = false;    // RMS op instead of the decomposed normalization
    bool mix_scaled = true;    // the 1/C of the branch mix
    int64_t reduce_axis = -2;  // branch axis
    size_t leading = 2;        // dynamic leading dimensions of the activations
};

// Canonical READ graph of the layer, decomposed as checkpoints export it.
std::shared_ptr<Model> make_read_model(const ReadOptions& opt) {
    auto residual =
        std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(opt.leading), CD));
    auto gain = std::make_shared<op::v0::Parameter>(element::f32, Shape{CD});

    const auto split = keep_rows(opt.leading, {static_cast<int64_t>(C), static_cast<int64_t>(D)});
    const auto flat = keep_rows(opt.leading, {static_cast<int64_t>(CD)});
    auto r3 = reshape_to(residual, split);
    Output<Node> rbar_flat;
    if (opt.rms_fused) {
        auto gamma = random_f32(Shape{C, D}, 1.0F);
        auto rms = std::make_shared<ov::op::internal::RMS>(r3, gamma, EPS);
        rbar_flat = reshape_to(rms, flat)->output(0);
    } else {
        auto sq = std::make_shared<op::v1::Power>(r3, scalar(2.0F));
        auto mean =
            std::make_shared<op::v1::ReduceMean>(sq, op::v0::Constant::create(element::i64, Shape{1}, {-1}), true);
        auto biased = std::make_shared<op::v1::Add>(mean, scalar(EPS));
        auto rms = std::make_shared<op::v0::Sqrt>(biased);
        auto inv = std::make_shared<op::v1::Divide>(scalar(1.0F), rms);
        auto normalized = std::make_shared<op::v1::Multiply>(r3, inv);
        rbar_flat = std::make_shared<op::v1::Multiply>(reshape_to(normalized, flat), gain)->output(0);
    }

    auto low = std::make_shared<op::v0::MatMul>(rbar_flat, weight(L, CD, opt.compressed), false, true);
    auto low_scaled = std::make_shared<op::v1::Multiply>(low, scalar(1.0F / C));
    auto act = std::make_shared<op::v4::Swish>(low_scaled);
    auto up = std::make_shared<op::v0::MatMul>(act, weight(CD, L, opt.compressed), false, true);
    auto read_gate = std::make_shared<op::v0::Sigmoid>(up);
    auto weighted = std::make_shared<op::v1::Multiply>(read_gate, rbar_flat);
    auto weighted3 = reshape_to(weighted, split);
    auto reduced =
        std::make_shared<op::v1::ReduceSum>(weighted3,
                                            op::v0::Constant::create(element::i64, Shape{1}, {opt.reduce_axis}),
                                            false);
    Output<Node> mixed = reduced->output(0);
    if (opt.mix_scaled) {
        mixed = std::make_shared<op::v1::Multiply>(reduced, scalar(1.0F / C))->output(0);
    }

    ResultVector results{std::make_shared<op::v0::Result>(mixed)};
    ParameterVector params{residual, gain};
    if (opt.with_gate) {
        auto inject = std::make_shared<op::v0::MatMul>(rbar_flat, weight(C, CD, opt.compressed), false, true);
        auto inject_scaled = std::make_shared<op::v1::Multiply>(inject, scalar(1.0F / C));
        auto gate = std::make_shared<op::v0::Sigmoid>(inject_scaled);
        auto write_gate = std::make_shared<op::v1::Multiply>(gate, scalar(2.0F));
        results.push_back(std::make_shared<op::v0::Result>(write_gate));
    }
    return std::make_shared<Model>(results, params);
}

// R' = R + Reshape(S * Y), the exporter's form: the Multiply is flattened to [.., C*D] and added to the flat residual.
// `y_rows` are the leading dimensions of Y. With `split` the Add is in the [.., C, D] domain instead (the expected
// form).
std::shared_ptr<Model> make_write_model(const PartialShape& rows, const PartialShape& y_rows, bool split) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(element::f32, with_last(y_rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, C));

    const auto leading = rows.size();
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto scaled = std::make_shared<op::v1::Multiply>(s3, y3);
    std::shared_ptr<Node> out;
    if (split) {
        auto r3 = reshape_to(residual, keep_rows(leading, {static_cast<int64_t>(C), static_cast<int64_t>(D)}));
        out = reshape_to(std::make_shared<op::v1::Add>(r3, scaled), keep_rows(leading, {static_cast<int64_t>(CD)}));
    } else {
        out =
            std::make_shared<op::v1::Add>(residual, reshape_to(scaled, keep_rows(leading, {static_cast<int64_t>(CD)})));
    }
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                   ParameterVector{residual, y, gate});
}

std::shared_ptr<Model> make_write_model(const PartialShape& rows = PartialShape::dynamic(2), bool split = false) {
    return make_write_model(rows, rows, split);
}

void run_fusion(const std::shared_ptr<Model>& model) {
    pass::Manager manager;
    manager.register_pass<pass::GatedResidualFusion>();
    manager.run_passes(model);
}

// Runs the fusion with a callback that rejects the fused ops it returns true for.
void run_fusion_with_callback(const std::shared_ptr<Model>& model,
                              const std::function<bool(const std::shared_ptr<const Node>&)>& reject) {
    pass::Manager manager;
    manager.register_pass<pass::GatedResidualFusion>();
    manager.get_pass_config()
        ->set_callback<pass::GatedResidualReadFusion,
                       pass::GatedResidualWriteFusion,
                       pass::GatedResidualWriteSinkReshape>([&reject](const std::shared_ptr<const Node>& node) {
            return reject(node);
        });
    manager.run_passes(model);
}

// Runs the fusion pass with the WRITE fusion disabled (what the CPU plugin does), so the Add is only moved.
void run_split(const std::shared_ptr<Model>& model) {
    pass::Manager manager;
    manager.register_pass<pass::GatedResidualFusion>();
    manager.get_pass_config()->disable<pass::GatedResidualWriteFusion>();
    manager.run_passes(model);
}

std::vector<std::shared_ptr<Gr>> gated_residuals(const std::shared_ptr<Model>& model) {
    std::vector<std::shared_ptr<Gr>> result;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto gr = ov::as_type_ptr<Gr>(node)) {
            result.push_back(gr);
        }
    }
    return result;
}

size_t count_ops(const std::shared_ptr<Model>& model, const DiscreteTypeInfo& type) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_info() == type) {
            ++count;
        }
    }
    return count;
}

// True when the result is Reshape(Add(.., Multiply)): the Add sits in the split domain next to the Multiply.
bool add_is_split(const std::shared_ptr<Model>& model) {
    const auto flat = model->get_results()[0]->get_input_node_shared_ptr(0);
    if (!ov::is_type<op::v1::Reshape>(flat)) {
        return false;
    }
    const auto add = flat->get_input_node_shared_ptr(0);
    return ov::is_type<op::v1::Add>(add) && ov::is_type<op::v1::Multiply>(add->get_input_node_shared_ptr(1)) &&
           ov::is_type<op::v1::Reshape>(add->get_input_node_shared_ptr(0));
}

}  // namespace

TEST(TransformationTests, GatedResidualRead_FusesDecomposedNorm) {
    auto model = make_read_model({});
    run_fusion(model);

    const auto grs = gated_residuals(model);
    ASSERT_EQ(grs.size(), 1U);
    EXPECT_EQ(grs[0]->get_mode(), Gr::Mode::READ);
    EXPECT_EQ(grs[0]->get_hc_count(), C);
    EXPECT_EQ(grs[0]->get_hidden_size(), D);
    EXPECT_EQ(grs[0]->get_lowrank(), L);
    EXPECT_FALSE(grs[0]->get_produce_write_gate());
    // The gain already holds any `1 + w` fold, so the op must not add one.
    EXPECT_FALSE(grs[0]->get_norm_gain_plus_one());
    EXPECT_EQ(grs[0]->get_input_size(), 4U);
    EXPECT_EQ(count_ops(model, op::v0::MatMul::get_type_info_static()), 0U);
}

TEST(TransformationTests, GatedResidualRead_FusesFusedRms) {
    ReadOptions opt;
    opt.rms_fused = true;
    auto model = make_read_model(opt);
    run_fusion(model);

    const auto grs = gated_residuals(model);
    ASSERT_EQ(grs.size(), 1U);
    EXPECT_EQ(grs[0]->get_mode(), Gr::Mode::READ);
    EXPECT_FLOAT_EQ(grs[0]->get_epsilon(), EPS);
}

TEST(TransformationTests, GatedResidualRead_WriteGateIsFused) {
    ReadOptions opt;
    opt.with_gate = true;
    auto model = make_read_model(opt);
    run_fusion(model);

    const auto grs = gated_residuals(model);
    ASSERT_EQ(grs.size(), 1U);
    EXPECT_TRUE(grs[0]->get_produce_write_gate());
    EXPECT_EQ(grs[0]->get_input_size(), 5U);
    EXPECT_EQ(grs[0]->get_output_size(), 2U);
    EXPECT_EQ(count_ops(model, op::v0::MatMul::get_type_info_static()), 0U);
}

TEST(TransformationTests, GatedResidualRead_CompressedScalesKeepTheNGLayout) {
    ReadOptions opt;
    opt.compressed = true;
    auto model = make_read_model(opt);
    run_fusion(model);

    const auto grs = gated_residuals(model);
    ASSERT_EQ(grs.size(), 1U);
    // residual, gain, w_down, w_up, scale_down, scale_up, zp_down, zp_up (the test weights have zero points)
    ASSERT_EQ(grs[0]->get_input_size(), 8U);
    EXPECT_EQ(grs[0]->get_input_element_type(Gr::ReadInputIdx::ZP_DOWN), element::u4);
    EXPECT_EQ(grs[0]->get_input_element_type(Gr::ReadInputIdx::W_DOWN), element::u4);
    EXPECT_EQ(grs[0]->get_input_partial_shape(Gr::ReadInputIdx::SCALE_DOWN),
              (PartialShape{static_cast<int64_t>(L), 2}));
    EXPECT_EQ(grs[0]->get_input_partial_shape(Gr::ReadInputIdx::SCALE_UP), (PartialShape{static_cast<int64_t>(CD), 2}));
}

TEST(TransformationTests, GatedResidualWrite_AddMovesNextToTheMultiply) {
    auto model = make_write_model();
    run_split(model);

    EXPECT_TRUE(add_is_split(model));
    EXPECT_TRUE(gated_residuals(model).empty());
    EXPECT_EQ(model->get_results()[0]->get_output_partial_shape(0), with_last(PartialShape::dynamic(2), CD));
    const auto reference = make_write_model(PartialShape::dynamic(2), true);
    EXPECT_TRUE(add_is_split(reference));
    EXPECT_EQ(count_ops(model, op::v1::Reshape::get_type_info_static()),
              count_ops(reference, op::v1::Reshape::get_type_info_static()));
}

TEST(TransformationTests, GatedResidualRead_BranchMixWithoutScaleIsNotFused) {
    ReadOptions opt;
    opt.mix_scaled = false;
    auto model = make_read_model(opt);
    run_fusion(model);
    EXPECT_TRUE(gated_residuals(model).empty());
}

TEST(TransformationTests, GatedResidualRead_ReduceOverNonBranchAxisIsNotFused) {
    ReadOptions opt;
    opt.reduce_axis = 1;  // the token axis
    auto model = make_read_model(opt);
    run_fusion(model);
    EXPECT_TRUE(gated_residuals(model).empty());
}

TEST(TransformationTests, GatedResidual_RejectedByTheCallbackStaysDecomposed) {
    auto read = make_read_model({});
    run_fusion_with_callback(read, [](const std::shared_ptr<const Node>&) {
        return true;
    });
    EXPECT_TRUE(gated_residuals(read).empty());
    EXPECT_EQ(count_ops(read, op::v0::MatMul::get_type_info_static()), 2U);

    auto write = make_write_model();
    run_fusion_with_callback(write, [](const std::shared_ptr<const Node>&) {
        return true;
    });
    EXPECT_FALSE(add_is_split(write));
    EXPECT_TRUE(gated_residuals(write).empty());
}

TEST(TransformationTests, GatedResidualRead_RejectedWriteGateIsRetriedWithoutIt) {
    ReadOptions opt;
    opt.with_gate = true;
    auto model = make_read_model(opt);
    // Only the form with the write-gate input is rejected, e.g. because its weights are not supported.
    run_fusion_with_callback(model, [](const std::shared_ptr<const Node>& node) {
        return node->get_input_size() > 4;
    });

    const auto grs = gated_residuals(model);
    ASSERT_EQ(grs.size(), 1U);
    EXPECT_FALSE(grs[0]->get_produce_write_gate());
    EXPECT_EQ(grs[0]->get_input_size(), 4U);
    // The projection of the write gate stays in the graph, with its two outputs still produced.
    EXPECT_EQ(count_ops(model, op::v0::MatMul::get_type_info_static()), 1U);
    EXPECT_EQ(model->get_results().size(), 2U);
}

TEST(TransformationTests, GatedResidualRead_OneDynamicRowDimensionIsFused) {
    ReadOptions opt;
    opt.with_gate = true;
    opt.leading = 1;  // [rows, C*D]
    auto model = make_read_model(opt);
    run_fusion(model);

    const auto grs = gated_residuals(model);
    ASSERT_EQ(grs.size(), 1U);
    EXPECT_EQ(grs[0]->get_output_partial_shape(0), with_last(PartialShape::dynamic(1), D));
    EXPECT_EQ(grs[0]->get_output_partial_shape(1), with_last(PartialShape::dynamic(1), C));
}

TEST(TransformationTests, GatedResidualWrite_AnyRowRankIsRewritten) {
    // [tokens, C*D] of paged attention, [B, T, C*D] of the other flows, and one more.
    for (const size_t leading : {size_t{1}, size_t{2}, size_t{3}}) {
        auto model = make_write_model(PartialShape::dynamic(leading));
        run_split(model);

        EXPECT_TRUE(add_is_split(model)) << leading;
        EXPECT_EQ(model->get_results()[0]->get_output_partial_shape(0), with_last(PartialShape::dynamic(leading), CD))
            << leading;
    }
}

TEST(TransformationTests, GatedResidualWrite_StaticRowsOfTheSameSizeAreRewritten) {
    auto model = make_write_model(PartialShape{2, 3});
    run_split(model);

    EXPECT_TRUE(add_is_split(model));
    EXPECT_EQ(model->get_results()[0]->get_output_partial_shape(0), (PartialShape{2, 3, static_cast<int64_t>(CD)}));
}

TEST(TransformationTests, GatedResidualWrite_BroadcastRowsAreRewritten) {
    // The rewrite keeps the broadcasting of the rows as it is, so it applies to them too.
    auto broadcast_y = make_write_model(PartialShape::dynamic(2), PartialShape{1, 1}, false);
    run_split(broadcast_y);
    EXPECT_TRUE(add_is_split(broadcast_y));

    auto static_broadcast_y = make_write_model(PartialShape{2, 3}, PartialShape{1, 3}, false);
    run_split(static_broadcast_y);
    EXPECT_TRUE(add_is_split(static_broadcast_y));
}

TEST(TransformationTests, GatedResidualWrite_ReshapeThatChangesTheRowsIsNotRewritten) {
    // [2, 3, C, D] flattened to [6, C*D] would no longer line up with the rows of the residual.
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, static_cast<int64_t>(CD)});
    auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(D)});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(C)});
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto scaled = std::make_shared<op::v1::Multiply>(s3, y3);
    auto flat = reshape_to(scaled, {6, static_cast<int64_t>(CD)});
    auto out = std::make_shared<op::v1::Add>(residual, flat);
    auto model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                         ParameterVector{residual, y, gate});
    run_split(model);
    EXPECT_FALSE(add_is_split(model));
}

TEST(TransformationTests, GatedResidualWrite_IsFusedByDefault) {
    // Both the exporter form and the split form fuse; the split is the alternative when the fusion is disabled.
    for (const bool split : {false, true}) {
        auto model = make_write_model(PartialShape::dynamic(2), split);
        run_fusion(model);

        const auto grs = gated_residuals(model);
        ASSERT_EQ(grs.size(), 1U) << split;
        EXPECT_EQ(grs[0]->get_mode(), Gr::Mode::WRITE);
        EXPECT_EQ(grs[0]->get_input_size(), 3U);
        EXPECT_EQ(grs[0]->get_output_partial_shape(0), with_last(PartialShape::dynamic(2), CD));
        EXPECT_FALSE(add_is_split(model));
    }
}

TEST(TransformationTests, GatedResidualWrite_FusionCanBeDisabledWithoutTheSplit) {
    auto model = make_write_model();
    pass::Manager manager;
    manager.register_pass<pass::GatedResidualFusion>();
    manager.get_pass_config()->disable<pass::GatedResidualWriteFusion>();
    manager.get_pass_config()->disable<pass::GatedResidualWriteSinkReshape>();
    manager.run_passes(model);

    EXPECT_TRUE(gated_residuals(model).empty());
    EXPECT_FALSE(add_is_split(model));
}

TEST(TransformationTests, GatedResidualWrite_RowsThatMayBroadcastAreNotFused) {
    auto broadcast_y = make_write_model(PartialShape::dynamic(2), PartialShape{1, 1}, true);
    run_fusion(broadcast_y);
    EXPECT_TRUE(gated_residuals(broadcast_y).empty());

    auto static_broadcast_y = make_write_model(PartialShape{2, 3}, PartialShape{1, 3}, true);
    run_fusion(static_broadcast_y);
    EXPECT_TRUE(gated_residuals(static_broadcast_y).empty());
}
