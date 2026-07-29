// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_reshape_baked_batch.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov;
using namespace ov::op;

namespace {

// Build a 1-element i64 Constant (a shape-vector element).
std::shared_ptr<v0::Constant> dim_const(int64_t value) {
    return v0::Constant::create(element::i64, Shape{1}, {value});
}

// Wire the window-reverse chain: inner Reshape -> Transpose(order) -> outer Reshape, and return the outer
// Reshape (the chain root the pass matches on).
std::shared_ptr<v1::Reshape> wire_chain(const Output<Node>& data,
                                        const std::shared_ptr<Node>& shape1,
                                        bool sz1,
                                        const std::vector<int64_t>& order,
                                        const std::shared_ptr<Node>& shape2,
                                        bool sz2) {
    auto reshape1 = std::make_shared<v1::Reshape>(data, shape1, sz1);
    auto ord = v0::Constant::create(element::i64, Shape{order.size()}, order);
    auto transpose = std::make_shared<v1::Transpose>(reshape1, ord);
    return std::make_shared<v1::Reshape>(transpose, shape2, sz2);
}

}  // namespace

// Positive. The exact window-reverse chain: an inner view [?,ws,ws,C] -> [?,H//ws,W//ws,ws,ws,C] whose
// channel resolves DIRECTLY from its data's static last dim C, a last-axis-preserving
// Transpose(order=[0,1,3,2,4,5]) whose output is fully dynamic, then the outer view [B,H,W,-1] whose data
// last dim is now DYNAMIC (it comes through the permute). Tracing froze the leading batch into Constant(1)
// and left the channel as the trailing -1 in BOTH views. The pass matches the chain on the outer reshape
// and rewrites BOTH shape Concats: leading Constant(1) -> Constant(-1) (batch inferred) and trailing -1 ->
// Constant(C) (channel taken from the inner view's static data last dim), keeping the interior intact.
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_chain_positive) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        // Dynamic spatial split (H//ws, W//ws) and (H, W) -- non-constant shape elements.
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});

        auto shape1 =
            std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)},
                                         0);
        auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
        auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
        model = std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
        // The rewrite only flips scalar Constants (1 -> -1, -1 -> C); the default comparator does not
        // inspect Constant values, so enable that explicitly to actually assert the rewrite.
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});

        auto shape1 =
            std::make_shared<v0::Concat>(OutputVector{dim_const(-1), h, w, dim_const(WS), dim_const(WS), dim_const(C)},
                                         0);
        auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(-1), H, W, dim_const(C)}, 0);
        auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});
    }
}

// End-to-end: SmartReshape runs inside Model::reshape, so re-batching the windows tensor must drive the
// pass over the whole model. We assert the rewrite happened through that full path: BOTH shape Concats
// (inner and outer view) have their leading baked-batch Constant relaxed to -1 (batch inferred) and their
// trailing -1 channel pinned to Constant(C). (The Reshape's inferred partial shape stays dynamic here
// because the interior spatial dims are dynamic Parameters and the leading -1 cannot be folded -- the
// rewrite is nonetheless value-correct, which the real-model A/B verifies; here we pin the graph effect.)
TEST(SmartReshapeTests, RestoreReshapeBakedBatch_reshape) {
    constexpr int64_t WS = 2, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});

    auto shape1 =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto reshape1 = std::make_shared<v1::Reshape>(data, shape1, false);
    auto order = v0::Constant::create(element::i64, Shape{6}, {0, 1, 3, 2, 4, 5});
    auto transpose = std::make_shared<v1::Transpose>(reshape1, order);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
    auto reshape2 = std::make_shared<v1::Reshape>(transpose, shape2, false);
    auto model = std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});

    // RestoreReshapeBakedBatch is called as a part of SmartReshape.
    OV_ASSERT_NO_THROW(model->reshape({{data->output(0), PartialShape{8, WS, WS, C}}}));

    for (const auto& reshape : {reshape1, reshape2}) {
        auto concat = ov::as_type_ptr<v0::Concat>(reshape->input_value(1).get_node_shared_ptr());
        ASSERT_NE(concat, nullptr);
        const auto& elems = concat->input_values();
        ASSERT_GE(elems.size(), 3u);

        auto leading = ov::as_type_ptr<v0::Constant>(elems.front().get_node_shared_ptr());
        ASSERT_NE(leading, nullptr) << "leading element must be the inferred -1 Constant";
        EXPECT_EQ(leading->cast_vector<int64_t>().at(0), -1) << "baked batch not relaxed to -1";

        auto channel = ov::as_type_ptr<v0::Constant>(elems.back().get_node_shared_ptr());
        ASSERT_NE(channel, nullptr) << "channel must be pinned to a static Constant";
        EXPECT_EQ(channel->cast_vector<int64_t>().at(0), C) << "channel not pinned to data's last dim";
    }
}

// --------------------------------------------------------------------------------------------------
// Negative cases. The pass runs inside every Model::reshape, so it must never fire on a graph it cannot
// prove value-preserving. Each builder produces a graph the pass must leave untouched; the TEST_P body
// sets only `model` (no model_ref), so TransformationTestsF::TearDown clones `model` BEFORE running the
// pass and compares against the clone -- an exact "did not fire" assertion.
//
// Two families:
//   (A) A structurally valid two-view chain in which ONE gate/guard is violated -- the chain matches the
//       pattern, the callback runs, and it bails at the named check. These prove the gate/guard logic.
//   (B) Graphs that do not form the chain at all -- these prove the matcher is narrow (a lone view or an
//       ordinary reshape is never touched).
// --------------------------------------------------------------------------------------------------

namespace {

// ---- family (A): valid chain with a single violated gate/guard -------------------------------------

// special_zero == true is a different reshape semantics; the inner view's structural gate must reject it.
std::shared_ptr<Model> build_neg_special_zero() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape1 =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, /*sz1=*/true, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});
}

// A non-constant (already symbolic) leading element means the batch already propagates -- there is
// nothing baked to relax, so the inner view's structural gate must reject.
std::shared_ptr<Model> build_neg_dynamic_leading() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto b = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic leading
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape1 =
        std::make_shared<v0::Concat>(OutputVector{b, h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, b, h, w, H, W});
}

// A fully baked shape (no dynamic interior) is an ordinary fixed reshape, not the window-reverse
// signature -- the inner view's structural gate must reject.
std::shared_ptr<Model> build_neg_no_dynamic_interior() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dim_const(2), dim_const(2), dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, H, W});
}

// A middle Transpose that MOVES the last axis (order.back() != rank-1) breaks the guarantee that the two
// views share a channel; is_last_axis_preserving_transpose must reject.
std::shared_ptr<Model> build_neg_transpose_moves_last_axis() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape1 =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2, 3, 5, 4}, shape2, false);  // last axis moved
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});
}

// Window-reverse with a DYNAMIC channel (no projection to a fixed width). Structural gates pass, but the
// channel cannot be recovered from the inner view's data static last dim (it is dynamic), so the callback
// bails at the channel-resolution step.
std::shared_ptr<Model> build_neg_dyn_channel() {
    constexpr int64_t WS = 8;
    auto data =
        std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, Dimension::dynamic()});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape1 =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});
}

// Spatial flatten view(1, C, -1) as the inner (direct-path) view: the shape vector is shorter than the
// data rank, so the trailing-block guard rejects it (the -1 would span more than data's last dim -- here
// H*W, not W). A valid outer view completes the chain so the callback reaches the inner guard.
std::shared_ptr<Model> build_neg_spatial_flatten() {
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 3, 4, 5});  // last dim 5
    auto c = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});              // dynamic C slot
    auto s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});              // outer interior
    auto shape1 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), c, dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), s, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, c, s});
}

// Head-merge view(1, T//2, -1) as the inner view: data has a STATIC last dim (D) but a DYNAMIC interior,
// so the trailing -1 spans more than D. Guard 1 is vacuous (output last dim dynamic); the trailing-block
// guard rejects at its data_dim.is_dynamic() branch (the kept interior dim is dynamic).
std::shared_ptr<Model> build_neg_head_merge() {
    constexpr int64_t D = 8;
    auto data =
        std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), D});
    auto t = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic T//2
    auto s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // outer interior
    auto shape1 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), t, dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), s, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, t, s});
}

// Head-split view(1, T*2, -1) as the inner view: data has a STATIC last dim (D) AND a static interior dim,
// but the kept interior shape element is dynamic (T*2) and cannot be proven equal to data's static
// interior dim, so the trailing-block guard rejects at its non-constant-interior branch (distinct from
// head_merge).
std::shared_ptr<Model> build_neg_head_split() {
    constexpr int64_t D = 8, T = 4;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), T, D});
    auto t = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic T*2
    auto s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // outer interior
    auto shape1 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), t, dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), s, dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, t, s});
}

// Idempotency: the already-rewritten chain (leading Constant(-1), trailing Constant(C) in both views).
// Re-running the pass must not re-fire -- a Constant(-1) leading element fails the positive-int leading
// gate, so the rewrite is a fixed point.
std::shared_ptr<Model> build_neg_already_rewritten() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape1 =
        std::make_shared<v0::Concat>(OutputVector{dim_const(-1), h, w, dim_const(WS), dim_const(WS), dim_const(C)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(-1), H, W, dim_const(C)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});
}

// ---- family (B): no chain -> matcher is narrow -----------------------------------------------------

// A single window-reverse view with NO following Transpose+second view. The pass matches only the full
// two-view chain, so a lone view is deliberately left untouched (the ticket model always emits the pair).
std::shared_ptr<Model> build_neg_lone_view() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});
}

// An ordinary reshape with a constant target shape (not a Concat) -- the common case -- is never matched.
std::shared_ptr<Model> build_neg_ordinary_reshape() {
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3, 4, 5});
    auto shape = v0::Constant::create(element::i64, Shape{2}, {-1, 60});
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
}

struct NegParams {
    std::string name;
    std::function<std::shared_ptr<Model>()> build;
};

}  // namespace

class RestoreReshapeBakedBatchNeg : public testing::WithParamInterface<NegParams>, public TransformationTestsF {};

TEST_P(RestoreReshapeBakedBatchNeg, PassDoesNotFire) {
    const auto& p = GetParam();
    model = p.build();
    manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
    // model_ref left null on purpose: TearDown clones `model` before running the pass and compares,
    // so the test asserts the pass made no change.
}

INSTANTIATE_TEST_SUITE_P(SmartReshapeTests,
                         RestoreReshapeBakedBatchNeg,
                         testing::ValuesIn(std::vector<NegParams>{
                             {"special_zero", build_neg_special_zero},
                             {"dynamic_leading", build_neg_dynamic_leading},
                             {"no_dynamic_interior", build_neg_no_dynamic_interior},
                             {"transpose_moves_last_axis", build_neg_transpose_moves_last_axis},
                             {"dyn_channel", build_neg_dyn_channel},
                             {"spatial_flatten", build_neg_spatial_flatten},
                             {"head_merge", build_neg_head_merge},
                             {"head_split", build_neg_head_split},
                             {"already_rewritten", build_neg_already_rewritten},
                             {"lone_view", build_neg_lone_view},
                             {"ordinary_reshape", build_neg_ordinary_reshape},
                         }),
                         [](const testing::TestParamInfo<NegParams>& info) {
                             return info.param.name;
                         });
