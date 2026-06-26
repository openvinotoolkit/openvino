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

}  // namespace

// Positive, DIRECT path. A window-reverse R8-style view: the windows tensor [num_win*B, ws, ws, C]
// is split into [B, H//ws, W//ws, ws, ws, C]. Tracing froze the leading batch into Constant(1) and
// left the channel as the trailing -1; the spatial split (H//ws, W//ws) stays dynamic. The pass must
// rewrite the shape Concat so the leading Constant becomes -1 (batch inferred) and the trailing -1
// becomes Constant(C) (channel recovered from data's static last dim), keeping the interior intact.
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_direct_positive) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        // Dynamic spatial split (H//ws, W//ws) -- non-constant shape elements.
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto shape =
            std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)},
                                         0);
        auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
        // The rewrite only flips two scalar Constants (1 -> -1, -1 -> C); the default comparator does
        // not inspect Constant values, so enable that explicitly to actually assert the rewrite.
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto shape =
            std::make_shared<v0::Concat>(OutputVector{dim_const(-1), h, w, dim_const(WS), dim_const(WS), dim_const(C)},
                                         0);
        auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});
    }
}

// Positive, WALK-BACK path -- the reason this pass is a ModelPass rather than a MatcherPass. Two chained
// window-reverse views: Reshape_1 [?,ws,ws,C] -> [?,?,?,ws,ws,C] (its channel resolves DIRECTLY from
// data's static last dim C), then a last-axis-preserving Transpose(order=[0,1,3,2,4,5]) whose output is
// fully dynamic, then Reshape_2 [B,H,W,-1] whose data last dim is now DYNAMIC. Reshape_2's channel cannot
// be read off its data; it is recovered by walking back through the Transpose to Reshape_1's recorded
// channel. BOTH reshapes must end up with leading -1 (batch inferred) and trailing Constant(C).
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_walkback_positive) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto H = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto W = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});

        auto shape1 =
            std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)},
                                         0);
        auto reshape1 = std::make_shared<v1::Reshape>(data, shape1, false);
        // Last-axis-preserving permute (order.back() == rank - 1): keeps the channel axis last.
        auto order = v0::Constant::create(element::i64, Shape{6}, {0, 1, 3, 2, 4, 5});
        auto transpose = std::make_shared<v1::Transpose>(reshape1, order);
        auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), H, W, dim_const(-1)}, 0);
        auto reshape2 = std::make_shared<v1::Reshape>(transpose, shape2, false);
        model = std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
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
        auto reshape1 = std::make_shared<v1::Reshape>(data, shape1, false);
        auto order = v0::Constant::create(element::i64, Shape{6}, {0, 1, 3, 2, 4, 5});
        auto transpose = std::make_shared<v1::Transpose>(reshape1, order);
        auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(-1), H, W, dim_const(C)}, 0);
        auto reshape2 = std::make_shared<v1::Reshape>(transpose, shape2, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape2}, ParameterVector{data, h, w, H, W});
    }
}

// End-to-end: SmartReshape runs inside Model::reshape, so re-batching the windows tensor must drive
// the pass over the whole model. We assert the rewrite happened through that full path: the shape
// Concat's leading baked-batch Constant is relaxed to -1 (batch inferred) and the trailing -1 channel
// is pinned to Constant(C). (The Reshape's inferred partial shape stays dynamic here because the
// interior spatial dims are dynamic Parameters and the leading -1 cannot be folded -- the rewrite is
// nonetheless value-correct, which the real-model A/B verifies; here we pin the graph-level effect.)
TEST(SmartReshapeTests, RestoreReshapeBakedBatch_reshape) {
    constexpr int64_t WS = 2, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    auto model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});

    // RestoreReshapeBakedBatch is called as a part of SmartReshape.
    OV_ASSERT_NO_THROW(model->reshape({{data->output(0), PartialShape{8, WS, WS, C}}}));

    auto concat = ov::as_type_ptr<v0::Concat>(reshape->input_value(1).get_node_shared_ptr());
    ASSERT_NE(concat, nullptr);
    const auto& elems = concat->input_values();
    ASSERT_EQ(elems.size(), 6u);

    auto leading = ov::as_type_ptr<v0::Constant>(elems.front().get_node_shared_ptr());
    ASSERT_NE(leading, nullptr) << "leading element must be the inferred -1 Constant";
    EXPECT_EQ(leading->cast_vector<int64_t>().at(0), -1) << "baked batch not relaxed to -1";

    auto channel = ov::as_type_ptr<v0::Constant>(elems.back().get_node_shared_ptr());
    ASSERT_NE(channel, nullptr) << "channel must be pinned to a static Constant";
    EXPECT_EQ(channel->cast_vector<int64_t>().at(0), C) << "channel not pinned to data's last dim";
}

// --------------------------------------------------------------------------------------------------
// Negative cases. The pass runs inside every Model::reshape, so it must never fire on a reshape it
// cannot prove value-preserving. Each builder produces a graph the pass must leave untouched; the
// TEST_P body sets only `model` (no model_ref), so TransformationTestsF::TearDown clones `model`
// BEFORE running the pass and compares against the clone -- an exact "did not fire" assertion.
// --------------------------------------------------------------------------------------------------

namespace {

// special_zero == true is a different reshape semantics and must never match.
std::shared_ptr<Model> build_neg_special_zero() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, /*special_zero=*/true);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});
}

// A non-constant (already symbolic) leading element means the batch already propagates -- there is
// nothing baked to relax, so the pass must not fire.
std::shared_ptr<Model> build_neg_dynamic_leading() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto batch = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic leading
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape =
        std::make_shared<v0::Concat>(OutputVector{batch, h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, batch, h, w});
}

// A fully baked shape (no dynamic interior) is an ordinary fixed reshape, not the window-reverse
// signature -- the pass must not fire.
std::shared_ptr<Model> build_neg_no_dynamic_interior() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto shape = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dim_const(2), dim_const(2), dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
}

// Spatial flatten view(1, C, -1). The shape vector is shorter than the data rank, so the trailing-block
// guard rejects it (the -1 would span more than data's last dim -- here H*W, not W).
std::shared_ptr<Model> build_neg_spatial_flatten() {
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 3, 4, 5});  // last dim 5
    auto c = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});              // dynamic C slot
    auto shape = std::make_shared<v0::Concat>(OutputVector{dim_const(1), c, dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, c});
}

// Head-merge view(1, T//2, -1): data has a STATIC last dim (D) but a DYNAMIC interior, so the trailing
// -1 spans more than D. The cheap output-channel guard is vacuous (output last dim dynamic), so the
// trailing-block guard must reject -- here at its data_dim.is_dynamic() branch (the kept interior dim is
// dynamic, so the rewrite cannot be proven to keep data's trailing block).
std::shared_ptr<Model> build_neg_head_merge() {
    constexpr int64_t D = 8;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), D});
    auto t = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic T//2
    auto shape = std::make_shared<v0::Concat>(OutputVector{dim_const(1), t, dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, t});
}

// Head-split view(1, T*2, -1): data has a STATIC last dim (D) AND a static interior dim, but the kept
// interior shape element is dynamic (T*2) and cannot be proven equal to data's static interior dim, so
// the trailing-block guard rejects -- here at its non-constant-interior branch (distinct from head_merge).
std::shared_ptr<Model> build_neg_head_split() {
    constexpr int64_t D = 8, T = 4;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), T, D});
    auto t = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic T*2
    auto shape = std::make_shared<v0::Concat>(OutputVector{dim_const(1), t, dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, t});
}

// Window-reverse with a DYNAMIC channel (no projection to a fixed width). The channel flows from the
// input and stays dynamic, so resolve_static_last_dim cannot recover a static last dim and returns
// nullopt -- the pass must not fire.
std::shared_ptr<Model> build_neg_dyn_channel() {
    constexpr int64_t WS = 8;
    auto data =
        std::make_shared<v0::Parameter>(element::f32,
                                        PartialShape{Dimension::dynamic(), WS, WS, Dimension::dynamic()});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape =
        std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});
}

// Idempotency: the already-rewritten shape (leading Constant(-1), trailing Constant(C)). Re-running the
// pass must not re-fire -- a Constant(-1) leading element fails the positive-int leading gate, so the
// rewrite is a fixed point.
std::shared_ptr<Model> build_neg_already_rewritten() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto shape =
        std::make_shared<v0::Concat>(OutputVector{dim_const(-1), h, w, dim_const(WS), dim_const(WS), dim_const(C)}, 0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});
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

INSTANTIATE_TEST_SUITE_P(
    SmartReshapeTests,
    RestoreReshapeBakedBatchNeg,
    testing::ValuesIn(std::vector<NegParams>{
        {"special_zero", build_neg_special_zero},
        {"dynamic_leading", build_neg_dynamic_leading},
        {"no_dynamic_interior", build_neg_no_dynamic_interior},
        {"spatial_flatten", build_neg_spatial_flatten},
        {"head_merge", build_neg_head_merge},
        {"head_split", build_neg_head_split},
        {"dyn_channel", build_neg_dyn_channel},
        {"already_rewritten", build_neg_already_rewritten},
    }),
    [](const testing::TestParamInfo<NegParams>& info) {
        return info.param.name;
    });
