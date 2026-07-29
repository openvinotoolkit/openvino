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

// n dynamic i64 shape-vector inputs (each PartialShape{1}).
ParameterVector make_dyn_dims(size_t n) {
    ParameterVector dims(n);
    for (auto& d : dims)
        d = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    return dims;
}

// {data, dims...} in consumption order.
ParameterVector params_of(const std::shared_ptr<v0::Parameter>& data, const ParameterVector& dims) {
    ParameterVector params{data};
    params.insert(params.end(), dims.begin(), dims.end());
    return params;
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

// Positive. The window-reverse chain: inner view [?,ws,ws,C] -> [?,H//ws,W//ws,ws,ws,C] (channel from its
// static data last dim C), a last-axis-preserving Transpose, then the outer view [B,H,W,-1] whose data
// last dim is now dynamic. Tracing froze the leading batch into Constant(1) and left the channel as the
// trailing -1 in both views; the pass relaxes both leading constants to -1 and pins both channels to C.
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_chain_positive) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto dims = make_dyn_dims(4);  // h, w, H, W
        auto shape1 = std::make_shared<v0::Concat>(
            OutputVector{dim_const(1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(-1)},
            0);
        auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[2], dims[3], dim_const(-1)}, 0);
        auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
        model = std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
        // The rewrite only flips scalar Constants, which the default comparator ignores.
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto dims = make_dyn_dims(4);
        auto shape1 = std::make_shared<v0::Concat>(
            OutputVector{dim_const(-1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(C)},
            0);
        auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(-1), dims[2], dims[3], dim_const(C)}, 0);
        auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
        model_ref = std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
    }
}

// End-to-end: SmartReshape runs inside Model::reshape, so re-batching drives the pass over the whole
// model. We assert on the graph (both leading constants relaxed to -1, both channels pinned to C) rather
// than the output shape: the interior spatial dims are dynamic Parameters and the leading -1 cannot fold,
// so the inferred shape stays dynamic although the rewrite is value-correct.
TEST(SmartReshapeTests, RestoreReshapeBakedBatch_reshape) {
    constexpr int64_t WS = 2, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{4, WS, WS, C});
    auto dims = make_dyn_dims(4);  // h, w, H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto reshape1 = std::make_shared<v1::Reshape>(data, shape1, false);
    auto order = v0::Constant::create(element::i64, Shape{6}, {0, 1, 3, 2, 4, 5});
    auto transpose = std::make_shared<v1::Transpose>(reshape1, order);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[2], dims[3], dim_const(-1)}, 0);
    auto reshape2 = std::make_shared<v1::Reshape>(transpose, shape2, false);
    auto model = std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));

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
// Negative cases -- the pass must never fire where it cannot prove value preservation. Two families:
//   (A) a valid two-view chain in which one gate/guard is violated (the callback runs and bails);
//   (B) graphs that do not form the chain at all (the matcher is narrow).
// The TEST_P body sets only `model`, so TransformationTestsF::TearDown compares against a pre-pass clone.
// --------------------------------------------------------------------------------------------------

namespace {

// ---- family (A): valid chain with a single violated gate/guard -------------------------------------

// special_zero=true on the inner view: the structural gate rejects.
std::shared_ptr<Model> build_neg_special_zero() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto dims = make_dyn_dims(4);  // h, w, H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[2], dims[3], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, /*sz1=*/true, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Non-constant leading element: batch already propagates, nothing baked to relax -- the gate rejects.
std::shared_ptr<Model> build_neg_dynamic_leading() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto dims = make_dyn_dims(5);  // b (dynamic leading), h, w, H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dims[0], dims[1], dims[2], dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[3], dims[4], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Fully baked shape (no dynamic interior): an ordinary fixed reshape -- the gate rejects.
std::shared_ptr<Model> build_neg_no_dynamic_interior() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto dims = make_dyn_dims(2);  // H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dim_const(2), dim_const(2), dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[0], dims[1], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Transpose moves the last axis (order.back() != rank-1): is_last_axis_preserving_transpose rejects.
std::shared_ptr<Model> build_neg_transpose_moves_last_axis() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto dims = make_dyn_dims(4);  // h, w, H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[2], dims[3], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2, 3, 5, 4}, shape2, false);  // last axis moved
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Dynamic channel (no projection to a fixed width): the channel cannot be recovered from the inner
// view's data static last dim, so the callback bails at channel resolution.
std::shared_ptr<Model> build_neg_dyn_channel() {
    constexpr int64_t WS = 8;
    auto data =
        std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, Dimension::dynamic()});
    auto dims = make_dyn_dims(4);  // h, w, H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[2], dims[3], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Spatial flatten view(1, C, -1) as the inner view: the shape vector is shorter than the data rank, so
// the trailing-block guard rejects (the -1 spans more than data's last dim).
std::shared_ptr<Model> build_neg_spatial_flatten() {
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 3, 4, 5});
    auto dims = make_dyn_dims(2);  // c (channel slot), s (outer interior)
    auto shape1 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[0], dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[1], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Head-merge view(1, T//2, -1) as the inner view: static data last dim but dynamic interior, so the
// trailing -1 spans more than the last dim -- the trailing-block guard rejects at its is_dynamic branch.
std::shared_ptr<Model> build_neg_head_merge() {
    constexpr int64_t D = 8;
    auto data =
        std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), D});
    auto dims = make_dyn_dims(2);  // t (T//2), s (outer interior)
    auto shape1 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[0], dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[1], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Head-split view(1, T*2, -1) as the inner view: static data interior dim but the kept shape element is
// dynamic and cannot be proven equal to it -- the trailing-block guard rejects at its non-constant branch.
std::shared_ptr<Model> build_neg_head_split() {
    constexpr int64_t D = 8, T = 4;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), T, D});
    auto dims = make_dyn_dims(2);  // t (T*2), s (outer interior)
    auto shape1 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[0], dim_const(-1)}, 0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(1), dims[1], dim_const(-1)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 2}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// Idempotency: an already-rewritten chain (leading Constant(-1)) fails the positive-int leading gate.
std::shared_ptr<Model> build_neg_already_rewritten() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto dims = make_dyn_dims(4);  // h, w, H, W
    auto shape1 = std::make_shared<v0::Concat>(
        OutputVector{dim_const(-1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(C)},
        0);
    auto shape2 = std::make_shared<v0::Concat>(OutputVector{dim_const(-1), dims[2], dims[3], dim_const(C)}, 0);
    auto reshape2 = wire_chain(data, shape1, false, {0, 1, 3, 2, 4, 5}, shape2, false);
    return std::make_shared<Model>(OutputVector{reshape2}, params_of(data, dims));
}

// ---- family (B): no chain -> matcher is narrow -----------------------------------------------------

// A lone view with no following Transpose+second view: the pass matches only the full chain.
std::shared_ptr<Model> build_neg_lone_view() {
    constexpr int64_t WS = 8, C = 16;
    auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
    auto dims = make_dyn_dims(2);  // h, w
    auto shape = std::make_shared<v0::Concat>(
        OutputVector{dim_const(1), dims[0], dims[1], dim_const(WS), dim_const(WS), dim_const(-1)},
        0);
    auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
    return std::make_shared<Model>(OutputVector{reshape}, params_of(data, dims));
}

// An ordinary reshape with a constant target shape (not a Concat) is never matched.
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
    // The pass rewrites only scalar Constant values, which the default comparator ignores, so enable
    // CONST_VALUES to make the clone comparison a true "did not fire" assertion.
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
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
