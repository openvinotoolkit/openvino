// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_reshape_baked_batch.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"

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

// Negative: special_zero == true is a different reshape semantics and must never match.
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_neg_special_zero) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto shape =
            std::make_shared<v0::Concat>(OutputVector{dim_const(1), h, w, dim_const(WS), dim_const(WS), dim_const(-1)},
                                         0);
        auto reshape = std::make_shared<v1::Reshape>(data, shape, /*special_zero=*/true);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, h, w});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
    }
}

// Negative: a non-constant (already-symbolic) leading element means the batch already propagates --
// there is nothing baked to relax, so the pass must not fire.
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_neg_dynamic_leading) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto batch = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});  // dynamic leading
        auto h = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto w = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
        auto shape =
            std::make_shared<v0::Concat>(OutputVector{batch, h, w, dim_const(WS), dim_const(WS), dim_const(-1)}, 0);
        auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, batch, h, w});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
    }
}

// Negative: a fully baked shape (no dynamic interior) is an ordinary fixed reshape, not the
// window-reverse signature -- the pass must not fire.
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_neg_no_dynamic_interior) {
    constexpr int64_t WS = 8, C = 16;
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), WS, WS, C});
        auto shape = std::make_shared<v0::Concat>(
            OutputVector{dim_const(1), dim_const(2), dim_const(2), dim_const(WS), dim_const(WS), dim_const(-1)},
            0);
        auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
    }
}

// Negative: spatial flatten view(1, C, -1). The shape vector is shorter than the data rank, so the
// trailing-block guard rejects it (the -1 would span more than data's last dim -- here H*W, not W).
TEST_F(TransformationTestsF, RestoreReshapeBakedBatch_neg_spatial_flatten) {
    {
        auto data = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, 3, 4, 5});  // last dim 5
        auto c = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});              // dynamic C slot
        auto shape = std::make_shared<v0::Concat>(OutputVector{dim_const(1), c, dim_const(-1)}, 0);
        auto reshape = std::make_shared<v1::Reshape>(data, shape, false);
        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data, c});

        manager.register_pass<ov::pass::RestoreReshapeBakedBatch>();
    }
}
