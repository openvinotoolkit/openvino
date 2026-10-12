// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <gtest/gtest.h>

#include <set>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov;

namespace {

Output<Node> i64(const std::vector<int64_t>& values) {
    return op::v0::Constant::create(element::i64, Shape{values.size()}, values);
}

Output<Node> concat(const OutputVector& inputs) {
    return std::make_shared<op::v0::Concat>(inputs, 0);
}

struct WindowReverse {
    bool restored = false;
    bool shifted = false;
    bool rebuilds_batch = true;
    bool merge_shared = false;
    bool roll_shared = false;
    bool split_target_shared = false;
    bool converted_batch = false;
    bool rank5_merge = false;
    bool static_trailing_dim = false;
    std::vector<int64_t> permutation{0, 1, 3, 2, 4, 5};
    std::vector<int64_t> roll_axes{1, 2};
};

// SwinIR's window_reverse for a 16x16 image split into 8x8 windows, as traced with batch one.
std::shared_ptr<Model> make_window_reverse_model(const WindowReverse& options) {
    const auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 16, 16, 4});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(x, options.converted_batch ? element::i32 : element::i64);
    const Output<Node> gathered_batch = std::make_shared<op::v8::Gather>(shape_of, i64({0}), i64({0}));
    const Output<Node> batch =
        options.converted_batch ? std::make_shared<op::v0::Convert>(gathered_batch, element::i64) : gathered_batch;
    const auto one = i64({1});
    const Output<Node> pinned = options.restored ? batch : one;

    const auto windows = std::make_shared<op::v1::Reshape>(x, i64({-1, 8, 8, 4}), false);
    const int64_t trailing = options.static_trailing_dim ? 4 : -1;
    const auto make_split_target = [&](const Output<Node>& leading) {
        return concat({leading, i64({2}), i64({2}), i64({8}), i64({8}), i64({trailing})});
    };
    const auto split_target = make_split_target(pinned);
    const auto split = std::make_shared<op::v1::Reshape>(windows, split_target, false);
    const auto permute = std::make_shared<op::v1::Transpose>(split, i64(options.permutation));
    const auto merge_target = options.rank5_merge ? concat({pinned, i64({16, 1}), i64({16}), i64({-1})})
                                                  : concat({pinned, i64({16}), i64({16}), i64({trailing})});
    const auto merge = std::make_shared<op::v1::Reshape>(permute, merge_target, false);
    std::shared_ptr<Node> merged = merge;
    if (options.shifted) {
        merged = std::make_shared<op::v7::Roll>(merge,
                                                i64(std::vector<int64_t>(options.roll_axes.size(), 4)),
                                                i64(options.roll_axes));
    }
    const Output<Node> rebuilt_batch = options.rebuilds_batch ? batch : one;
    const auto rebuild_target = concat({rebuilt_batch, i64({256}), i64({4})});
    const auto rebuilt = std::make_shared<op::v1::Reshape>(merged, rebuild_target, false);

    ResultVector results{std::make_shared<op::v0::Result>(rebuilt)};
    if (options.merge_shared) {
        results.push_back(std::make_shared<op::v0::Result>(merge));
    }
    if (options.roll_shared) {
        results.push_back(std::make_shared<op::v0::Result>(merged));
    }
    if (options.split_target_shared) {
        // Another Reshape reads the same traced target.
        results.push_back(
            std::make_shared<op::v0::Result>(std::make_shared<op::v1::Reshape>(windows, split_target, false)));
    }
    return std::make_shared<Model>(results, ParameterVector{x});
}

}  // namespace

class RestoreTracedBatchTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::RestoreTracedBatch>();
    }
};

TEST_F(RestoreTracedBatchTests, WindowReverseTakesBatchFromInputShape) {
    WindowReverse traced;
    WindowReverse restored;
    restored.restored = true;
    model = make_window_reverse_model(traced);
    model_ref = make_window_reverse_model(restored);
}

TEST_F(RestoreTracedBatchTests, ShiftedWindowReverseTakesBatchFromInputShape) {
    WindowReverse traced;
    traced.shifted = true;
    WindowReverse restored = traced;
    restored.restored = true;
    model = make_window_reverse_model(traced);
    model_ref = make_window_reverse_model(restored);
}

TEST_F(RestoreTracedBatchTests, TargetsWithoutFlattenedTailAreLeftAlone) {
    WindowReverse other;
    other.static_trailing_dim = true;
    model = make_window_reverse_model(other);
}

TEST_F(RestoreTracedBatchTests, RestoredWindowReverseIsLeftAlone) {
    WindowReverse restored;
    restored.restored = true;
    model = make_window_reverse_model(restored);
}

TEST_F(RestoreTracedBatchTests, WindowReverseWithoutBatchFromInputShapeIsLeftAlone) {
    WindowReverse collapsed;
    collapsed.rebuilds_batch = false;
    model = make_window_reverse_model(collapsed);
}

TEST_F(RestoreTracedBatchTests, OtherLeadingAxisPermutationTakesBatchFromInputShape) {
    WindowReverse traced;
    traced.permutation = {0, 2, 1, 3, 4, 5};
    WindowReverse restored = traced;
    restored.restored = true;
    model = make_window_reverse_model(traced);
    model_ref = make_window_reverse_model(restored);
}

TEST_F(RestoreTracedBatchTests, PermutationMovingLeadingAxisIsLeftAlone) {
    WindowReverse other;
    other.permutation = {1, 0, 2, 3, 4, 5};
    model = make_window_reverse_model(other);
}

TEST_F(RestoreTracedBatchTests, OtherNonLeadingRollAxesTakeBatchFromInputShape) {
    WindowReverse traced;
    traced.shifted = true;
    traced.roll_axes = {2, 3};
    WindowReverse restored = traced;
    restored.restored = true;
    model = make_window_reverse_model(traced);
    model_ref = make_window_reverse_model(restored);
}

TEST_F(RestoreTracedBatchTests, RollOnLeadingAxisIsLeftAlone) {
    WindowReverse other;
    other.shifted = true;
    other.roll_axes = {0, 2};
    model = make_window_reverse_model(other);
}

TEST_F(RestoreTracedBatchTests, WindowMergeWithOtherConsumersIsLeftAlone) {
    WindowReverse shared;
    shared.merge_shared = true;
    model = make_window_reverse_model(shared);
}

TEST_F(RestoreTracedBatchTests, RollWithOtherConsumersIsLeftAlone) {
    WindowReverse shared;
    shared.shifted = true;
    shared.roll_shared = true;
    model = make_window_reverse_model(shared);
}

TEST_F(RestoreTracedBatchTests, RollOnNegativeLeadingAxisOfOtherRankIsLeftAlone) {
    WindowReverse other;
    other.rank5_merge = true;
    other.shifted = true;
    other.roll_axes = {-5};
    model = make_window_reverse_model(other);
}

TEST_F(RestoreTracedBatchTests, SharedSplitTargetIsRestoredForAllConsumers) {
    WindowReverse traced;
    traced.split_target_shared = true;
    WindowReverse restored = traced;
    restored.restored = true;
    model = make_window_reverse_model(traced);
    model_ref = make_window_reverse_model(restored);
}

TEST_F(RestoreTracedBatchTests, ConvertedBatchIsTakenFromInputShape) {
    WindowReverse traced;
    traced.converted_batch = true;
    WindowReverse restored = traced;
    restored.restored = true;
    model = make_window_reverse_model(traced);
    model_ref = make_window_reverse_model(restored);
}

TEST(RestoreTracedBatch, TargetsKeepNames) {
    const auto model = make_window_reverse_model({});
    const auto concat_names = [&] {
        std::set<std::string> names;
        for (const auto& node : model->get_ops()) {
            if (ov::is_type<op::v0::Concat>(node)) {
                names.insert(node->get_friendly_name());
            }
        }
        return names;
    };
    const auto names_before = concat_names();

    pass::Manager manager;
    manager.register_pass<pass::RestoreTracedBatch>();
    ASSERT_TRUE(manager.run_passes(model));

    EXPECT_EQ(concat_names(), names_before);
}

TEST(SmartReshapeTests, ReshapeRestoresWindowReverseBatch) {
    const auto model = make_window_reverse_model({});

    model->reshape(PartialShape{2, 16, 16, 4});

    EXPECT_EQ(model->get_results()[0]->get_input_partial_shape(0), (PartialShape{2, 256, 4}));
}
