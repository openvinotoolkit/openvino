// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <gtest/gtest.h>

#include <optional>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;

namespace {

Output<Node> leading_dimension_of(const std::shared_ptr<op::v0::Parameter>& parameter) {
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(parameter);
    return std::make_shared<op::v8::Gather>(shape_of,
                                            op::v0::Constant::create(element::i64, Shape{1}, {0}),
                                            op::v0::Constant::create(element::i64, Shape{}, {0}));
}

// Reproduces a traced graph: a Reshape keeps the leading dimension live, the next one pins it to a constant, and a
// third one rebuilds the batch from the input shape. Passing restored builds what the transformation is expected to
// produce, which is the only difference between the model and its reference.
std::shared_ptr<Model> make_traced_model(const std::vector<int64_t>& pinned_target,
                                         bool rebuilds_batch,
                                         bool restored = false,
                                         const std::optional<PartialShape>& pinned_source = std::nullopt) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto live = leading_dimension_of(param);
    const auto inferred = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto kept = op::v0::Constant::create(element::i64, Shape{1}, {4});

    ParameterVector parameters{param};
    std::shared_ptr<Node> data =
        std::make_shared<op::v1::Reshape>(param, std::make_shared<op::v0::Concat>(OutputVector{live, kept}, 0), false);
    if (pinned_source) {
        const auto batch_independent = std::make_shared<op::v0::Parameter>(element::f32, *pinned_source);
        parameters.push_back(batch_independent);
        data = batch_independent;
    }

    OutputVector pinned_inputs;
    for (size_t index = 0; index < pinned_target.size(); ++index) {
        if (index == 0 && restored) {
            pinned_inputs.push_back(live);
        } else {
            pinned_inputs.push_back(op::v0::Constant::create(element::i64, Shape{1}, {pinned_target[index]}));
        }
    }

    std::shared_ptr<Node> result =
        std::make_shared<op::v1::Reshape>(data, std::make_shared<op::v0::Concat>(pinned_inputs, 0), false);
    if (rebuilds_batch) {
        result = std::make_shared<op::v1::Reshape>(result,
                                                   std::make_shared<op::v0::Concat>(OutputVector{live, inferred}, 0),
                                                   false);
    }
    return std::make_shared<Model>(OutputVector{result}, parameters);
}

// Adds an intentional collapse of the batch next to the traced pin of make_traced_model. With shared_target both
// Reshapes read the same target node, as frontends produce for identical shape subgraphs.
std::shared_ptr<Model> make_model_with_collapse(bool shared_target, bool restored = false) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto live = leading_dimension_of(param);
    const auto one = op::v0::Constant::create(element::i64, Shape{1}, {1});
    const auto inferred = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto kept = op::v0::Constant::create(element::i64, Shape{1}, {4});

    const auto pinned_target = std::make_shared<op::v0::Concat>(OutputVector{one, inferred}, 0);
    const auto collapse_target =
        shared_target ? pinned_target : std::make_shared<op::v0::Concat>(OutputVector{one, inferred}, 0);
    const auto traced_target =
        restored ? std::make_shared<op::v0::Concat>(OutputVector{live, inferred}, 0) : pinned_target;

    const auto kept_batch =
        std::make_shared<op::v1::Reshape>(param, std::make_shared<op::v0::Concat>(OutputVector{live, kept}, 0), false);
    const auto pinned = std::make_shared<op::v1::Reshape>(kept_batch, traced_target, false);
    const auto rebuilt =
        std::make_shared<op::v1::Reshape>(pinned,
                                          std::make_shared<op::v0::Concat>(OutputVector{live, inferred}, 0),
                                          false);
    const auto collapsed = std::make_shared<op::v1::Reshape>(param, collapse_target, false);
    return std::make_shared<Model>(OutputVector{rebuilt, collapsed}, ParameterVector{param});
}

}  // namespace

class RestoreTracedBatchTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::RestoreTracedBatch>();
    }
};

TEST_F(RestoreTracedBatchTests, PinnedLeadingDimensionTakenFromInputShape) {
    model = make_traced_model({1, -1}, true);
    model_ref = make_traced_model({1, -1}, true, true);
}

TEST_F(RestoreTracedBatchTests, RestoredModelIsLeftAlone) {
    model = make_traced_model({1, -1}, true, true);
}

// A collapse to a leading dimension of one is indistinguishable from a traced batch by target shape alone, so only
// the absence of a rebuilt batch keeps the transformation away from it.
TEST_F(RestoreTracedBatchTests, CollapseThatKeepsBatchOutOfTheGraphIsLeftAlone) {
    model = make_traced_model({1, -1}, false);
}

TEST_F(RestoreTracedBatchTests, TargetWithoutInferredDimensionIsLeftAlone) {
    model = make_traced_model({1, 2, 2}, true);
}

TEST_F(RestoreTracedBatchTests, BatchIndependentSourceIsLeftAlone) {
    model = make_traced_model({1, -1}, true, false, PartialShape{1, 4});
}

TEST_F(RestoreTracedBatchTests, CollapseNextToTracedPinIsLeftAlone) {
    model = make_model_with_collapse(false);
    model_ref = make_model_with_collapse(false, true);
}

TEST_F(RestoreTracedBatchTests, CollapseSharingTargetWithTracedPinIsLeftAlone) {
    model = make_model_with_collapse(true);
    model_ref = make_model_with_collapse(true, true);
}

// The later Reshape takes the batch from the input, but its data already carries the batch again.
TEST_F(RestoreTracedBatchTests, CollapseBroadcastBackToBatchIsLeftAlone) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto live = leading_dimension_of(param);
    const auto inferred = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto kept = std::make_shared<op::v1::Reshape>(
        param,
        std::make_shared<op::v0::Concat>(OutputVector{live, op::v0::Constant::create(element::i64, Shape{1}, {4})}, 0),
        false);
    const auto collapsed = std::make_shared<op::v1::Reshape>(
        kept,
        std::make_shared<op::v0::Concat>(OutputVector{op::v0::Constant::create(element::i64, Shape{1}, {1}), inferred},
                                         0),
        false);
    const auto global =
        std::make_shared<op::v1::ReduceSum>(collapsed, op::v0::Constant::create(element::i64, Shape{1}, {1}), true);
    const auto per_batch = std::make_shared<op::v1::Add>(kept, global);
    const auto result =
        std::make_shared<op::v1::Reshape>(per_batch,
                                          std::make_shared<op::v0::Concat>(OutputVector{live, inferred}, 0),
                                          false);
    model = std::make_shared<Model>(OutputVector{result}, ParameterVector{param});
}

TEST_F(RestoreTracedBatchTests, ModelWithoutLeadingDimensionExpressionIsLeftAlone) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto target =
        std::make_shared<op::v0::Concat>(OutputVector{op::v0::Constant::create(element::i64, Shape{1}, {1}),
                                                      op::v0::Constant::create(element::i64, Shape{1}, {-1})},
                                         0);
    model = std::make_shared<Model>(OutputVector{std::make_shared<op::v1::Reshape>(param, target, false)},
                                    ParameterVector{param});
}

TEST(SmartReshapeTests, ReshapeRestoresTracedBatch) {
    const auto model = make_traced_model({1, -1}, true);

    model->reshape(PartialShape{2, 4});

    const auto pinned_reshape = model->get_results()[0]->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
    EXPECT_EQ(pinned_reshape->get_output_partial_shape(0), (PartialShape{2, 4}));
}
