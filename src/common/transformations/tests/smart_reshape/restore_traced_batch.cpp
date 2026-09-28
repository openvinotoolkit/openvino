// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/restore_traced_batch.hpp"

#include <gtest/gtest.h>

#include <optional>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
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

TEST_F(RestoreTracedBatchTests, ModelWithoutLeadingDimensionExpressionIsLeftAlone) {
    const auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto target =
        std::make_shared<op::v0::Concat>(OutputVector{op::v0::Constant::create(element::i64, Shape{1}, {1}),
                                                      op::v0::Constant::create(element::i64, Shape{1}, {-1})},
                                         0);
    model = std::make_shared<Model>(OutputVector{std::make_shared<op::v1::Reshape>(param, target, false)},
                                    ParameterVector{param});
}
