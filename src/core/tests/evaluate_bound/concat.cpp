// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gmock/gmock.h"
#include "openvino/core/dimension.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/graph_rewrite.hpp"

using namespace ov::opset9;
using namespace testing;

using ShapeVector = std::vector<ov::PartialShape>;
using LabeledShape = std::tuple<ov::PartialShape, bool>;
using LabeledShapeVector = std::vector<LabeledShape>;
using TestParams = std::tuple<int64_t, LabeledShapeVector>;

class EvaluateLabelTest : public Test {
protected:
    bool exp_evaluate_status;
    ov::TensorLabelVector out_labels;
    ov::TensorVector exp_result, inputs;
};

class ConcatEvaluateLabelTest : public EvaluateLabelTest, public WithParamInterface<TestParams> {
protected:
    void SetUp() override {
        const auto& labeled_shapes = std::get<1>(GetParam());

        exp_evaluate_status =
            std::any_of(labeled_shapes.cbegin(), labeled_shapes.cend(), [](const LabeledShape& l_shape) {
                return std::get<1>(l_shape);
            });

        for (const auto& labeled_shape : labeled_shapes) {
            ov::PartialShape shape;
            bool add_labels;
            std::tie(shape, add_labels) = labeled_shape;

            auto param = params.make<Parameter>(ov::element::from<ov::label_t>(), shape);

            if (exp_evaluate_status) {
                auto min_shape = shape.get_min_shape();
                ov::TensorLabel labels(ov::shape_size(min_shape), ov::no_label);

                if (add_labels) {
                    std::iota(labels.begin(), labels.end(), 1);
                    param->get_default_output().get_tensor().set_value_label(labels);
                }

                inputs.emplace_back(ov::element::from<ov::label_t>(), min_shape);
                std::copy_n(labels.begin(), labels.size(), inputs.back().data<ov::label_t>());
            }
        }
    }

    std::shared_ptr<Concat> concat;
    ov::pass::NodeRegistry params;
};

const auto shape1 = ov::PartialShape({3, 2, 1});
const auto shape2 = ov::PartialShape({3, 4, 1});

const auto contactable_shapes_axis_1 = Values(
    LabeledShapeVector{std::make_tuple(shape1, false)},
    LabeledShapeVector{std::make_tuple(shape2, false)},
    LabeledShapeVector{std::make_tuple(shape2, false), std::make_tuple(shape1, false)},
    LabeledShapeVector{std::make_tuple(shape1, true), std::make_tuple(shape2, false)},
    LabeledShapeVector{std::make_tuple(shape2, false), std::make_tuple(shape1, true)},
    LabeledShapeVector{std::make_tuple(shape1, true), std::make_tuple(shape2, false), std::make_tuple(shape1, false)},
    LabeledShapeVector{std::make_tuple(shape1, true), std::make_tuple(shape2, false), std::make_tuple(shape2, true)},
    LabeledShapeVector{std::make_tuple(shape1, true),
                       std::make_tuple(shape2, true),
                       std::make_tuple(shape2, true),
                       std::make_tuple(shape1, true)});

INSTANTIATE_TEST_SUITE_P(evaluate_bound_contactable_axis_1,
                         ConcatEvaluateLabelTest,
                         Combine(Values(1), contactable_shapes_axis_1),
                         PrintToStringParamName());

const auto contactable_shapes = Values(
    LabeledShapeVector{std::make_tuple(shape1, false)},
    LabeledShapeVector{std::make_tuple(shape1, false), std::make_tuple(shape1, false)},
    LabeledShapeVector{std::make_tuple(shape2, false), std::make_tuple(shape2, false), std::make_tuple(shape2, true)},
    LabeledShapeVector{std::make_tuple(shape2, true), std::make_tuple(shape2, false), std::make_tuple(shape2, true)},
    LabeledShapeVector{std::make_tuple(shape1, true), std::make_tuple(shape1, true), std::make_tuple(shape1, true)});

INSTANTIATE_TEST_SUITE_P(evaluate_bound,
                         ConcatEvaluateLabelTest,
                         Combine(testing::Range<int64_t>(-3, 3), contactable_shapes),
                         PrintToStringParamName());

/** \brief Test evaluate label for combination of different shapes and each shape may be labeled. */
TEST_P(ConcatEvaluateLabelTest, evaluate_label) {
    const auto concat = std::make_shared<Concat>(params.get(), std::get<0>(GetParam()));
    out_labels.resize(concat->get_output_size());
    exp_result.emplace_back(ov::element::from<ov::label_t>(), concat->get_output_partial_shape(0).to_shape());

    ASSERT_EQ(concat->evaluate_label(out_labels), exp_evaluate_status);

    if (exp_evaluate_status) {
        concat->evaluate(exp_result, inputs);

        ASSERT_THAT(out_labels.front(),
                    ElementsAreArray(exp_result.front().data<ov::label_t>(), exp_result.front().get_size()));
    }
}
