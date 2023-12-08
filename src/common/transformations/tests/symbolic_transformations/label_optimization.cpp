// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/label_optimization.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/symbolic_transformations/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

#define TEST_SHAPE(source, ...)                                                                 \
    {                                                                                           \
        auto expected_labels = ov::TensorLabel({__VA_ARGS__});                                  \
        ov::TensorLabel labels;                                                                 \
        EXPECT_TRUE(ov::symbol::util::get_labels(source->get_output_partial_shape(0), labels)); \
        EXPECT_EQ(labels, expected_labels);                                                     \
    }

TEST(TransformationTests, ApplyTableOfEquivalence_Concat) {
    auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);
    // shape inference notes that all the non-axis dimensions are equal to each other
    auto model = make_shared<Model>(NodeVector{concat}, ParameterVector{input_2, input_1});

    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::SymbolicPropagation>();
    manager.register_pass<pass::ApplyTableOfEquivalence>();
    manager.run_passes(model);

    TEST_SHAPE(input_1, {1, 2, 3, 4});
    TEST_SHAPE(input_2, {1, 2, 3, 8});
    TEST_SHAPE(concat, {1, 2, 3, 9});
}

TEST_F(TransformationTestsF, ApplyTableOfEquivalence_Concat_Values) {
    {
        auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);

        auto shape = make_shared<v0::ShapeOf>(concat);
        auto gather = make_shared<v1::Gather>(shape,
                                              v0::Constant::create(element::i64, {1}, {-1}),
                                              v0::Constant::create(element::i64, {}, {0}));

        auto reshape = make_shared<v1::Reshape>(
            concat,
            make_shared<v0::Concat>(OutputVector{gather, v0::Constant::create(element::i64, {1}, {-1})}, 0),
            false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{input_2, input_1});

        manager.set_per_pass_validation(false);
        manager.register_pass<pass::SymbolicPropagation>();
        manager.register_pass<pass::ApplyTableOfEquivalence>();
        manager.register_pass<pass::OptimizeLabelsUsedAsValues>();
    }
    {
        auto input_1 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto input_2 = make_shared<v0::Parameter>(element::f32, PartialShape::dynamic(4));
        auto concat = make_shared<v0::Concat>(OutputVector{input_1, input_2}, -1);

        auto shape_1 = make_shared<v3::ShapeOf>(input_1);
        auto gather_1 = make_shared<v8::Gather>(shape_1,
                                                v0::Constant::create(element::i64, {1}, {3}),
                                                v0::Constant::create(element::i64, {}, {0}));

        auto shape_2 = make_shared<v3::ShapeOf>(input_2);
        auto gather_2 = make_shared<v8::Gather>(shape_2,
                                                v0::Constant::create(element::i64, {1}, {3}),
                                                v0::Constant::create(element::i64, {}, {0}));

        auto sum = make_shared<v1::Add>(gather_1, gather_2);

        auto reshape = make_shared<v1::Reshape>(
            concat,
            make_shared<v0::Concat>(OutputVector{sum, v0::Constant::create(element::i64, {1}, {-1})}, 0),
            false);
        model_ref = make_shared<Model>(NodeVector{reshape}, ParameterVector{input_2, input_1});
    }
}
