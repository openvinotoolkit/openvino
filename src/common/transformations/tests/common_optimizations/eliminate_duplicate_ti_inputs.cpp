// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_duplicate_ti_inputs.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace testing;
using namespace std;
using namespace ov::opset10;

TEST(TransformationTests, EliminateDuplicateTIInputs) {
    shared_ptr<ov::Model> model;

    auto invariant = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto split = make_shared<Parameter>(ov::element::f32, ov::Shape{5});
    auto merged = make_shared<Parameter>(ov::element::f32, ov::Shape{1});

    auto ti = make_shared<TensorIterator>();

    auto inv_A = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto inv_B = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto inv_C = make_shared<Parameter>(ov::element::f32, ov::Shape{1});

    auto split_A = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto split_B = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto split_C = make_shared<Parameter>(ov::element::f32, ov::Shape{1});

    auto merged_A = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto merged_B = make_shared<Parameter>(ov::element::f32, ov::Shape{1});
    auto merged_C = make_shared<Parameter>(ov::element::f32, ov::Shape{1});

    auto relu = make_shared<Relu>(merged_A);

    auto res_A = make_shared<Result>(relu);
    auto concat = make_shared<Concat>(
        ov::OutputVector{inv_A, inv_B, inv_C, split_A, split_B, split_C, merged_A, merged_B, merged_C},
        0);

    auto ti_res = make_shared<Result>(concat);
    auto body = make_shared<ov::Model>(
        ov::ResultVector{ti_res, res_A},
        ov::ParameterVector{inv_A, inv_B, inv_C, split_A, split_B, split_C, merged_A, merged_B, merged_C});

    ti->set_body(body);
    ti->set_invariant_input(inv_A, invariant);
    ti->set_invariant_input(inv_B, invariant);
    ti->set_invariant_input(inv_C, invariant);

    ti->set_sliced_input(split_A, split, 0, 1, 1, -1, 0);
    ti->set_sliced_input(split_B, split, 0, 1, 1, -1, 0);
    ti->set_sliced_input(split_C, split, 0, 1, 1, -1, 0);

    ti->set_merged_input(merged_A, merged, res_A);
    ti->set_merged_input(merged_B, merged, res_A);
    ti->set_merged_input(merged_C, merged, res_A);

    ti->get_iter_value(ti_res);
    auto res = make_shared<Result>(ti->output(0));
    model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{invariant, split, merged});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::EliminateDuplicateTIInputs>();
    manager.run_passes(model);

    shared_ptr<TensorIterator> ti_after_transformation;
    for (const auto& op : model->get_ordered_ops()) {
        if ((ti_after_transformation = ov::as_type_ptr<TensorIterator>(op))) {
            break;
        }
    }

    EXPECT_NE(ti_after_transformation, nullptr);
    EXPECT_EQ(ti_after_transformation->get_body()->get_parameters().size(), 3);
    EXPECT_EQ(ti_after_transformation->inputs().size(), 3);
    EXPECT_EQ(ti_after_transformation->get_input_descriptions().size(), 3);
}
