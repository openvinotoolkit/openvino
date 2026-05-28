// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_gratuitous_slice_cascade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;

namespace {

// Builds the exact subgraph that translate_slice_op emits, but with the Less and Add
// already constant-folded (as TransposeSinking's inner CF does in the read_model path).
// The Select is fed by:
//   - cond:  Constant<bool>(`cond_values`)
//   - then:  ConvertLike(ShapeOf(Parameter<input_shape>), size_constant)
//   - else:  Add(start_constant, size_constant)   (kept as Add — matcher must compute start+size)
// The Select feeds Slice's stop input.
std::shared_ptr<Model> build_slice_cascade_model(const PartialShape& input_shape,
                                                 const std::vector<int32_t>& start_vals,
                                                 const std::vector<int32_t>& size_vals,
                                                 const std::vector<bool>& cond_vals,
                                                 bool add_inputs_constant = true) {
    const auto rank = start_vals.size();
    auto data = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    auto start_const = op::v0::Constant::create(element::i32, Shape{rank}, start_vals);
    std::shared_ptr<Node> size_input;
    if (add_inputs_constant) {
        size_input = op::v0::Constant::create(element::i32, Shape{rank}, size_vals);
    } else {
        size_input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{static_cast<int64_t>(rank)});
    }
    auto cond_const = op::v0::Constant::create(element::boolean, Shape{rank}, cond_vals);

    auto shape_of = std::make_shared<op::v3::ShapeOf>(data);
    auto then_branch = std::make_shared<op::v1::ConvertLike>(shape_of, size_input);
    auto else_branch = std::make_shared<op::v1::Add>(start_const, size_input);

    auto stop = std::make_shared<op::v1::Select>(cond_const, then_branch, else_branch);
    auto const_one = op::v0::Constant::create(element::i32, Shape{rank}, std::vector<int32_t>(rank, 1));
    auto slice = std::make_shared<op::v8::Slice>(data, start_const, stop, const_one);

    ParameterVector params{data};
    if (!add_inputs_constant) {
        params.push_back(ov::as_type_ptr<op::v0::Parameter>(size_input));
    }
    return std::make_shared<Model>(OutputVector{slice}, params);
}

}  // namespace

// Positive case: condition is all-false, Add inputs are both Constants, size is non-negative.
// Matcher must fire: Select is removed; Slice's stop input is fed by a Constant; Slice
// output shape resolves to static.
TEST(EliminateGratuitousSliceCascade, all_nonneg_size_folds_cascade_into_constant) {
    auto model = build_slice_cascade_model(/*input_shape=*/PartialShape{1, 128, 8, 256},
                                           /*start=*/{0, 0, 0, 0},
                                           /*size=*/{1, 128, 4, 128},
                                           /*cond=*/{false, false, false, false});

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 0);

    const auto results = model->get_results();
    ASSERT_EQ(results.size(), 1u);
    const auto slice = ov::as_type_ptr<op::v8::Slice>(results[0]->get_input_node_shared_ptr(0));
    ASSERT_NE(slice, nullptr);

    // Slice's stop input must now be a Constant produced by the matcher.
    auto stop_producer = slice->get_input_node_shared_ptr(2);
    auto stop_const = ov::as_type_ptr<op::v0::Constant>(stop_producer);
    ASSERT_NE(stop_const, nullptr);
    EXPECT_EQ(stop_const->cast_vector<int64_t>(), (std::vector<int64_t>{1, 128, 4, 128}));

    // Slice output shape must be static after the pass + validate.
    EXPECT_TRUE(slice->get_output_partial_shape(0).is_static());
    EXPECT_EQ(slice->get_output_partial_shape(0).get_shape(), Shape({1, 128, 4, 128}));
}

// Same as above but condition is a live Less(size_const, zero_const) — the form emitted
// straight by the translator, before any ConstantFolding has run. Matcher must still fire.
TEST(EliminateGratuitousSliceCascade, live_less_condition_folds_cascade) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 128, 8, 256});
    auto start_const = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{0, 0, 0, 0});
    auto size_const = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{1, 128, 4, 128});
    auto zero_scalar = op::v0::Constant::create(element::i32, Shape{}, std::vector<int32_t>{0});
    auto less = std::make_shared<op::v1::Less>(size_const, zero_scalar);
    auto shape_of = std::make_shared<op::v3::ShapeOf>(data);
    auto cvtlike = std::make_shared<op::v1::ConvertLike>(shape_of, size_const);
    auto add = std::make_shared<op::v1::Add>(start_const, size_const);
    auto select = std::make_shared<op::v1::Select>(less, cvtlike, add);
    auto step = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{1, 1, 1, 1});
    auto slice = std::make_shared<op::v8::Slice>(data, start_const, select, step);
    auto model = std::make_shared<Model>(OutputVector{slice}, ParameterVector{data});

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 0);
    auto stop_const = ov::as_type_ptr<op::v0::Constant>(slice->get_input_node_shared_ptr(2));
    ASSERT_NE(stop_const, nullptr);
    EXPECT_EQ(stop_const->cast_vector<int64_t>(), (std::vector<int64_t>{1, 128, 4, 128}));
    EXPECT_TRUE(slice->get_output_partial_shape(0).is_static());
    EXPECT_EQ(slice->get_output_partial_shape(0).get_shape(), Shape({1, 128, 4, 128}));
}

// Negative case: a Less with a non-negative-size lhs but a non-zero rhs would not produce
// an all-false mask — matcher must not fire.
TEST(EliminateGratuitousSliceCascade, less_with_nonzero_rhs_keeps_cascade) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 128, 8, 256});
    auto start_const = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{0, 0, 0, 0});
    auto size_const = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{1, 128, 4, 128});
    auto wrong_rhs = op::v0::Constant::create(element::i32, Shape{}, std::vector<int32_t>{2});
    auto less = std::make_shared<op::v1::Less>(size_const, wrong_rhs);
    auto shape_of = std::make_shared<op::v3::ShapeOf>(data);
    auto cvtlike = std::make_shared<op::v1::ConvertLike>(shape_of, size_const);
    auto add = std::make_shared<op::v1::Add>(start_const, size_const);
    auto select = std::make_shared<op::v1::Select>(less, cvtlike, add);
    auto step = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{1, 1, 1, 1});
    auto slice = std::make_shared<op::v8::Slice>(data, start_const, select, step);
    auto model = std::make_shared<Model>(OutputVector{slice}, ParameterVector{data});

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 1);
}

// Negative case: a Less whose lhs has a negative value (the legitimate `size=-1`
// cascade path) must not be folded — matcher must not fire.
TEST(EliminateGratuitousSliceCascade, less_with_neg_size_keeps_cascade) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 128, 8, 256});
    auto start_const = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{0, 0, 0, 0});
    auto size_const = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{1, 128, 4, -1});
    auto zero_scalar = op::v0::Constant::create(element::i32, Shape{}, std::vector<int32_t>{0});
    auto less = std::make_shared<op::v1::Less>(size_const, zero_scalar);
    auto shape_of = std::make_shared<op::v3::ShapeOf>(data);
    auto cvtlike = std::make_shared<op::v1::ConvertLike>(shape_of, size_const);
    auto add = std::make_shared<op::v1::Add>(start_const, size_const);
    auto select = std::make_shared<op::v1::Select>(less, cvtlike, add);
    auto step = op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{1, 1, 1, 1});
    auto slice = std::make_shared<op::v8::Slice>(data, start_const, select, step);
    auto model = std::make_shared<Model>(OutputVector{slice}, ParameterVector{data});

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 1);
}

// Negative case A: condition is mixed (some true, some false). The original Less mask
// would not be all-false, so the rewrite is algebraically invalid — matcher must NOT fire.
TEST(EliminateGratuitousSliceCascade, mixed_condition_keeps_cascade) {
    auto model = build_slice_cascade_model(/*input_shape=*/PartialShape{1, 128, 8, 256},
                                           /*start=*/{0, 0, 0, 0},
                                           /*size=*/{1, 128, 4, 128},
                                           /*cond=*/{false, false, false, true});

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 1);
}

// Negative case B: Add's inputs are not both Constants (size is a Parameter here). The
// matcher cannot synthesise a literal stop Constant, so it must NOT fire — Select stays.
TEST(EliminateGratuitousSliceCascade, non_constant_add_input_keeps_cascade) {
    auto model = build_slice_cascade_model(/*input_shape=*/PartialShape{1, 128, 8, 256},
                                           /*start=*/{0, 0, 0, 0},
                                           /*size=*/{1, 128, 4, 128},
                                           /*cond=*/{false, false, false, false},
                                           /*add_inputs_constant=*/false);

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 1);
}

// Negative case C: A generic Select whose then-branch isn't ConvertLike(ShapeOf(...)) — the
// matcher's pattern is intentionally narrow and must not touch unrelated Select nodes.
TEST(EliminateGratuitousSliceCascade, unrelated_select_pattern_is_preserved) {
    auto x = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    auto y = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    auto cond = op::v0::Constant::create(element::boolean, Shape{4}, std::vector<bool>{false, false, false, false});
    auto select = std::make_shared<op::v1::Select>(cond, x, y);
    auto model = std::make_shared<Model>(OutputVector{select}, ParameterVector{x, y});

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 1);
}
