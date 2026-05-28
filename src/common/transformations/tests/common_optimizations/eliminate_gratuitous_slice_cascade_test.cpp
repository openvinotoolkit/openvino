// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_gratuitous_slice_cascade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
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

// Parameterized case for the gratuitous-Slice-cascade matcher.
//
// All six cases share the same topology:
//     Slice(data, start_const, Select(cond, ConvertLike(ShapeOf(data), size), Add(start, size)), step)
// They differ along four axes:
//   - cond_form        : the Select's condition is either a pre-folded boolean Constant
//                        (TransposeSinking's inner CF already collapsed `Less(size, 0)`)
//                        or a live `Less(size_const, zero_scalar)`.
//   - bool_cond_vals   : values for the pre-folded boolean Constant (used iff cond_form == BoolConstant).
//   - less_rhs_val     : rhs of the live `Less` (used iff cond_form == LiveLess). The lhs is the
//                        same `size` Constant used by the Add — matching the translator output.
//   - size_is_parameter: when true, `size` is a Parameter (forces the matcher off — no folded
//                        start+size Constant is synthesisable).
// `expect_fold` controls the assertion: the matcher must fire (Select count → 0; Slice's stop is
// now a Constant carrying `expected_stop_vals`; Slice output is static) or not fire (Select count
// stays 1).
struct CascadeCase {
    enum class CondForm { BoolConstant, LiveLess };

    std::string name;
    PartialShape input_shape;

    CondForm cond_form;
    std::vector<bool> bool_cond_vals;  // iff CondForm::BoolConstant
    int32_t less_rhs_val;              // iff CondForm::LiveLess

    std::vector<int32_t> start_vals;
    std::vector<int32_t> size_vals;
    bool size_is_parameter;

    bool expect_fold;
    std::vector<int64_t> expected_stop_vals;
};

std::shared_ptr<Model> build_cascade_model(const CascadeCase& c) {
    const auto rank = c.start_vals.size();
    auto data = std::make_shared<op::v0::Parameter>(element::f32, c.input_shape);
    auto start_const = op::v0::Constant::create(element::i32, Shape{rank}, c.start_vals);

    std::shared_ptr<Node> size_input;
    if (c.size_is_parameter) {
        size_input = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{static_cast<int64_t>(rank)});
    } else {
        size_input = op::v0::Constant::create(element::i32, Shape{rank}, c.size_vals);
    }

    std::shared_ptr<Node> cond_node;
    if (c.cond_form == CascadeCase::CondForm::BoolConstant) {
        cond_node = op::v0::Constant::create(element::boolean, Shape{rank}, c.bool_cond_vals);
    } else {
        // The live `Less` mirrors translator output: lhs is the same size Constant used by Add.
        auto less_rhs = op::v0::Constant::create(element::i32, Shape{}, std::vector<int32_t>{c.less_rhs_val});
        cond_node = std::make_shared<op::v1::Less>(size_input, less_rhs);
    }

    auto shape_of = std::make_shared<op::v3::ShapeOf>(data);
    auto cvtlike = std::make_shared<op::v1::ConvertLike>(shape_of, size_input);
    auto add = std::make_shared<op::v1::Add>(start_const, size_input);
    auto select = std::make_shared<op::v1::Select>(cond_node, cvtlike, add);
    auto step = op::v0::Constant::create(element::i32, Shape{rank}, std::vector<int32_t>(rank, 1));
    auto slice = std::make_shared<op::v8::Slice>(data, start_const, select, step);

    ParameterVector params{data};
    if (c.size_is_parameter) {
        params.push_back(ov::as_type_ptr<op::v0::Parameter>(size_input));
    }
    return std::make_shared<Model>(OutputVector{slice}, params);
}

}  // namespace

class EliminateGratuitousSliceCascadeP : public testing::WithParamInterface<CascadeCase>, public testing::Test {};

TEST_P(EliminateGratuitousSliceCascadeP, MatcherBehavesAsPredicted) {
    const auto& param = GetParam();
    auto model = build_cascade_model(param);

    pass::Manager manager;
    manager.register_pass<pass::EliminateGratuitousSliceCascade>();
    manager.run_passes(model);

    if (!param.expect_fold) {
        EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 1);
        return;
    }

    EXPECT_EQ(count_ops_of_type<op::v1::Select>(model), 0);

    const auto results = model->get_results();
    ASSERT_EQ(results.size(), 1u);
    const auto slice = ov::as_type_ptr<op::v8::Slice>(results[0]->get_input_node_shared_ptr(0));
    ASSERT_NE(slice, nullptr);

    auto stop_const = ov::as_type_ptr<op::v0::Constant>(slice->get_input_node_shared_ptr(2));
    ASSERT_NE(stop_const, nullptr);
    EXPECT_EQ(stop_const->cast_vector<int64_t>(), param.expected_stop_vals);

    ASSERT_TRUE(slice->get_output_partial_shape(0).is_static());
    Shape expected_out;
    for (size_t i = 0; i < param.expected_stop_vals.size(); ++i) {
        expected_out.push_back(static_cast<size_t>(param.expected_stop_vals[i] - param.start_vals[i]));
    }
    EXPECT_EQ(slice->get_output_partial_shape(0).get_shape(), expected_out);
}

INSTANTIATE_TEST_SUITE_P(
    EliminateGratuitousSliceCascade,
    EliminateGratuitousSliceCascadeP,
    ::testing::ValuesIn(std::vector<CascadeCase>{
        // Positive: condition is all-false bool Constant, Add inputs are both Constants → fold.
        {"all_nonneg_size_folds_cascade_into_constant",
         PartialShape{1, 128, 8, 256},
         CascadeCase::CondForm::BoolConstant,
         /*bool_cond_vals=*/{false, false, false, false},
         /*less_rhs_val=*/0,
         /*start_vals=*/{0, 0, 0, 0},
         /*size_vals=*/{1, 128, 4, 128},
         /*size_is_parameter=*/false,
         /*expect_fold=*/true,
         /*expected_stop_vals=*/{1, 128, 4, 128}},
        // Positive: condition is a live `Less(size_const, 0)` with non-negative size → fold.
        {"live_less_condition_folds_cascade",
         PartialShape{1, 128, 8, 256},
         CascadeCase::CondForm::LiveLess,
         /*bool_cond_vals=*/{},
         /*less_rhs_val=*/0,
         /*start_vals=*/{0, 0, 0, 0},
         /*size_vals=*/{1, 128, 4, 128},
         /*size_is_parameter=*/false,
         /*expect_fold=*/true,
         /*expected_stop_vals=*/{1, 128, 4, 128}},
        // Negative: live `Less` rhs is non-zero → not the original `Less(size, 0)` mask → keep.
        {"less_with_nonzero_rhs_keeps_cascade",
         PartialShape{1, 128, 8, 256},
         CascadeCase::CondForm::LiveLess,
         /*bool_cond_vals=*/{},
         /*less_rhs_val=*/2,
         /*start_vals=*/{0, 0, 0, 0},
         /*size_vals=*/{1, 128, 4, 128},
         /*size_is_parameter=*/false,
         /*expect_fold=*/false,
         /*expected_stop_vals=*/{}},
        // Negative: live `Less` lhs has a negative value (legitimate `size=-1` cascade) → keep.
        {"less_with_neg_size_keeps_cascade",
         PartialShape{1, 128, 8, 256},
         CascadeCase::CondForm::LiveLess,
         /*bool_cond_vals=*/{},
         /*less_rhs_val=*/0,
         /*start_vals=*/{0, 0, 0, 0},
         /*size_vals=*/{1, 128, 4, -1},
         /*size_is_parameter=*/false,
         /*expect_fold=*/false,
         /*expected_stop_vals=*/{}},
        // Negative: pre-folded condition is mixed → not all-false → keep.
        {"mixed_condition_keeps_cascade",
         PartialShape{1, 128, 8, 256},
         CascadeCase::CondForm::BoolConstant,
         /*bool_cond_vals=*/{false, false, false, true},
         /*less_rhs_val=*/0,
         /*start_vals=*/{0, 0, 0, 0},
         /*size_vals=*/{1, 128, 4, 128},
         /*size_is_parameter=*/false,
         /*expect_fold=*/false,
         /*expected_stop_vals=*/{}},
        // Negative: Add's size operand is a Parameter, not a Constant → cannot synthesise stop → keep.
        {"non_constant_add_input_keeps_cascade",
         PartialShape{1, 128, 8, 256},
         CascadeCase::CondForm::BoolConstant,
         /*bool_cond_vals=*/{false, false, false, false},
         /*less_rhs_val=*/0,
         /*start_vals=*/{0, 0, 0, 0},
         /*size_vals=*/{1, 128, 4, 128},
         /*size_is_parameter=*/true,
         /*expect_fold=*/false,
         /*expected_stop_vals=*/{}},
    }),
    [](const testing::TestParamInfo<CascadeCase>& info) {
        return info.param.name;
    });

// A generic Select whose then-branch isn't ConvertLike(ShapeOf(...)) — the matcher's pattern is
// intentionally narrow and must not touch unrelated Select nodes. Kept as a standalone test
// because the topology is fundamentally different from the cascade cases above.
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
