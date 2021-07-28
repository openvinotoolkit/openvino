// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/builder/autobroadcast.hpp>
#include <transformations/common_optimizations/algebraic_simplification.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace ngraph;
using namespace std;

TEST(algebraic_simplification, add_negative_tests) {
    Shape shape{};
    auto type = element::f32;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    auto abs_a = make_shared<op::Abs>(a);
    auto iconst2 = ngraph::make_constant_from_string("2", type, shape);
    auto add_a_0 = std::make_shared<ngraph::op::v1::Add>(a, iconst2);
    auto add_a_0_0 = std::make_shared<ngraph::op::v1::Add>(add_a_0, iconst2);
    auto add_b_0 = std::make_shared<ngraph::op::v1::Add>(b, abs_a);
    auto add_b_0_0 = std::make_shared<ngraph::op::v1::Add>(add_b_0, abs_a);

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    auto expected = ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++) {
        ASSERT_EQ(expected.at(i), results.at(i)->input_value(0).get_node_shared_ptr());
    }
}

TEST(algebraic_simplification, multiply_negative_tests) {
    Shape shape{};
    auto type = element::f32;
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    auto abs_a = make_shared<op::Abs>(a);
    auto iconst2 = ngraph::make_constant_from_string("2", type, shape);
    auto add_a_0 = make_shared<op::v1::Multiply>(a, iconst2);
    auto add_a_0_0 = make_shared<op::v1::Multiply>(add_a_0, iconst2);
    auto add_b_0 = make_shared<op::v1::Multiply>(b, abs_a);
    auto add_b_0_0 = make_shared<op::v1::Multiply>(add_b_0, abs_a);

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    auto expected = ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++) {
        ASSERT_EQ(expected.at(i), results.at(i)->input_value(0).get_node_shared_ptr());
    }
}

TEST(algebraic_simplification, multiply_prod_negative) {
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{2}, {1.0, 1.0});
    auto broadcast = builder::opset1::make_broadcast(fconst1, Shape{2, 5}, AxisSet{1});
    auto axes = op::Constant::create(element::i64, {2}, {0, 1});
    auto prod_fconst1 = std::make_shared<op::v1::ReduceProd>(broadcast, axes);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{prod_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto f_prod = f->get_results().at(0)->input_value(0).get_node_shared_ptr();
    ASSERT_EQ(f_prod, prod_fconst1);
}

TEST(algebraic_simplification, multiply_sum_negative) {
    auto fconst1 = ngraph::op::Constant::create(element::f64, Shape{2}, {1.0, 1.0});
    auto broadcast = builder::opset1::make_broadcast(fconst1, Shape{2, 5}, AxisSet{1});
    auto axes = op::Constant::create(element::i64, {2}, {0, 1});
    auto sum_fconst1 = std::make_shared<op::v1::ReduceSum>(broadcast, axes);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{sum_fconst1}, ParameterVector{});
    pass_manager.run_passes(f);
    auto f_sum = f->get_results().at(0)->input_value(0).get_node_shared_ptr();
    ASSERT_EQ(f_sum, sum_fconst1);
}

TEST(algebraic_simplification, concat_parameter_slices_reversed) {
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto strides = op::Constant::create(element::i64, {2}, {1, 1});
    std::vector<int64_t> mask(2, 0);
    auto slice1 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {0, 0}),
            op::Constant::create(element::i64, {2}, {32, 100}),
            strides, mask, mask);
    auto slice2 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {32, 0}),
            op::Constant::create(element::i64, {2}, {64, 100}),
            strides, mask, mask);
    auto slice3 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {64, 0}),
            op::Constant::create(element::i64, {2}, {96, 100}),
            strides, mask, mask);

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice3, slice2, slice1}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->input_value(0).get_node_shared_ptr(), concat);
}

TEST(algebraic_simplification, concat_parameter_slices_element_count) {
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    // slicing 30 elements out of 96; should trigger a check that some elements are missing
    auto strides = op::Constant::create(element::i64, {2}, {1, 1});
    std::vector<int64_t> mask(2, 0);
    auto slice1 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {0, 0}),
            op::Constant::create(element::i64, {2}, {10, 100}),
            strides, mask, mask);
    auto slice2 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {10, 0}),
            op::Constant::create(element::i64, {2}, {20, 100}),
            strides, mask, mask);
    auto slice3 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {20, 0}),
            op::Constant::create(element::i64, {2}, {30, 100}),
            strides, mask, mask);

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->input_value(0).get_node_shared_ptr(), concat);
}

TEST(algebraic_simplification, concat_parameter_non_uniform_slices) {
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto strides = op::Constant::create(element::i64, {2}, {1, 1});
    std::vector<int64_t> mask(2, 0);
    auto slice1 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {0, 0}),
            op::Constant::create(element::i64, {2}, {38, 100}),
            strides, mask, mask);
    auto slice2 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {38, 0}),
            op::Constant::create(element::i64, {2}, {64, 100}),
            strides, mask, mask);
    auto slice3 = make_shared<op::v1::StridedSlice>(a,
            op::Constant::create(element::i64, {2}, {64, 0}),
            op::Constant::create(element::i64, {2}, {96, 100}),
            strides, mask, mask);

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->input_value(0).get_node_shared_ptr(), concat);
}

TEST(algebraic_simplification, concat_different_inputs) {
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto goe1 = -a;
    auto goe2 = -a;
    auto strides = op::Constant::create(element::i64, {2}, {1, 1});
    std::vector<int64_t> mask(2, 0);
    auto slice1 = make_shared<op::v1::StridedSlice>(goe1,
            op::Constant::create(element::i64, {2}, {0, 0}),
            op::Constant::create(element::i64, {2}, {32, 100}),
            strides, mask, mask);
    auto slice2 = make_shared<op::v1::StridedSlice>(goe2,
            op::Constant::create(element::i64, {2}, {32, 0}),
            op::Constant::create(element::i64, {2}, {64, 100}),
            strides, mask, mask);
    auto slice3 = make_shared<op::v1::StridedSlice>(goe1,
            op::Constant::create(element::i64, {2}, {64, 0}),
            op::Constant::create(element::i64, {2}, {96, 100}),
            strides, mask, mask);

    size_t concat_axis = 0;
    auto concat = make_shared<op::Concat>(NodeVector{slice1, slice2, slice3}, concat_axis);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{concat}, ParameterVector{a});
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->input_value(0).get_node_shared_ptr(), concat);
}

TEST(algebraic_simplification, log_no_exp) {
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto b = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto abs_a = make_shared<op::Abs>(a);
    auto div = std::make_shared<op::v1::Divide>(abs_a, b);
    auto log_div = make_shared<op::Log>(div);

    auto neg_inner = make_shared<op::Negative>(log_div);
    auto neg2 = make_shared<op::Negative>(neg_inner);
    auto neg3 = make_shared<op::Negative>(neg2);
    auto neg4 = make_shared<op::Negative>(neg3);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{neg4}, ParameterVector{a, b});
    pass_manager.run_passes(f);
    ASSERT_EQ(neg_inner->input_value(0).get_node_shared_ptr(), log_div);
}

TEST(algebraic_simplification, log_no_divide) {
    auto a = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto b = make_shared<op::Parameter>(element::f32, Shape{96, 100});
    auto exp_a = make_shared<op::Exp>(a);
    auto mul = make_shared<op::v1::Multiply>(exp_a, b);
    auto log_mul = make_shared<op::Log>(mul);

    auto neg_inner = make_shared<op::Negative>(log_mul);
    auto neg2 = make_shared<op::Negative>(neg_inner);
    auto neg3 = make_shared<op::Negative>(neg2);
    auto neg4 = make_shared<op::Negative>(neg3);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AlgebraicSimplification>();

    auto f = std::make_shared<Function>(ngraph::NodeVector{neg4}, ParameterVector{a, b});
    pass_manager.run_passes(f);
    ASSERT_EQ(neg_inner->input_value(0).get_node_shared_ptr(), log_mul);
}

TEST(algebraic_simplification, pass_property) {
    auto pass = std::make_shared<ngraph::pass::AlgebraicSimplification>();

    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(algebraic_simplification, replace_transpose_with_reshape) {
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& perm_val,
                            bool i32,
                            bool multiout,
                            size_t num) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> perm;
        if (i32) {
            std::vector<int32_t> perm_val_i32(perm_val.begin(), perm_val.end());
            perm =
                op::Constant::create<int32_t>(element::i32, Shape{perm_val.size()}, perm_val_i32);
        } else {
            perm = op::Constant::create<int64_t>(element::i64, Shape{perm_val.size()}, perm_val);
        }
        auto param = make_shared<op::Parameter>(element::f32, shape);
        shared_ptr<Node> A1;
        if (multiout) {
            shared_ptr<Node> k;
            auto last_dim = shape.rank().get_length() - 1;
            if (shape[last_dim].is_dynamic()) {
                k = make_shared<op::v1::Gather>(make_shared<op::ShapeOf>(param),
                                                op::Constant::create(element::i64, {}, {last_dim}),
                                                op::Constant::create(element::i64, {}, {0}));
            } else {
                k = make_shared<op::Constant>(element::i64, Shape{}, std::vector<int64_t>{shape[last_dim].get_length()});
            }
            A1 = make_shared<op::v1::TopK>(param, k, last_dim,
                                           op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::NONE);
        } else {
            A1 = make_shared<op::v0::Abs>(param);
        }
        auto transpose = make_shared<op::v1::Transpose>((multiout ? A1->output(0) : A1), perm);
        auto transpose1 = make_shared<op::v0::Abs>(transpose);
        auto baseline_f = make_shared<Function>(transpose1, ParameterVector{param});
        auto optimized_f = clone_function(*baseline_f);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::TransposeToReshape>();
        pass_manager.run_passes(optimized_f);

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;

        ASSERT_EQ(count_ops_of_type<op::v1::Transpose>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 0);
        ASSERT_EQ(count_ops_of_type<op::v1::Transpose>(optimized_f), num);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), (num ? 0 : 1));
    };

    for (auto& i32 : {true, false})
        for (auto& multiout : {true, false}) {
            check_usecase(Shape{1, 3}, vector<int64_t>{1, 0}, i32, multiout, 0);
            check_usecase(Shape{2, 3, 1}, vector<int64_t>{2, 0, 1}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 1}, vector<int64_t>{0, 2, 3, 1}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 20}, vector<int64_t>{0, 3, 1, 2}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 2}, vector<int64_t>{0, 2, 1, 3}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 1, 20}, vector<int64_t>{0, 4, 1, 2, 3}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 1, 1}, vector<int64_t>{0, 2, 3, 4, 1}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 1, 1}, vector<int64_t>{1, 4, 2, 3, 0}, i32, multiout, 0);
            check_usecase(Shape{10, 1, 1, 1, 1}, vector<int64_t>{4, 2, 0, 1, 3}, i32, multiout, 0);
            check_usecase(Shape{10, 20, 1, 2}, vector<int64_t>{0, 2, 3, 1}, i32, multiout, 1);
            check_usecase(Shape{10, 20, 1, 2}, vector<int64_t>{0, 3, 1, 2}, i32, multiout, 1);
            check_usecase(Shape{10, 20}, vector<int64_t>{1, 0}, i32, multiout, 1);

            check_usecase(PartialShape{Dimension::dynamic(), 20, 1, 1},
                          vector<int64_t>{
                              0, 2, 3, 1,
                          },
                          i32,
                          multiout,
                          0);
            check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 20, 1, 1},
                          vector<int64_t>{0, 1, 3, 2, 4},
                          i32,
                          multiout,
                          0);
            check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 20, 1, 1},
                          vector<int64_t>{0, 2, 1, 4, 3},
                          i32,
                          multiout,
                          1);
        }
}

// the following gather test will be used to test when
// gather is Nop and will be removed during `simplify_gather`
// algebraic_simplification pass

TEST(algebraic_simplification, gather_3d_indices_constant_axis_1) {
    auto check_usecase = [](const PartialShape& pshape,
                            bool i32,
                            bool multiout,
                            const std::vector<int64_t>& indices_val,
                            int64_t axis_val,
                            size_t num) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> indices;
        shared_ptr<Node> axis;
        if (i32) {
            std::vector<int32_t> indices_val_i32(indices_val.begin(), indices_val.end());
            indices = op::Constant::create<int32_t>(
                element::i32, Shape{indices_val.size()}, indices_val_i32);
            axis = op::Constant::create<int32_t>(element::i32, Shape{}, {(int32_t)axis_val});
        } else {
            indices =
                op::Constant::create<int64_t>(element::i64, Shape{indices_val.size()}, indices_val);
            axis = op::Constant::create<int64_t>(element::i64, Shape{}, {axis_val});
        }

        auto A = make_shared<op::Parameter>(element::f32, pshape);
        shared_ptr<Node> A1;
        if (multiout) {
            auto last_dim = pshape.rank().get_length() - 1;
            A1 = make_shared<op::v1::TopK>(A, op::Constant::create(element::i64, {}, {1}), last_dim, op::v1::TopK::Mode::MAX, op::v1::TopK::SortType::NONE);
        } else {
            A1 = make_shared<op::v0::Abs>(A);
        }
        auto G = make_shared<op::v1::Gather>((multiout ? A1->output(0) : A1), indices, axis);

        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(G), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::AlgebraicSimplification>();
        pass_manager.run_passes(optimized_f);

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;

        ASSERT_EQ(count_ops_of_type<op::v1::Gather>(baseline_f), 1) << casename;
        // the pass should short cut the Gather i/p with the gather users
        // since we are fetching the whole tensor using gather op
        ASSERT_EQ(count_ops_of_type<op::v1::Gather>(optimized_f), num) << casename;
    };
    for (auto& i32 : {true, false})
        for (auto& multiout : {true, false}) {
            check_usecase(PartialShape{1, 3, 2}, i32, multiout, std::vector<int64_t>{1}, 0, 0);
            check_usecase(PartialShape{3, 2, 1}, i32, multiout, std::vector<int64_t>{0, 1}, 1, 0);
            check_usecase(PartialShape{3, 2, 1}, i32, multiout, std::vector<int64_t>{1}, 2, 0);
            check_usecase(PartialShape{1, 16}, i32, multiout, std::vector<int64_t>{0, 0}, 0, 1);
        }
}

TEST(algebraic_simplification, gather_shapeof) {
    auto check_usecase = [](const PartialShape& pshape,
                            bool is_scalar_index,
                            bool opset2,
                            bool i32,
                            bool multiout,
                            bool multiout_1,
                            const std::vector<int64_t>& indices_val,
                            int64_t axis_val) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> indices;
        shared_ptr<Node> axis;
        if (i32) {
            std::vector<int32_t> indices_val_i32(indices_val.begin(), indices_val.end());
            indices = is_scalar_index
                          ? op::Constant::create<int32_t>(element::i32, Shape{}, indices_val_i32)
                          : op::Constant::create<int32_t>(
                                element::i32, Shape{indices_val.size()}, indices_val_i32);
            axis = op::Constant::create<int32_t>(element::i32, Shape{}, {(int32_t)axis_val});
        } else {
            indices = is_scalar_index
                          ? op::Constant::create<int64_t>(element::i64, Shape{}, indices_val)
                          : op::Constant::create<int64_t>(
                                element::i64, Shape{indices_val.size()}, indices_val);
            axis = op::Constant::create<int64_t>(element::i64, Shape{}, {axis_val});
        }

        auto dims_1 = std::vector<Dimension>(pshape);
        dims_1.push_back(11);
        dims_1.push_back(13);
        auto pshape_1 = PartialShape(dims_1);
        auto A = make_shared<op::Parameter>(element::f32, pshape);
        auto AA = make_shared<op::Parameter>(element::f64, pshape_1);
        shared_ptr<Node> A1;
        if (multiout) {
            A1 = make_shared<TestOpMultiOut>(A, AA);
        } else {
            A1 = make_shared<op::v0::Abs>(A);
        }
        auto B = make_shared<op::v1::Gather>(
            (multiout ? (multiout_1 ? A1->output(1) : A1->output(0)) : A1), indices, axis);
        shared_ptr<Node> B1;
        if (opset2) {
            B1 = make_shared<op::v0::ShapeOf>(B);
        } else {
            B1 = make_shared<op::v3::ShapeOf>(B);
        }
        auto baseline_f = make_shared<Function>(
            make_shared<op::v0::Abs>(B1), (multiout ? ParameterVector{A, AA} : ParameterVector{A}));
        auto optimized_f = clone_function(*baseline_f);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::AlgebraicSimplification>();
        pass_manager.run_passes(optimized_f);

        ASSERT_EQ(baseline_f->get_results().at(0)->get_element_type(),
                  optimized_f->get_results().at(0)->get_element_type());

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        EXPECT_TRUE(ps.same_scheme(ps_r)) << casename;

        ASSERT_EQ(count_ops_of_type<op::v1::Gather>(baseline_f), 1) << casename;

        auto last_node = optimized_f->get_results()[0]->input_value(0).get_node_shared_ptr();
        if (is_scalar_index) {
            ASSERT_EQ(count_ops_of_type<op::v3::ShapeOf>(optimized_f), 1) << casename;
            ASSERT_EQ(count_ops_of_type<op::v1::Gather>(optimized_f), 1) << casename;
            EXPECT_TRUE(
                as_type_ptr<op::v1::Gather>(last_node->input_value(0).get_node_shared_ptr()))
                << casename;
        } else {
            ASSERT_EQ(count_ops_of_type<op::v0::Concat>(optimized_f), 1) << casename;
            EXPECT_TRUE(
                as_type_ptr<op::v0::Concat>(last_node->input_value(0).get_node_shared_ptr()))
                << casename;
        }
    };

    for (auto& opset2 : {true, false})
        for (auto& i32 : {true, false})
            for (auto& multiout : {true, false})
                for (auto& multiout_1 : {true, false}) {
                    check_usecase(PartialShape{2, 3, 2, 1},
                                  true,
                                  opset2,
                                  i32,
                                  multiout,
                                  multiout_1,
                                  std::vector<int64_t>{0},
                                  3);
                    check_usecase(PartialShape{2, Dimension::dynamic(), 2, 1},
                                  true,
                                  opset2,
                                  i32,
                                  multiout,
                                  multiout_1,
                                  std::vector<int64_t>{0},
                                  3);
                }
    for (auto& opset2 : {true, false})
        for (auto& i32 : {true, false})
            for (auto& multiout : {true, false})
                for (auto& multiout_1 : {true, false}) {
                    check_usecase(PartialShape{2, 3, 2, 1},
                                  false,
                                  opset2,
                                  i32,
                                  multiout,
                                  multiout_1,
                                  std::vector<int64_t>{0, 2},
                                  1);
                    check_usecase(PartialShape{2, Dimension::dynamic(), 2, 1},
                                  false,
                                  opset2,
                                  i32,
                                  multiout,
                                  multiout_1,
                                  std::vector<int64_t>{0, 2},
                                  1);
                }
}
