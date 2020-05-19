//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "util/all_close.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(nop_elimination, eliminate_pad)
{
    Shape shape_a{2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    CoordinateDiff padding_below{0};
    CoordinateDiff padding_above{0};
    auto p = make_shared<op::v0::Pad>(A, B, padding_below, padding_above);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(p), ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Pad>(f), 0);
}

TEST(nop_elimination, eliminate_sum)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::v0::Sum>(A, AxisSet{});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Sum>(f), 0);
}

TEST(nop_elimination, eliminate_convert)
{
    Shape shape{};
    auto type = element::f32;
    auto A = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::v0::Convert>(A, element::f32);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(c), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(nop_elimination, convert_type_agnostic)
{
    Shape shape{};
    auto type = element::from<char>();
    auto A = make_shared<op::Parameter>(type, shape);
    auto c1 = make_shared<op::v0::Convert>(A, element::from<uint8_t>());
    auto c = make_shared<op::v0::Convert>(c1, element::f32);
    auto z = make_shared<op::v3::NonZero>(c);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(z), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(nop_elimination, eliminate_slice)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::v0::Slice>(A, Coordinate{0, 0}, Coordinate{2, 2});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Slice>(f), 0);
}

TEST(nop_elimination, eliminate_broadcast)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::v0::Broadcast>(A, shape, AxisSet{});
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(b), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Broadcast>(f), 0);
}

TEST(nop_elimination, eliminate_stop_gradient)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto s = make_shared<op::v0::StopGradient>(A);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(s), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::StopGradient>(f), 0);
}

TEST(nop_elimination, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::NopElimination>();
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}

TEST(nop_elimination, reshape_elimination_v1)
{
    auto generate_func = [](bool zero) {
        auto arg = std::make_shared<op::Parameter>(element::i64, PartialShape{8, 16, 2, 3});
        auto pattern_org = op::Constant::create(element::i64, Shape{3}, vector<int64_t>{8, 16, 6});
        auto pattern = op::Constant::create(element::i64, Shape{3}, vector<int64_t>{8, 16, 6});
        auto reshape_v1_org = std::make_shared<op::v1::Reshape>(arg, pattern_org, zero);
        auto reshape_v1 = std::make_shared<op::v1::Reshape>(reshape_v1_org, pattern, zero);
        auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
        return std::make_shared<Function>(NodeVector{abs}, ParameterVector{arg});
    };

    auto func = generate_func(false);
    auto nopass_func = generate_func(false);
    auto func_zero = generate_func(true);
    auto nopass_func_zero = generate_func(true);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(func);
    pass_manager.run_passes(func_zero);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func) == 2);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func) == 1);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(nopass_func_zero) == 2);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(func_zero) == 1);
}

TEST(nop_elimination, reshape_elimination_v1_dynamic)
{
    auto arg = std::make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto reshape_v1 = std::make_shared<op::v1::Reshape>(arg, pattern, false);
    auto abs = std::make_shared<op::v0::Abs>(reshape_v1);
    auto f = std::make_shared<Function>(NodeVector{abs}, ParameterVector{arg, pattern});
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::v1::Reshape>(f) == 1);
}

TEST(nop_elimination, concat_elimination_single_node)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto f =
        make_shared<Function>(make_shared<op::v0::Concat>(NodeVector{A}, a), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 1);
}

TEST(nop_elimination, concat_elimination_single_input)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto B = make_shared<op::v0::Concat>(NodeVector{A}, a);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
}

TEST(nop_elimination, concat_elimination_single_input_dynamic)
{
    int64_t a = 0;
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 3});
    auto B = make_shared<op::v0::Concat>(NodeVector{A}, a);
    auto f = make_shared<Function>(make_shared<op::v0::Abs>(B), ParameterVector{A});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Validate>();
    pass_manager.register_pass<pass::NopElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Concat>(f), 0);
}

TEST(nop_elimination, squeeze_unsqueeze_overlap_elimination)
{
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& sq_axes_val,
                            const std::vector<int64_t>& unsq_axes_val,
                            bool sq_to_unsq,
                            bool i32,
                            bool multiout,
                            size_t sc,
                            size_t usc,
                            size_t rc) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);

        shared_ptr<Node> sq_axes;
        shared_ptr<Node> unsq_axes;
        if (i32)
        {
            std::vector<int32_t> sq_axes_val_i32(sq_axes_val.begin(), sq_axes_val.end());
            std::vector<int32_t> unsq_axes_val_i32(unsq_axes_val.begin(), unsq_axes_val.end());
            sq_axes = op::Constant::create<int32_t>(
                element::i32, Shape{sq_axes_val.size()}, sq_axes_val_i32);
            unsq_axes = op::Constant::create<int32_t>(
                element::i32, Shape{unsq_axes_val.size()}, unsq_axes_val_i32);
        }
        else
        {
            sq_axes =
                op::Constant::create<int64_t>(element::i64, Shape{sq_axes_val.size()}, sq_axes_val);
            unsq_axes = op::Constant::create<int64_t>(
                element::i64, Shape{unsq_axes_val.size()}, unsq_axes_val);
        }

        auto A = make_shared<op::Parameter>(element::f32, shape);
        shared_ptr<Node> A1;
        if (multiout)
        {
            auto last_dim = shape.rank().get_length() - 1;
            A1 = make_shared<op::v0::TopK>(A, last_dim, element::i32);
        }
        else
        {
            A1 = make_shared<op::v0::Abs>(A);
        }

        shared_ptr<Node> B1;
        if (sq_to_unsq)
        {
            auto B = make_shared<op::v0::Squeeze>((multiout ? A1->output(0) : A1), sq_axes);
            B1 = make_shared<op::v0::Unsqueeze>(B, unsq_axes);
        }
        else
        {
            auto B = make_shared<op::v0::Unsqueeze>((multiout ? A1->output(0) : A1), unsq_axes);
            B1 = make_shared<op::v0::Squeeze>(B, sq_axes);
        }

        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::NopElimination>();
        pass_manager.run_passes(optimized_f);

        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;

        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), sc) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), usc) << casename;
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), rc) << casename;
    };

    // static shapes, all squeeze/unsqueeze replaced by reshape
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, false, false, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, true, false, 0, 0, 1);
    // multioutout ops
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, false, true, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{1}, {0}, {0, 1, 2, 3}, true, true, true, 0, 0, 1);

    // axes match - expect all squeeze/unsqueeze/reshape cancel out
    check_usecase(PartialShape{2, 1, 1, 6}, {1, 2}, {1, 2}, true, true, true, 0, 0, 0);

    // dynamic shapes - axes match, expect all cancel
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1},
                  {0, 2, 4},
                  {0, 2, 4},
                  true,
                  true,
                  true,
                  0,
                  0,
                  0);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, 2, 1},
                  {0, 2, 4},
                  {0, 2, 4},
                  true,
                  false,
                  true,
                  0,
                  0,
                  0);

    // squeeze axes overlap fully
    check_usecase(
        PartialShape{Dimension::dynamic(), 1, 1, 3}, {1, 2}, {1, 2, 3}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {1, 2},
                  {1, 2, 3},
                  true,
                  true,
                  true,
                  0,
                  1,
                  0);
    check_usecase(PartialShape{2, 1, 1, 4}, {1, 2}, {1, 2, 3}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, Dimension::dynamic(), Dimension::dynamic()},
                  {1, 2},
                  {1, 2, 3},
                  true,
                  true,
                  true,
                  0,
                  1,
                  0);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {2, 3},
                  {2, 3, 5},
                  true,
                  true,
                  true,
                  0,
                  1,
                  0);

    // unsqueeze axes overlap fully
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, 1, 1, Dimension::dynamic(), 3},
                  {2, 3},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  0,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1},
                  {2, 3},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  0,
                  0);
    check_usecase(
        PartialShape{Dimension::dynamic(), 3, 1, 1}, {2, 3}, {2}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{3, 1, 1}, {1, 2}, {1}, true, true, true, 0, 0, 1);

    // squeeze->unsqueeze axes overlap
    check_usecase(
        PartialShape{Dimension::dynamic(), 1, 1, 4}, {1, 2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {1, 2},
                  {0},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{3, 1, 1, 4}, {1, 2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{2, 1, 1, Dimension::dynamic(), Dimension::dynamic()},
                  {1, 2},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, 3, Dimension::dynamic(), 4},
                  {1, 2},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{2, 1, Dimension::dynamic(), 1, Dimension::dynamic()},
                  {1, 3},
                  {3},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1, 1, 4},
                  {4, 5},
                  {1, 5},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);

    //
    // Unsqueeze->Squeeze cases, testcase 23 - ..
    //
    // static shapes, all unsqueeze/squeeze replaced by reshape
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, false, false, 0, 0, 1);
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, true, false, 0, 0, 1);
    // multioutout ops
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, false, true, 0, 0, 1);
    check_usecase(PartialShape{2, 6, 1}, {4}, {1, 2}, false, true, true, 0, 0, 1);
    check_usecase(PartialShape{1}, {0}, {0, 1, 2, 3}, false, true, true, 0, 0, 1);
    check_usecase(PartialShape{3, 1, 1, 4}, {2, 3}, {0}, false, true, true, 0, 0, 1);
    // dynamic shapes
    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1},
                  {0, 2, 4},
                  {0, 2, 4},
                  false,
                  true,
                  true,
                  0,
                  0,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, Dimension::dynamic()},
                  {2},
                  {0},
                  true,
                  true,
                  true,
                  1,
                  1,
                  0);
    check_usecase(PartialShape{Dimension::dynamic(), 1, 1, 4}, {2}, {0}, true, true, true, 0, 0, 1);
    check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1},
                  {2, 3},
                  {2},
                  true,
                  true,
                  true,
                  1,
                  0,
                  0);
}

TEST(nop_elimination, squeeze_squeeze_overlap_elimination)
{
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& sq_axes_val_1,
                            const std::vector<int64_t>& sq_axes_val_2,
                            size_t sq) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);
        auto sq_axes_1 =
            op::Constant::create<int64_t>(element::i64, Shape{sq_axes_val_1.size()}, sq_axes_val_1);
        auto sq_axes_2 =
            op::Constant::create<int64_t>(element::i64, Shape{sq_axes_val_2.size()}, sq_axes_val_2);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Squeeze>(A1, sq_axes_1);
        auto B1 = make_shared<op::v0::Squeeze>(B, sq_axes_2);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::NopElimination>();
        pass_manager.run_passes(optimized_f);
        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 2) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), sq) << casename;
        ;
    };

    check_usecase(PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic()}, {0}, {1}, 1);
    check_usecase(
        PartialShape{1, 1, 1, Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {2, 1}, {2, 4}, 1);
    check_usecase(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), 1, 1}, {-1, -5}, {2}, 1);
    check_usecase(
        PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {0}, {1, 3}, 1);
}

TEST(nop_elimination, unsqueeze_unsqueeze_overlap_elimination)
{
    auto check_usecase = [](const PartialShape& shape,
                            const std::vector<int64_t>& unsq_axes_val_1,
                            const std::vector<int64_t>& unsq_axes_val_2,
                            size_t unsq) {
        static size_t id = 0;
        auto casename = string("usecase #") + to_string(++id);
        auto unsq_axes_1 = op::Constant::create<int64_t>(
            element::i64, Shape{unsq_axes_val_1.size()}, unsq_axes_val_1);
        auto unsq_axes_2 = op::Constant::create<int64_t>(
            element::i64, Shape{unsq_axes_val_2.size()}, unsq_axes_val_2);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Unsqueeze>(A1, unsq_axes_1);
        auto B1 = make_shared<op::v0::Unsqueeze>(B, unsq_axes_2);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::Validate>();
        pass_manager.register_pass<pass::NopElimination>();
        pass_manager.run_passes(optimized_f);
        auto ps = baseline_f->get_results()[0]->get_output_partial_shape(0);
        auto ps_r = optimized_f->get_results()[0]->get_output_partial_shape(0);
        EXPECT_TRUE(ps.rank().is_static() && ps_r.rank().is_static()) << casename;
        ASSERT_EQ(ps.rank().get_length(), ps_r.rank().get_length()) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 2) << casename;
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), unsq) << casename;
        ;
    };

    check_usecase(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()}, {0}, {2}, 1);
    check_usecase(
        PartialShape{1, Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {2, 1}, {2, 4}, 1);
    check_usecase(PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1}, {-1, -3}, {2}, 1);
    check_usecase(PartialShape{Dimension::dynamic(), 1, Dimension::dynamic(), 1}, {0}, {1, 3}, 1);
}

TEST(nop_elimination, unsqueeze_squeeze_elimination)
{
    auto generate_func = [](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::v0::Unsqueeze>(A1, axes);
        auto B1 = make_shared<op::v0::Squeeze>(B, axes);
        return make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
    };

    auto check_usecase = [&](const Shape& shape, const std::vector<int64_t>& axes_val) {
        auto baseline_f = generate_func(shape, axes_val);
        auto optimized_f = generate_func(shape, axes_val);
        EXPECT_TRUE((compare_pass_int<pass::NopElimination, float>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{6}, std::vector<int64_t>{0});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{0, 3});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{0, 2, 4});
    check_usecase(Shape{3, 2}, std::vector<int64_t>{-1, -4});
}

TEST(nop_elimination, reshape_unsqueeze_elimination)
{
    auto check_usecase = [](const Shape& shape,
                            const std::vector<int64_t>& pat_val,
                            bool zero,
                            const std::vector<int64_t>& axes_val) {
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
        auto pat = op::Constant::create<int64_t>(element::i64, Shape{pat_val.size()}, pat_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v1::Reshape>(A1, pat, zero);
        auto pat2 =
            op::Constant::create<int64_t>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
        auto B1 = make_shared<op::v0::Unsqueeze>(B, axes);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);
        EXPECT_TRUE((compare_pass_int<pass::NopElimination, float>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{1, 2, 3, 2, 1}, {2, 3, 2}, false, {2, 4});
    check_usecase(Shape{12}, {2, 3, 2}, false, {3});
    check_usecase(Shape{3, 2, 1, 2}, {0, 2, 2}, true, {1, 4});
    check_usecase(Shape{2, 3, 2}, {2, -1, 2}, false, {2});
    check_usecase(Shape{2, 3, 2, 1}, {2, 3, 2}, false, {0});
}
TEST(nop_elimination, reshape_squeeze_elimination)
{
    auto check_usecase = [](const Shape& shape,
                            const std::vector<int64_t>& pat_val,
                            bool zero,
                            const std::vector<int64_t>& axes_val) {
        auto axes = op::Constant::create<int64_t>(element::i64, Shape{axes_val.size()}, axes_val);
        auto pat = op::Constant::create<int64_t>(element::i64, Shape{pat_val.size()}, pat_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v1::Reshape>(A1, pat, zero);
        auto pat2 =
            op::Constant::create<int64_t>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
        auto B1 = make_shared<op::v0::Squeeze>(B, axes);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);
        EXPECT_TRUE((compare_pass_int<pass::NopElimination, float>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
    };

    check_usecase(Shape{1, 2, 3, 2, 1}, {2, 3, 1, 2, 1}, false, {2, 4});
    check_usecase(Shape{12}, {2, 3, 2, 1}, false, {3});
    check_usecase(Shape{3, 2, 1, 2}, {0, 1, 2, 2, 1}, true, {1, 4});
    check_usecase(Shape{2, 3, 2}, {2, -1, 1, 2}, false, {2});
    check_usecase(Shape{2, 3, 2, 1}, {1, 2, 3, 2}, false, {0});
}

TEST(nop_elimination, reshape_reshape_elimination)
{
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& pat_val, bool zero) {
        auto pat = op::Constant::create<int64_t>(element::i64, Shape{pat_val.size()}, pat_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v1::Reshape>(A1, pat, zero);
        auto pat2 =
            op::Constant::create<int64_t>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
        auto B1 = make_shared<op::v1::Reshape>(B, pat2, true);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);
        EXPECT_TRUE((compare_pass_int<pass::NopElimination, float>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 2);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), 1);
    };

    check_usecase(Shape{1, 2, 3, 2, 1}, std::vector<int64_t>{2, 3, 2}, false);
    check_usecase(Shape{12}, std::vector<int64_t>{2, 3, 2}, false);
    check_usecase(Shape{3, 2, 1, 2}, std::vector<int64_t>{0, 2, 2}, true);
    check_usecase(Shape{2, 3, 2}, ::vector<int64_t>{2, -1, 2}, false);
    check_usecase(Shape{2, 3, 2, 1}, ::vector<int64_t>{2, 3, 2}, false);
}

TEST(nop_elimination, squeeze_reshape_elimination)
{
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& indices_val) {
        auto indices =
            op::Constant::create<int64_t>(element::i64, Shape{indices_val.size()}, indices_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v0::Squeeze>(A1, indices);
        auto pat2 = op::Constant::create<int64_t>(element::i64, Shape{1}, std::vector<int64_t>{-1});
        auto B1 = make_shared<op::v1::Reshape>(B, pat2, false);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);
        EXPECT_TRUE((compare_pass_int<pass::NopElimination, float>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Squeeze>(optimized_f), 0);
    };

    check_usecase(Shape{1, 2, 3, 2, 1}, std::vector<int64_t>{0, 4});
    check_usecase(Shape{1, 1}, std::vector<int64_t>{0, 1});
    check_usecase(Shape{2, 3, 1, 2}, std::vector<int64_t>{2});
    check_usecase(Shape{1, 6, 2, 1}, std::vector<int64_t>{3});
}

TEST(nop_elimination, unsqueeze_reshape_elimination)
{
    auto check_usecase = [](const Shape& shape, const std::vector<int64_t>& indices_val) {
        auto indices =
            op::Constant::create<int64_t>(element::i64, Shape{indices_val.size()}, indices_val);
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto A1 = make_shared<op::v0::Abs>(A);

        auto B = make_shared<op::v0::Unsqueeze>(A1, indices);
        auto pat2 = op::Constant::create<int64_t>(element::i64, Shape{1}, std::vector<int64_t>{-1});
        auto B1 = make_shared<op::v1::Reshape>(B, pat2, false);
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(B1), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);
        EXPECT_TRUE((compare_pass_int<pass::NopElimination, float>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(optimized_f), 1);
        ASSERT_EQ(count_ops_of_type<op::v0::Unsqueeze>(optimized_f), 0);
    };

    check_usecase(Shape{2, 3, 2}, std::vector<int64_t>{0, 4});
    check_usecase(Shape{}, std::vector<int64_t>{0, 1});
    check_usecase(Shape{2, 3, 2}, std::vector<int64_t>{2});
    check_usecase(Shape{1, 6, 2}, std::vector<int64_t>{3});
}

TEST(nop_elimination, topk_convert_elimination)
{
    auto check_usecase = []() {
        auto A = make_shared<op::Parameter>(element::f32, Shape{20, 3, 4});
        auto A1 = make_shared<op::v0::Abs>(A);
        auto B = make_shared<op::TopK>(A1, 0, element::i64, 10);
        auto C = make_shared<op::Convert>(B->output(0), B->output(0).get_element_type());
        auto baseline_f = make_shared<Function>(make_shared<op::v0::Abs>(C), ParameterVector{A});
        auto optimized_f = clone_function(*baseline_f);
        EXPECT_TRUE(
            (compare_pass_int<pass::NopElimination, float, int64_t>(baseline_f, optimized_f)));

        ASSERT_EQ(count_ops_of_type<op::Convert>(baseline_f), 1);
        ASSERT_EQ(count_ops_of_type<op::Convert>(optimized_f), 0);
    };

    check_usecase();
}
