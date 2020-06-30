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
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(CSE, abs_abs)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs1 = std::make_shared<op::Abs>(A);
    auto abs2 = std::make_shared<op::Abs>(A);
    auto f = std::make_shared<Function>(NodeVector{abs1, abs2}, ParameterVector{A});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, abs_abs_negative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs1 = std::make_shared<op::Abs>(A);
    auto abs2 = std::make_shared<op::Abs>(B);
    auto f = std::make_shared<Function>(NodeVector{abs1, abs2}, ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), abs1);
    ASSERT_EQ(f->get_results().at(1)->get_argument(0), abs2);
}

TEST(CSE, add_add)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add1 = std::make_shared<op::v1::Add>(A, B);
    auto add2 = std::make_shared<op::v1::Add>(A, B);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, add_add_commutative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add1 = std::make_shared<op::v1::Add>(A, B);
    auto add2 = std::make_shared<op::v1::Add>(B, A);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, add_add_negative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto C = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto D = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add1 = std::make_shared<op::v1::Add>(A, B);
    auto add2 = std::make_shared<op::v1::Add>(C, D);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, ParameterVector{A, B, C, D});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), add1);
    ASSERT_EQ(f->get_results().at(1)->get_argument(0), add2);
}

TEST(CSE, abs_add)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::v1::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::v1::Add>(abs_a2, abs_b2);
    auto f = std::make_shared<Function>(NodeVector{add1, add2}, ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, abs_add_reshape_broadcast)
{
    Shape zero_shape{1};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::v1::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::v1::Add>(abs_a2, abs_b2);
    {
        // success case
        auto reshape1 = std::make_shared<op::Reshape>(add1, AxisVector{0}, Shape{1, 1});
        auto reshape2 = std::make_shared<op::Reshape>(add2, AxisVector{0}, Shape{1, 1});
        auto broadcast1 = std::make_shared<op::Broadcast>(reshape1, Shape{1, 1, 3}, AxisSet{2});
        auto broadcast2 = std::make_shared<op::Broadcast>(reshape2, Shape{1, 1, 3}, AxisSet{2});
        auto f =
            std::make_shared<Function>(NodeVector{broadcast1, broadcast2}, ParameterVector{A, B});
        pass::Manager pass_manager;

        pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
        pass_manager.run_passes(f);
        ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
    }
    {
        // fail case
        auto reshape1 = std::make_shared<op::Reshape>(add1, AxisVector{0}, Shape{1});
        auto reshape2 = std::make_shared<op::Reshape>(add2, AxisVector{0}, Shape{1, 1});
        auto f = std::make_shared<Function>(NodeVector{reshape1, reshape2}, ParameterVector{A, B});
        pass::Manager pass_manager;

        pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
        pass_manager.run_passes(f);
        ASSERT_NE(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
    }
    {
        // fail case
        auto broadcast1 = std::make_shared<op::Broadcast>(add1, Shape{1, 2}, AxisSet{1});
        auto broadcast2 = std::make_shared<op::Broadcast>(add2, Shape{1, 1, 2}, AxisSet{1, 2});
        auto f =
            std::make_shared<Function>(NodeVector{broadcast1, broadcast2}, ParameterVector{A, B});
        pass::Manager pass_manager;

        pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
        pass_manager.run_passes(f);
        ASSERT_NE(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
    }
}

TEST(CSE, abs_add_abs_add)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::v1::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::v1::Add>(abs_a2, abs_b2);
    auto abs_add1 = std::make_shared<op::Abs>(add1);
    auto abs_add2 = std::make_shared<op::Abs>(add2);
    auto C = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add3 = std::make_shared<op::v1::Add>(abs_add1, C);
    auto add4 = std::make_shared<op::v1::Add>(abs_add2, C);
    auto f = std::make_shared<Function>(NodeVector{add3, add4}, ParameterVector{A, B, C});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
}

TEST(CSE, abs_add_abs_add_negative)
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto B = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto abs_a1 = std::make_shared<op::Abs>(A);
    auto abs_b1 = std::make_shared<op::Abs>(B);
    auto abs_a2 = std::make_shared<op::Abs>(A);
    auto abs_b2 = std::make_shared<op::Abs>(B);
    auto add1 = std::make_shared<op::v1::Add>(abs_a1, abs_b1);
    auto add2 = std::make_shared<op::v1::Add>(abs_a2, abs_b2);
    auto abs_add1 = std::make_shared<op::Abs>(add1);
    auto abs_add2 = std::make_shared<op::Abs>(add2);
    auto C = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto D = std::make_shared<op::Parameter>(element::i32, zero_shape);
    auto add3 = std::make_shared<op::v1::Add>(abs_add1, C);
    auto add4 = std::make_shared<op::v1::Add>(abs_add2, D);
    auto f = std::make_shared<Function>(NodeVector{add3, add4}, ParameterVector{A, B, C, D});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);
    auto oadd3 = f->get_results().at(0)->get_argument(0);
    auto oadd4 = f->get_results().at(1)->get_argument(0);
    ASSERT_EQ(oadd3, add3);
    ASSERT_EQ(oadd4, add4);
    ASSERT_EQ(oadd3->get_argument(1), C);
    ASSERT_EQ(oadd4->get_argument(1), D);
    ASSERT_EQ(oadd3->get_argument(0), oadd4->get_argument(0));
}

template <typename T>
static void execute_cse_reduction_test()
{
    Shape zero_shape{0};
    auto A = std::make_shared<op::Parameter>(element::i32, Shape{3, 5});
    auto a_reduction_op = std::make_shared<T>(A, AxisSet{0, 1});
    auto a_reduction_op2 = std::make_shared<T>(A, AxisSet{0, 1});
    auto a_reduction_op3 = std::make_shared<T>(A, AxisSet{0});
    auto sub_aa = a_reduction_op - a_reduction_op2;

    auto B = std::make_shared<op::Parameter>(element::i32, Shape{3, 5});
    auto b_reduction_op = std::make_shared<T>(B, AxisSet{0, 1});

    auto sub_ab = a_reduction_op - b_reduction_op;
    auto f = std::make_shared<Function>(NodeVector{sub_aa, sub_ab, a_reduction_op3},
                                        ParameterVector{A, B});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(sub_aa->get_argument(0), sub_aa->get_argument(1));
    ASSERT_NE(sub_ab->get_argument(0), sub_ab->get_argument(1));
    ASSERT_NE(f->get_results().at(2)->get_argument(0), sub_aa->get_argument(0));
}

TEST(CSE, reduction_ops)
{
    execute_cse_reduction_test<op::Sum>();
    execute_cse_reduction_test<op::Product>();
}

TEST(CSE, constant)
{
    Shape zero_shape{0};
    auto iconst0 = op::Constant::create(element::i32, Shape{}, {0});
    auto iconst0_1 = op::Constant::create(element::i32, Shape{}, {0});
    auto iconst1 = op::Constant::create(element::i32, Shape{}, {1});
    auto iconst1_1 = op::Constant::create(element::i32, Shape{}, {1});
    auto fconst0 = op::Constant::create(element::f32, Shape{}, {0});
    auto iconst111 = op::Constant::create(element::i32, Shape{3}, {1, 1, 1});
    auto iconst112 = op::Constant::create(element::i32, Shape{3}, {1, 1, 2});

    auto abs0 = std::make_shared<op::Abs>(iconst0);
    auto abs0_1 = std::make_shared<op::Abs>(iconst0_1);

    auto abs1 = std::make_shared<op::Abs>(iconst1);
    auto abs1_1 = std::make_shared<op::Abs>(iconst1_1);

    auto absf = std::make_shared<op::Abs>(fconst0);

    auto abs111 = std::make_shared<op::Abs>(iconst111);
    auto abs112 = std::make_shared<op::Abs>(iconst112);

    auto f = std::make_shared<Function>(
        NodeVector{abs0, abs0_1, abs1, abs1_1, absf, abs111, abs112}, ParameterVector{});
    pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(abs0->get_argument(0), abs0_1->get_argument(0));
    ASSERT_EQ(abs1->get_argument(0), abs1_1->get_argument(0));
    ASSERT_NE(abs0->get_argument(0), abs1->get_argument(0));
    ASSERT_NE(abs0->get_argument(0), absf->get_argument(0));
    ASSERT_NE(abs111->get_argument(0), abs112->get_argument(0));
}

TEST(CSE, one_hot)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    {
        Shape param_shape{8};
        Shape out_shape{8, 16};
        auto A = std::make_shared<op::Parameter>(element::i32, param_shape);
        auto onehot1 = std::make_shared<op::OneHot>(A, out_shape, 1);
        auto onehot2 = std::make_shared<op::OneHot>(A, out_shape, 1);
        auto f = std::make_shared<Function>(NodeVector{onehot1, onehot2}, ParameterVector{A});
        pass_manager.run_passes(f);
        ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
    }
    {
        Shape param_shape{8, 1};
        Shape out_shape{8, 16};
        auto A = std::make_shared<op::Parameter>(element::i32, param_shape);
        auto reshape1 = std::make_shared<op::Reshape>(A, AxisVector{0, 1}, Shape{8});
        auto reshape2 = std::make_shared<op::Reshape>(A, AxisVector{0, 1}, Shape{8});
        auto onehot1 = std::make_shared<op::OneHot>(reshape1, out_shape, 1);
        auto onehot2 = std::make_shared<op::OneHot>(reshape2, out_shape, 1);
        auto f = std::make_shared<Function>(NodeVector{onehot1, onehot2}, ParameterVector{A});
        pass_manager.run_passes(f);
        ASSERT_EQ(f->get_results().at(0)->get_argument(0), f->get_results().at(1)->get_argument(0));
    }
}

TEST(CSE, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::CommonSubexpressionElimination>();
    ASSERT_TRUE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
