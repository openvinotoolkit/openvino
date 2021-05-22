// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "pass/shape_relevance.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;
using namespace std;

TEST(shape_relevance, simple)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 6});
    auto x = make_shared<op::v1::Add>(param0, param1);

    auto f = make_shared<Function>(x, ParameterVector{param0, param1});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
    ASSERT_FALSE(param1->is_relevant_to_shapes());
}

TEST(shape_relevance, param_direct)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});
    auto param1 = make_shared<op::Parameter>(element::i64, Shape{4});
    auto x = make_shared<op::v1::Reshape>(param0, param1, true);

    auto f = make_shared<Function>(x, ParameterVector{param0, param1});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
    ASSERT_TRUE(param1->is_relevant_to_shapes());
}

TEST(shape_relevance, param_indirect)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});
    auto param1 = make_shared<op::Parameter>(element::i64, Shape{4});
    auto param2 = make_shared<op::Parameter>(element::i64, Shape{2});

    auto c = make_shared<op::Concat>(NodeVector{param1, param2}, 0);
    auto x = make_shared<op::v1::Reshape>(param0, c, true);

    auto f = make_shared<Function>(x, ParameterVector{param0, param1, param2});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
    ASSERT_TRUE(param1->is_relevant_to_shapes());
    ASSERT_TRUE(param2->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_direct_v0)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto x = make_shared<op::v1::Reshape>(param0, make_shared<op::v0::ShapeOf>(param0), true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_direct_v3)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto x = make_shared<op::v1::Reshape>(param0, make_shared<op::v3::ShapeOf>(param0), true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_direct_i32_v3)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto x = make_shared<op::v1::Reshape>(
        param0, make_shared<op::v3::ShapeOf>(param0, element::i32), true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_indirect_v0)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto s = make_shared<op::v0::ShapeOf>(param0);
    auto r = make_shared<op::v1::Reverse>(
        s, op::Constant::create(element::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX);
    auto x = make_shared<op::v1::Reshape>(param0, r, true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_indirect_v3)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto s = make_shared<op::v3::ShapeOf>(param0);
    auto r = make_shared<op::v1::Reverse>(
        s, op::Constant::create(element::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX);
    auto x = make_shared<op::v1::Reshape>(param0, r, true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}

TEST(shape_relevance, param_shape_of_indirect_i32_v3)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto s = make_shared<op::v3::ShapeOf>(param0, element::i32);
    auto r = make_shared<op::v1::Reverse>(
        s, op::Constant::create(element::i64, {1}, {0}), op::v1::Reverse::Mode::INDEX);
    auto x = make_shared<op::v1::Reshape>(param0, r, true);

    auto f = make_shared<Function>(x, ParameterVector{param0});

    pass::Manager manager;
    manager.register_pass<pass::ShapeRelevance>();
    manager.run_passes(f);

    ASSERT_FALSE(param0->is_relevant_to_shapes());
}
