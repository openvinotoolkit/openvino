// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/collapse_subgraph.hpp>
#include <snippets/op/subgraph.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, StartSubgraphMultipleOutputs) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<opset1::Add>(data0, data1);
        auto sub = std::make_shared<opset1::Subtract>(add, data1);
        auto mul = std::make_shared<opset1::Multiply>(add, sub);
        f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::StartSubgraph>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<snippets::op::Subgraph>(NodeVector{data0, data1},
            std::make_shared<Function>(NodeVector{std::make_shared<opset1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto sub = std::make_shared<opset1::Subtract>(add, data1);
        auto mul = std::make_shared<opset1::Multiply>(add, sub);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DontStartSubgraphSingleOuptut) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<opset1::Add>(data0, data1);
        auto sub = std::make_shared<opset1::Subtract>(add, data1);
        auto mul = std::make_shared<opset1::Multiply>(data0, sub);
        f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::StartSubgraph>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<opset1::Add>(data0, data1);
        auto sub = std::make_shared<opset1::Subtract>(add, data1);
        auto mul = std::make_shared<opset1::Multiply>(data0, sub);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, AttachToSubgraph) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<snippets::op::Subgraph>(NodeVector{data0, data1},
            std::make_shared<Function>(NodeVector{std::make_shared<opset1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto neg = std::make_shared<opset1::Negative>(add);
        auto concat = std::make_shared<opset1::Concat>(NodeVector{add, neg}, 0);
        f = std::make_shared<Function>(NodeVector{concat}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::AttachToSubgraph>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto inner = std::make_shared<opset1::Add>(indata0, indata1);
        auto add = std::make_shared<snippets::op::Subgraph>(NodeVector{data0, data1},
            std::make_shared<Function>(NodeVector{std::make_shared<opset1::Negative>(inner), inner}, ParameterVector{indata0, indata1}));
        auto concat = std::make_shared<opset1::Concat>(OutputVector{add->output(0), add->output(1)}, 0);
        f_ref = std::make_shared<Function>(NodeVector{concat}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DontAttachToSubgraphIfLoop) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<snippets::op::Subgraph>(NodeVector{data0, data1},
            std::make_shared<Function>(NodeVector{std::make_shared<opset1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto log = std::make_shared<opset1::Log>(add);
        auto mul = std::make_shared<opset1::Multiply>(add, log);
        f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::AttachToSubgraph>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<snippets::op::Subgraph>(NodeVector{data0, data1},
            std::make_shared<Function>(NodeVector{std::make_shared<opset1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto log = std::make_shared<opset1::Log>(add);
        auto mul = std::make_shared<opset1::Multiply>(add, log);
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}