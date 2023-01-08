// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/insert_load_store.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

//  todo: Rewrite this test using Snippets test infrastructure. See ./include/canonicalization.hpp for example

TEST(TransformationTests, InsertLoadStore) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto neg = std::make_shared<opset1::Negative>(data);
        f = std::make_shared<Function>(NodeVector{neg}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::InsertLoad>();
        m.register_pass<snippets::pass::InsertStore>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data);
        auto neg = std::make_shared<opset1::Negative>(load);
        auto store = std::make_shared<snippets::isa::Store>(neg);
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, InsertLoadTwise) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto neg = std::make_shared<opset1::Negative>(data);
        f = std::make_shared<Function>(NodeVector{neg}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::InsertLoad>();
        m.register_pass<snippets::pass::InsertLoad>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load = std::make_shared<snippets::isa::Load>(data);
        auto neg = std::make_shared<opset1::Negative>(load);
        f_ref = std::make_shared<Function>(NodeVector{neg}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, InsertStoreTwise) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto neg = std::make_shared<opset1::Negative>(data);
        f = std::make_shared<Function>(NodeVector{neg}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::InsertStore>();
        m.register_pass<snippets::pass::InsertStore>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto neg = std::make_shared<opset1::Negative>(data);
        auto store = std::make_shared<snippets::isa::Store>(neg);
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}