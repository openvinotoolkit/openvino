// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/load_movebroadcast_to_broadcastload.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

//  todo: Rewrite this test using Snippets test infrastructure. See ./include/canonicalization.hpp for example

TEST(TransformationTests, FuseLoadWithBroadcastMoveByX) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 1});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load0 = std::make_shared<snippets::isa::Load>(data0);
        auto load1 = std::make_shared<snippets::isa::Load>(data1);
        auto bct = std::make_shared<snippets::isa::BroadcastMove>(load0, load1->get_shape());
        auto add = std::make_shared<opset1::Add>(bct, load1);
        auto store = std::make_shared<snippets::isa::Store>(add);
        f = std::make_shared<Function>(NodeVector{store}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::LoadMoveBroadcastToBroadcastLoad>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 1});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto load0 = std::make_shared<snippets::isa::BroadcastLoad>(data0, data1->get_shape());
        auto load1 = std::make_shared<snippets::isa::Load>(data1);
        auto add = std::make_shared<opset1::Add>(load0, load1);
        auto store = std::make_shared<snippets::isa::Store>(add);
        f_ref = std::make_shared<Function>(NodeVector{store}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NoFuseLoadWithBroadcastMoveMultipleUsers) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 1});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 1});

        auto load0 = std::make_shared<snippets::isa::Load>(data0);
        auto load1 = std::make_shared<snippets::isa::Load>(data1);
        auto load2 = std::make_shared<snippets::isa::Load>(data2);

        auto bct1 = std::make_shared<snippets::isa::BroadcastMove>(load1, load0->get_shape());

        auto add = std::make_shared<opset1::Add>(load0, bct1);
        auto mul = std::make_shared<opset1::Multiply>(load1, load2);

        auto store0 = std::make_shared<snippets::isa::Store>(add);
        auto store1 = std::make_shared<snippets::isa::Store>(mul);
        f = std::make_shared<Function>(NodeVector{store0, store1}, ParameterVector{data0, data1, data2});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::LoadMoveBroadcastToBroadcastLoad>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 1});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 1});

        auto load0 = std::make_shared<snippets::isa::Load>(data0);
        auto load1 = std::make_shared<snippets::isa::Load>(data1);
        auto load2 = std::make_shared<snippets::isa::Load>(data2);

        auto bct1 = std::make_shared<snippets::isa::BroadcastMove>(load1, load0->get_shape());

        auto add = std::make_shared<opset1::Add>(load0, bct1);
        auto mul = std::make_shared<opset1::Multiply>(load1, load2);

        auto store0 = std::make_shared<snippets::isa::Store>(add);
        auto store1 = std::make_shared<snippets::isa::Store>(mul);
        f_ref = std::make_shared<Function>(NodeVector{store0, store1}, ParameterVector{data0, data1, data2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
