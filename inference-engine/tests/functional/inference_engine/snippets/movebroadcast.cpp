// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/insert_movebroadcast.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, InsertBroadcastMove) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<opset1::Add>(data0, data1);
        f = std::make_shared<Function>(NodeVector{add}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<snippets::pass::InsertMoveBroadcast>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data0 = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3});
        auto move = std::make_shared<snippets::isa::BroadcastMove>(data1, data0->output(0).get_shape());
        auto add = std::make_shared<opset1::Add>(data0, move);
        f_ref = std::make_shared<Function>(NodeVector{add}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
