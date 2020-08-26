// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <transformations/mvn_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, MVNFusion) {
    float eps = 0.01;
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 3, 2, 2});
        auto axes = std::make_shared<opset4::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 3});
        auto mean = std::make_shared<opset4::ReduceMean>(data, axes, true);
        auto numerator = std::make_shared<opset4::Subtract>(data, mean);
        auto mean_for_variance = std::make_shared<opset4::ReduceMean>(data, axes, true);
        auto diff = std::make_shared<opset4::Subtract>(data, mean_for_variance);
        auto sqr = std::make_shared<opset4::Power>(diff, std::make_shared<opset4::Constant>(element::f32, Shape{}, std::vector<float>{2}));
        auto variance = std::make_shared<opset4::ReduceMean>(sqr, axes, true);
        auto stddev = std::make_shared<opset4::Power>(variance, std::make_shared<opset4::Constant>(element::f32, Shape{}, std::vector<float>{0.5}));
        auto add = std::make_shared<opset4::Add>(stddev, std::make_shared<opset4::Constant>(element::f32, Shape{}, std::vector<float>{eps}));
        auto reciprocal = std::make_shared<opset4::Power>(add, std::make_shared<opset4::Constant>(element::f32, Shape{}, std::vector<float>{-1}));
        auto res = std::make_shared<opset4::Multiply>(numerator, reciprocal);

        f = std::make_shared<Function>(NodeVector{res}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::MVNFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 3, 2, 2});
        AxisSet axes{0, 2, 3};
        auto mvn = std::make_shared<opset4::MVN>(data, axes, true, eps);
        f_ref = std::make_shared<Function>(NodeVector{mvn}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
