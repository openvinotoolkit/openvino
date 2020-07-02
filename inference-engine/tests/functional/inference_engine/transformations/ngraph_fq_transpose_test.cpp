// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/fake_quantize.hpp>
#include <transformations/pull_transpose_through_fq.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, FQTransposeTest1) {
    auto data1 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 3}, {1, 2, 3});
    auto data2 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {1, 2, 3});
    auto data3 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3}, {1, 2, 3});
    auto data4 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3}, {1, 2, 3});
    auto data5 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3}, {1, 2, 3});
    auto transpose_order = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});

    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto fq = std::make_shared<ngraph::op::FakeQuantize>(data1, data2, data3, data4, data5, 1);
        auto transpose = std::make_shared<ngraph::op::Transpose>(fq, transpose_order);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::PullTransposeThroughFQUp>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
    }
    std::vector<size_t> ref_shape{1, 3, 1};
    for (auto op : f->get_ops()) {
        if (auto constant = ngraph::as_type_ptr<ngraph::op::Constant>(op)) {
            auto shape = constant->get_shape();
            ASSERT_EQ(shape, ref_shape);
        }
    }
}
