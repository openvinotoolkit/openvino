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
#include <ngraph/op/fused/mod.hpp>
#include <transformations/convert_mod.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include "ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ModDecompositionTests) {
    auto data1 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 3}, {1, 2, 3});
    auto data2 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {1, 2, 3});

    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto mod = std::make_shared<ngraph::op::v1::Mod>(data1, data2);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mod}, ngraph::ParameterVector{});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMod().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(f->get_ops().size(), 12);
}
