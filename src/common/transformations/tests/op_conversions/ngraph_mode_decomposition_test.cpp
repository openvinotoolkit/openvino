// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/mod.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_mod.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST(TransformationTests, ModDecompositionTests) {
    auto data1 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 3}, {1, 2, 3});
    auto data2 = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {1, 2, 3});

    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto mod = std::make_shared<ngraph::op::v1::Mod>(data1, data2);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mod}, ngraph::ParameterVector{});
        auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitUniqueNames>(unh);
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertMod>();
        m.register_pass<ngraph::pass::CheckUniqueNames>(unh);
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(f->get_ops().size(), 12);
}
