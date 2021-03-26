// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, normalize_l2_op)
{
    NodeBuilder::get_ops().register_factory<opset1::NormalizeL2>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{1});
    const auto axes = make_shared<op::Constant>(element::i32, Shape{}, vector<int32_t>{0});

    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize_l2 = make_shared<opset1::NormalizeL2>(data, axes, eps, eps_mode);
    NodeBuilder builder(normalize_l2);
    auto g_normalize_l2 = as_type_ptr<opset1::NormalizeL2>(builder.create());

    EXPECT_EQ(g_normalize_l2->get_eps(), normalize_l2->get_eps());
    EXPECT_EQ(g_normalize_l2->get_eps_mode(), normalize_l2->get_eps_mode());
}
