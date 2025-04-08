// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/insert_movebroadcast.hpp"

#include "transformations/init_node_info.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;

//  todo: Rewrite this test using Snippets test infrastructure. See ./include/canonicalization.hpp for example

TEST_F(TransformationTestsF, InsertBroadcastMove) {
    {
        auto data0 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 1});
        auto add = std::make_shared<ov::op::v1::Add>(data0, data1);
        model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data0, data1});

        manager.register_pass<snippets::pass::InsertMoveBroadcast>();
    }
    {
        auto data0 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 1});
        auto move1 = std::make_shared<snippets::isa::BroadcastMove>(data1, ov::Dimension{3});
        auto add = std::make_shared<ov::op::v1::Add>(data0, move1);
        model_ref = std::make_shared<Model>(NodeVector{add}, ParameterVector{data0, data1});
    }
}
