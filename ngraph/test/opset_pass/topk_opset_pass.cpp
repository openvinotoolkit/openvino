//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "opset0_downgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_topk_upgrade_pass)
{
    const size_t axis = 2;
    const size_t k = 10;
    const auto data = make_shared<op::Parameter>(element::i32, Shape{5, 10, 15});
    const auto topk_v0 = make_shared<op::v0::TopK>(data, axis, element::i32, k);
    const auto result = make_shared<op::Result>(topk_v0->output(0));
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto topk_v1 = as_type_ptr<op::v1::TopK>(pass_replacement_node);
    ASSERT_TRUE(topk_v1);
    EXPECT_EQ(topk_v1->get_axis(), axis);
    EXPECT_EQ(topk_v1->get_mode(), op::v1::TopK::Mode::MAX);
    EXPECT_EQ(topk_v1->get_sort_type(), op::v1::TopK::SortType::SORT_VALUES);

    const auto values_out_element_type = topk_v1->get_output_element_type(0);
    EXPECT_EQ(values_out_element_type, data->get_element_type());
}

TEST(opset_transform, opset1_topk_downgrade_pass)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{5, 10, 15});
    const int32_t k = 10;
    const auto k_node = op::Constant::create(element::i64, Shape{}, {k});
    const size_t axis = 2;
    const auto mode = op::v1::TopK::Mode::MAX;
    const auto sort = op::v1::TopK::SortType::SORT_INDICES;
    const auto elem_type = element::i64;

    const auto topk_v1 = make_shared<op::v1::TopK>(data, k_node, axis, mode, sort, elem_type);
    const auto result = make_shared<op::Result>(topk_v1->output(0));
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto topk_v0 = as_type_ptr<op::v0::TopK>(pass_replacement_node);
    ASSERT_TRUE(topk_v0);
    EXPECT_EQ(topk_v0->get_k(), k);
    EXPECT_EQ(topk_v0->get_top_k_axis(), axis);
    EXPECT_EQ(topk_v0->get_compute_max(), true);
    EXPECT_EQ(topk_v0->get_sort(), op::v0::TopK::SortType::SORT_INDICES);
    EXPECT_EQ(topk_v0->get_index_element_type(), elem_type);
}
