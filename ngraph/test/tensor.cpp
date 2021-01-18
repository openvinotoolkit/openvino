//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "ngraph/pass/manager.hpp"
#include "pass/liveness.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(tensor, size)
{
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::Liveness>();

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
        auto add = make_shared<op::v1::Add>(arg0, arg0);
        auto f0 = make_shared<Function>(add, ParameterVector{arg0});

        pass_manager.run_passes(f0);

        ASSERT_EQ(1, arg0->get_output_size());
        descriptor::Tensor& output = arg0->get_output_tensor(0);
        EXPECT_EQ(2 * 3 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{});
        auto add = make_shared<op::v1::Add>(arg0, arg0);
        auto f0 = make_shared<Function>(add, ParameterVector{arg0});

        pass_manager.run_passes(f0);

        ASSERT_EQ(1, arg0->get_output_size());
        descriptor::Tensor& output = arg0->get_output_tensor(0);
        EXPECT_EQ(1 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
        auto add = make_shared<op::v1::Add>(arg0, arg0);
        auto f0 = make_shared<Function>(add, ParameterVector{arg0});

        pass_manager.run_passes(f0);

        ASSERT_EQ(1, arg0->get_output_size());
        descriptor::Tensor& output = arg0->get_output_tensor(0);
        EXPECT_EQ(1 * 4, output.size());
    }
}

TEST(tensor, output_flag)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Liveness>();

    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto add = make_shared<op::v1::Add>(arg0, arg0);
    auto f0 = make_shared<Function>(add, ParameterVector{arg0});

    pass_manager.run_passes(f0);

    for (size_t i = 0; i < f0->get_output_size(); ++i)
    {
        EXPECT_TRUE(op::is_output(f0->get_output_op(i)));
    }
}

TEST(tensor, tensor_names)
{
    auto arg0 = make_shared<opset6::Parameter>(element::f32, Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = make_shared<opset6::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f0 = make_shared<Function>(relu, ParameterVector{arg0});

    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->get_input_tensor(0).get_names());
    ASSERT_EQ(arg0->get_output_tensor(0).get_names(),
              relu->input_value(0).get_tensor().get_names());
    ASSERT_EQ(f0->get_result()->get_input_tensor(0).get_names(),
              relu->get_output_tensor(0).get_names());
    ASSERT_EQ(f0->get_result()->input_value(0).get_tensor().get_names(),
              relu->get_output_tensor(0).get_names());
}
