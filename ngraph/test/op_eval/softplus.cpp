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

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/softplus.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, softplus_4D)
{
    auto p = make_shared<op::Parameter>(element::Type_t::f32, Shape{4});
    auto softplus = make_shared<op::v4::SoftPlus>(p);
    auto fun = make_shared<Function>(OutputVector{softplus}, ParameterVector{p});

    std::vector<float> inputs{-1.0, 0.0, 1.0, 20.0};
    std::vector<float> expected_result{0.31326166, 0.69314718, 1.3132616, 20.0};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{4}, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(result->get_shape(), Shape{4});
    auto result_data = read_vector<float>(result);
    for (size_t i = 0; i < inputs.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}
