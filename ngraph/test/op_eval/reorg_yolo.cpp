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

#include "ngraph/op/reorg_yolo.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, reorg_yolo)
{
    // in_shape [N,C,H,W]
    const auto in_shape = Shape{1, 8, 4, 4};
    auto p = make_shared<op::Parameter>(element::f32, in_shape);
    int stride = 2;
    auto reorg_yolo = make_shared<op::v0::ReorgYolo>(p, stride);
    auto fun = make_shared<Function>(OutputVector{reorg_yolo}, ParameterVector{p});

    std::vector<float> inputs(128);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<float> expected_result{
    0, 2, 4, 6, 16, 18, 20, 22, 32, 34, 36, 38, 48, 50, 52, 54, 64, 66, 68, 70, 80, 82, 84, 86, 96, 98, 100, 102, 112, 114, 116, 118, 1, 3, 5, 7, 17, 19, 21, 23, 33, 35, 37, 39, 49, 51, 53, 55, 65, 67, 69, 71, 81, 83, 85, 87, 97, 99, 101, 103, 113, 115, 117, 119, 8, 10, 12, 14, 24, 26, 28, 30, 40, 42, 44, 46, 56, 58, 60, 62, 72, 74, 76, 78, 88, 90, 92, 94, 104, 106, 108, 110, 120, 122, 124, 126, 9, 11, 13, 15, 25, 27, 29, 31, 41, 43, 45, 47, 57, 59, 61, 63, 73, 75, 77, 79, 89, 91, 93, 95, 105, 107, 109, 111, 121, 123, 125, 127
    };

    // in_shape [N,C,H,W] -> out_shape [N, C*stride*stride, H/stride, W/stride]
    Shape expected_shape = Shape{
        in_shape[0], in_shape[1]*stride*stride, in_shape[2]/stride, in_shape[3]/stride};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(
        fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(in_shape, inputs)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < inputs.size(); ++i)
    {
        EXPECT_EQ(result_data[i], expected_result[i]);
    }
}
