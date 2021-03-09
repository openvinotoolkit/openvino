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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

TEST(type_prop, dft_constant_axes_and_there_are_no_signal_size_static_shapes)
{
    struct ShapesAndValues
    {
        Shape input_shape;
        Shape axes_shape;
        Shape ref_output_shape;
        std::vector<int64_t> axes;
    };

    std::vector<ShapesAndValues> shapes_and_values = {
        {{2, 180, 180, 2}, {2}, {2, 180, 180, 2}, {1, 2}}
    };

    for (const auto& s : shapes_and_values)
    {
        auto data = std::make_shared<op::Parameter>(element::f32, s.input_shape);
        auto axes_input = op::Constant::create<int64_t>(element::i64, s.axes_shape, s.axes);
        auto dft = std::make_shared<op::v7::DFT>(data, axes_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_shape(), (s.ref_output_shape));
    }
}
