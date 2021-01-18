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

using namespace std;
using namespace ngraph;

TEST(type_prop, extractimagepatches_i32)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_i64)
{
    auto data = make_shared<op::Parameter>(element::i64, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i64);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_rates_change)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_input_shape_change)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 9, 9});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
}

TEST(type_prop, extractimagepatches_dynamic_shape)
{
    auto data = make_shared<op::Parameter>(element::i32, PartialShape::dynamic(4));
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{2, 2};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_TRUE(
        extractimagepatches->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, extractimagepatches_dynamic_batch_shape)
{
    auto data =
        make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_TRUE(extractimagepatches->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_padding_same_lower1)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_LOWER;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}

TEST(type_prop, extractimagepatches_padding_same_lower2)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 9, 9});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_LOWER;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 2}));
}
TEST(type_prop, extractimagepatches_padding_same_upper)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 11, 11});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_UPPER;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 3, 3}));
}

TEST(type_prop, extractimagepatches_padding_same_upper2)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 6, 11});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_UPPER;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 2, 3}));
}

TEST(type_prop, extractimagepatches_zero_dim_inputs)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 0, 0, 0});
    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 0, 0, 0}));
}

TEST(type_prop, extractimagepatches_large_stride_valid_padding)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{15, 15};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::VALID;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
}

TEST(type_prop, extractimagepatches_large_stride_same_padding)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});
    auto sizes = Shape{3, 3};
    auto strides = Strides{15, 15};
    auto rates = Shape{1, 1};
    auto padtype_padding = op::PadType::SAME_UPPER;
    auto extractimagepatches =
        make_shared<op::v3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);

    EXPECT_EQ(extractimagepatches->get_output_element_type(0), element::i32);
    EXPECT_EQ(extractimagepatches->get_output_shape(0), (Shape{64, 27, 1, 1}));
}
