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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, crop_and_resize_valid)
{
    Dimension N = 4;
    Dimension W_image = 400;
    Dimension H_image = 300;
    Dimension C_image = 3;
    Dimension num_boxes = 20;
    int32_t W_crop = 30;
    int32_t H_crop = 40;

    PartialShape result_shape{num_boxes, H_crop, W_crop, C_image};

    auto image =
        make_shared<op::Parameter>(element::f32, PartialShape{N, H_image, W_image, C_image});
    auto boxes = make_shared<op::Parameter>(element::f32, PartialShape{num_boxes, 4});
    auto box_indices = make_shared<op::Parameter>(element::i32, PartialShape{num_boxes});
    auto crop_shape = op::Constant::create(element::i32, Shape{2}, {H_crop, W_crop});

    auto crop_and_resize = make_shared<op::CropAndResize>(
        image, boxes, box_indices, crop_shape, op::CropAndResize::ResizeMethod::bilinear, 0);
    auto result = crop_and_resize->output(0);
    ASSERT_EQ(result.get_shape(), result_shape.to_shape());
    ASSERT_EQ(result.get_element_type(), image->get_output_element_type(0));
}

TEST(type_prop, crop_and_resize_not_constant)
{
    Dimension N = 4;
    Dimension W_image = 400;
    Dimension H_image = 300;
    Dimension C_image = 3;
    Dimension num_boxes = 20;
    int32_t W_crop = 30;
    int32_t H_crop = 40;

    PartialShape result_shape{num_boxes, H_crop, W_crop, C_image};

    auto image =
        make_shared<op::Parameter>(element::f32, PartialShape{N, H_image, W_image, C_image});
    auto boxes = make_shared<op::Parameter>(element::f32, PartialShape{num_boxes, 4});
    auto box_indices = make_shared<op::Parameter>(element::i32, PartialShape{num_boxes});
    auto crop_shape = make_shared<op::Parameter>(element::i32, PartialShape{2});

    try
    {
        auto crop_and_resize = make_shared<op::CropAndResize>(
            image, boxes, box_indices, crop_shape, op::CropAndResize::ResizeMethod::bilinear, 0);
        FAIL() << "CropAndReshape without constant crop shape should fail";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("crop_size must be a constant"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
