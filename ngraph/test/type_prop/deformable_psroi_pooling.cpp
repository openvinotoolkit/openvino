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

TEST(type_prop, deformable_psroi_pooling_output_shape)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1024, 63, 38});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const int64_t output_dim = 882;
    const float spatial_scale = 0.0625;
    const int64_t group_size = 3;

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input, coords, offsets, output_dim, spatial_scale, group_size);

    ASSERT_EQ(def_psroi_pool->get_output_shape(0), (Shape{300, 882, 3, 3}));
}

TEST(type_prop, deformable_psroi_pooling_output_shape_2)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 7938, 38, 38});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const int64_t output_dim = 162;
    const float spatial_scale = 0.0625;
    const int64_t group_size = 7;

    auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
        input, coords, offsets, output_dim, spatial_scale, group_size);

    ASSERT_EQ(def_psroi_pool->get_output_shape(0), (Shape{300, 162, 7, 7}));
}

TEST(type_prop, deformable_psroi_pooling_invalid_input_rank)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const int64_t output_dim = 4;
    const float spatial_scale = 0.9;
    const int64_t group_size = 7;
    try
    {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
            input, coords, offsets, output_dim, spatial_scale, group_size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid feature map input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Feature map input rank must equal to 4 (input rank: 3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_box_coordinates_rank)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const int64_t output_dim = 4;
    const float spatial_scale = 0.9;
    const int64_t group_size = 7;
    try
    {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
            input, coords, offsets, output_dim, spatial_scale, group_size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Ivalid box coordinates input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Box coordinates input rank must equal to 2 (input rank: 3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, deformable_psroi_pooling_invalid_offstes_rank)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4, 5});
    const int64_t output_dim = 4;
    const float spatial_scale = 0.9;
    const int64_t group_size = 7;
    try
    {
        auto def_psroi_pool = make_shared<op::v1::DeformablePSROIPooling>(
            input, coords, offsets, output_dim, spatial_scale, group_size);
        // Should have thrown, so fail if it didn't
        FAIL() << "Offsets input rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Offsets input rank must equal to 4 (input rank: 5)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
