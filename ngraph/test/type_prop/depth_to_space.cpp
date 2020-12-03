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

TEST(type_prop, depth_to_space_output_shape_block_first_4D)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 128, 8, 8});
    auto space_to_depth =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_4D_2)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 12, 1080, 1616});
    auto space_to_depth =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_5D)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 16, 3, 1080, 1616});
    auto space_to_depth =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_depth_first_4D)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 12, 1080, 1616});
    auto space_to_depth =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_depth_first_5D)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 16, 3, 1080, 1616});
    auto space_to_depth =
        make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_input_rank_not_supported)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 8});
    try
    {
        auto space_to_depth =
            make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not supported input shape for DepthToSpace exception not thrown";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "The input tensor with rank lower than 3 is not supported (input rank: 2)");
    }
    catch (...)
    {
        FAIL() << "DepthToSpace decomposition failed for unexpected reason";
    }
}

TEST(type_prop, depth_to_space_blocksize_not_matched)
{
    auto A = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 7, 4, 4});
    try
    {
        auto space_to_depth =
            make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not matched blocksize for DepthToSpace exception not thrown";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "DepthToSpace: The input data's 'channels' axis size: 7"
                             " must be a equivalent to 'block_size'^'spatial_dims': 4");
    }
    catch (...)
    {
        FAIL() << "DepthToSpace decomposition failed for unexpected reason";
    }
}
