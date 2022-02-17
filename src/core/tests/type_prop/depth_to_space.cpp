// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

TEST(type_prop, depth_to_space_output_dynamicshape_block_first_5D_when_depth_is_static) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{{2, 10}, 24, {3, 7}, {423, 3000}, {235, 1345}});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_output_partial_shape(0),
              (PartialShape{{2, 10}, 3, {3 * 2, 7 * 2}, {423 * 2, 3000 * 2}, {235 * 2, 1345 * 2}}));
}

TEST(type_prop, depth_to_space_output_dynamicshape_block_first_5D_when_depth_is_dynamic) {
    auto A =
        make_shared<op::Parameter>(element::f32, PartialShape{{2, 10}, {81, 82}, {3, 7}, {423, 3000}, {235, 1345}});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 3);

    ASSERT_EQ(space_to_depth->get_output_partial_shape(0),
              (PartialShape{{2, 10},
                            {DIV_ROUND_UP(81, 27), 82 / 27},
                            {3 * 3, 7 * 3},
                            {423 * 3, 3000 * 3},
                            {235 * 3, 1345 * 3}}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_4D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 128, 8, 8});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_4D_2) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_5D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 3, 1080, 1616});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_depth_first_4D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_depth_first_5D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 16, 3, 1080, 1616});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_input_rank_not_supported) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 8});
    try {
        auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not supported input shape for DepthToSpace exception not thrown";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input tensor with rank lower than 3 is not supported (input rank: 2)");
    } catch (...) {
        FAIL() << "DepthToSpace decomposition failed for unexpected reason";
    }
}

TEST(type_prop, depth_to_space_blocksize_not_matched) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 7, 4, 4});
    try {
        auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not matched blocksize for DepthToSpace exception not thrown";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dimension value: [ 7, 7] must be a multiple of divisor: 4");
    } catch (...) {
        FAIL() << "DepthToSpace decomposition failed for unexpected reason";
    }
}

TEST(type_prop, depth_to_space_dynamic_shape_static_rank) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, depth_to_space_dynamic_shape_dynamic_rank) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto space_to_depth = make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0), PartialShape::dynamic());
}
