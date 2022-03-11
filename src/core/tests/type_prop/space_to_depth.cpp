// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

TEST(type_prop, space_to_depth_output_shape_block_first_4D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 128, 8, 8}));
}

TEST(type_prop, space_to_depth_output_shape_block_first_4D_2) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 12 * 4, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_depth_first_4D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 12 * 4, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_depth_first_5D) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 12, 4, 1080, 1616});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 12 * 8, 4 / 2, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_when_space_is_static) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{{1, 4}, {12, 36}, 1080, 1616});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0),
              (PartialShape{{1, 4}, {12 * 4, 36 * 4}, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_when_space_is_dynamic) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{{1, 4}, {12, 36}, {100, 1081}, {99, 1616}});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(
        space_to_depth->get_output_partial_shape(0),
        (PartialShape{{1, 4}, {12 * 4, 36 * 4}, {DIV_ROUND_UP(100, 2), 1081 / 2}, {DIV_ROUND_UP(99, 2), 1616 / 2}}));
}

TEST(type_prop, space_to_depth_dynamic_shape_static_rank) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, space_to_depth_dynamic_shape_dynamic_rank) {
    auto A = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, space_to_depth_input_rank_not_supported) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 8});
    try {
        auto space_to_depth = make_shared<op::SpaceToDepth>(A, op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        FAIL() << "Not supported input shape for SpaceToDepth exception not thrown";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input tensor with rank lower than 3 is not supported (input rank: 2)");
    } catch (...) {
        FAIL() << "SpaceToDepth decomposition failed for unexpected reason";
    }
}

TEST(type_prop, space_to_depth_blocksize_not_matched) {
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 3, 8, 7});
    try {
        auto space_to_depth = make_shared<op::SpaceToDepth>(A, op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 4);
        FAIL() << "Not matched blocksize SpaceToDepth exception not thrown";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dimension value: [ 7, 7] must be a multiple of divisor: 4");
    } catch (...) {
        FAIL() << "SpaceToDepth decomposition failed for unexpected reason";
    }
}
