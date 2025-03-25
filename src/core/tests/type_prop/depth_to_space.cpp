// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/depth_to_space.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/util/common_util.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, depth_to_space_input_interval_shape_block_first_5D_when_depth_is_static) {
    auto a_shape = PartialShape{{2, 10}, 24, {3, 7}, {423, 3000}, {235, 1345}};
    auto symbols = set_shape_symbols(a_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    EXPECT_EQ(depth_to_space->get_output_element_type(0), element::f32);
    EXPECT_EQ(depth_to_space->get_output_partial_shape(0),
              (PartialShape{{2, 10}, 3, {3 * 2, 7 * 2}, {423 * 2, 3000 * 2}, {235 * 2, 1345 * 2}}));
    EXPECT_THAT(get_shape_symbols(depth_to_space->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr, nullptr));
}

TEST(type_prop, depth_to_space_input_interval_shape_default_block_size) {
    auto a_shape = PartialShape{{2, 10}, 24, {3, 7}, {423, 3000}, {235, 1345}};
    auto symbols = set_shape_symbols(a_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST);

    EXPECT_EQ(depth_to_space->get_output_element_type(0), element::f32);
    EXPECT_EQ(depth_to_space->get_output_partial_shape(0), a_shape);
    EXPECT_THAT(get_shape_symbols(depth_to_space->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, depth_to_space_output_dynamicshape_block_first_5D_when_depth_is_dynamic) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32,
                                                PartialShape{{2, 10}, {81, 82}, {3, 7}, {423, 3000}, {235, 1345}});
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 3);

    ASSERT_EQ(depth_to_space->get_output_partial_shape(0),
              (PartialShape{{2, 10},
                            {ov::util::ceil_div(81, 27), 82 / 27},
                            {3 * 3, 7 * 3},
                            {423 * 3, 3000 * 3},
                            {235 * 3, 1345 * 3}}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_4D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 128, 8, 8});
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 8);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_4D_2) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_shape(), (Shape{1, 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_block_first_5D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 16, 3, 1080, 1616});
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_shape(), (Shape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_depth_first_4D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_shape(), (Shape{1, 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_output_shape_depth_first_5D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 16, 3, 1080, 1616});
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_shape(), (Shape{1, 2, 2 * 3, 2 * 1080, 2 * 1616}));
}

TEST(type_prop, depth_to_space_input_rank_not_supported) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 8});
    try {
        auto depth_to_space =
            make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not supported input shape for DepthToSpace exception not thrown";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input tensor with rank lower than 3 is not supported (input rank: 2)");
    } catch (...) {
        FAIL() << "DepthToSpace decomposition failed for unexpected reason";
    }
}

TEST(type_prop, depth_to_space_blocksize_not_matched) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 7, 4, 4});
    try {
        auto depth_to_space =
            make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not matched blocksize for DepthToSpace exception not thrown";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dimension value: [ 7, 7] must be a multiple of divisor: 4");
    } catch (...) {
        FAIL() << "DepthToSpace decomposition failed for unexpected reason";
    }
}

TEST(type_prop, depth_to_space_dynamic_shape_static_rank) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto depth_to_space =
        make_shared<ov::op::v0::DepthToSpace>(A, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, depth_to_space_dynamic_shape_dynamic_rank) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto depth_to_space = make_shared<ov::op::v0::DepthToSpace>(A, "depth_first", 2);

    ASSERT_EQ(depth_to_space->get_element_type(), element::f32);
    ASSERT_EQ(depth_to_space->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, depth_to_space_default_ctor) {
    const auto a_shape = PartialShape{{2, 10}, 27, {0, 54}, {9, -1}};
    const auto A = make_shared<ov::op::v0::Parameter>(element::u32, a_shape);

    const auto depth_to_space = make_shared<ov::op::v0::DepthToSpace>();
    depth_to_space->set_block_size(3);
    depth_to_space->set_mode(ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST);
    depth_to_space->set_argument(0, A);
    depth_to_space->validate_and_infer_types();

    EXPECT_EQ(depth_to_space->get_block_size(), 3);
    EXPECT_EQ(depth_to_space->get_mode(), ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST);
    EXPECT_EQ(depth_to_space->get_input_size(), 1);
    EXPECT_EQ(depth_to_space->get_output_size(), 1);
    EXPECT_EQ(depth_to_space->get_output_element_type(0), element::u32);
    EXPECT_EQ(depth_to_space->get_output_partial_shape(0), (PartialShape{{2, 10}, 3, {0 * 3, 54 * 3}, {9 * 3, -1}}));
}
