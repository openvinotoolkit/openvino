// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

TEST(type_prop, space_to_depth_output_shape_block_first_4D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 64, 64});
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 128, 8, 8}));
}

TEST(type_prop, space_to_depth_output_shape_block_first_4D_2) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 12 * 4, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_depth_first_4D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 1080, 1616});
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 12 * 4, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_depth_first_5D) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 4, 1080, 1616});
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 12 * 8, 4 / 2, 1080 / 2, 1616 / 2}));
}

TEST(type_prop, space_to_depth_output_shape_depth_first_5D_1) {
    auto a_shape = PartialShape{{1, 4}, {12, 36}, 1080, 1616};
    auto symbols = set_shape_symbols(a_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 1);

    EXPECT_EQ(space_to_depth->get_element_type(), element::f32);
    EXPECT_EQ(space_to_depth->get_output_partial_shape(0), a_shape);
    EXPECT_THAT(get_shape_symbols(space_to_depth->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, space_to_depth_output_shape_when_space_is_static) {
    auto a_shape = PartialShape{{1, 4}, {12, 36}, 1080, 1616};
    auto symbols = set_shape_symbols(a_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);

    EXPECT_EQ(space_to_depth->get_element_type(), element::f32);
    EXPECT_EQ(space_to_depth->get_output_partial_shape(0),
              (PartialShape{{1, 4}, {12 * 4, 36 * 4}, 1080 / 2, 1616 / 2}));
    EXPECT_THAT(get_shape_symbols(space_to_depth->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST(type_prop, space_to_depth_output_shape_when_space_is_dynamic) {
    auto a_shape = PartialShape{{1, 4}, {12, 36}, {100, 1081}, {99, 1616}};
    auto symbols = set_shape_symbols(a_shape);
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, a_shape);
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 2);

    EXPECT_EQ(space_to_depth->get_element_type(), element::f32);
    EXPECT_EQ(
        space_to_depth->get_output_partial_shape(0),
        (PartialShape{{1, 4}, {12 * 4, 36 * 4}, {DIV_ROUND_UP(100, 2), 1081 / 2}, {DIV_ROUND_UP(99, 2), 1616 / 2}}));
    EXPECT_THAT(get_shape_symbols(space_to_depth->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST(type_prop, space_to_depth_dynamic_shape_static_rank) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, space_to_depth_dynamic_shape_dynamic_rank) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::v0::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, space_to_depth_default_ctor) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f64, PartialShape{{1, 4}, {12, 36}, 900, 3});

    const auto space_to_depth = make_shared<op::v0::SpaceToDepth>();
    space_to_depth->set_block_size(3);
    space_to_depth->set_mode(op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST);
    space_to_depth->set_argument(0, A);
    space_to_depth->validate_and_infer_types();

    EXPECT_EQ(space_to_depth->get_block_size(), 3);
    EXPECT_EQ(space_to_depth->get_mode(), op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST);
    EXPECT_EQ(space_to_depth->get_input_size(), 1);
    EXPECT_EQ(space_to_depth->get_output_size(), 1);
    EXPECT_EQ(space_to_depth->get_element_type(), element::f64);
    EXPECT_EQ(space_to_depth->get_output_partial_shape(0), (PartialShape{{1, 4}, {12 * 9, 36 * 9}, 900 / 3, 3 / 3}));
}

TEST(type_prop, space_to_depth_input_rank_not_supported) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 8});
    try {
        auto space_to_depth =
            make_shared<op::v0::SpaceToDepth>(A, op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 2);
        FAIL() << "Not supported input shape for SpaceToDepth exception not thrown";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input tensor with rank lower than 3 is not supported (input rank: 2)");
    } catch (...) {
        FAIL() << "SpaceToDepth decomposition failed for unexpected reason";
    }
}

TEST(type_prop, space_to_depth_blocksize_not_matched) {
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 8, 7});
    try {
        auto space_to_depth =
            make_shared<op::v0::SpaceToDepth>(A, op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, 4);
        FAIL() << "Not matched blocksize SpaceToDepth exception not thrown";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Dimension value: [ 7, 7] must be a multiple of divisor: 4");
    } catch (...) {
        FAIL() << "SpaceToDepth decomposition failed for unexpected reason";
    }
}
