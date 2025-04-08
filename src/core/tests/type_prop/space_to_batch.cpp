// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_batch.hpp"

#include <gmock/gmock.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;
using namespace testing;

#define DIV_ROUND_UP(n, d) (((n) + (d)-1) / (d))

TEST(type_prop, space_to_batch_output_shape_2D) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(), (Shape{2 * 5, (128 + 2) / 5}));
}

TEST(type_prop, space_to_batch_output_shape_4D) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 64, 64, 3});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(), (Shape{2 * 10 * 5, (64 + 3 + 3) / 10, (64 + 1) / 5, 3}));
}

TEST(type_prop, space_to_batch_output_shape_5D) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 32, 64, 128, 256});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i32, Shape{5}, vector<int64_t>{1, 6, 5, 1, 16});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i32, Shape{5}, vector<int64_t>{0, 2, 0, 0, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i32, Shape{5}, vector<int64_t>{0, 2, 1, 0, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_shape(), (Shape{2 * 6 * 5 * 16, (32 + 2 + 2) / 6, (64 + 1) / 5, 128, 256 / 16}));
}

TEST(type_prop, space_to_batch_and_batch_to_space) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{2, {100, -1}, 1024, 3});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 12, 100, 2});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 38, 1});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 5, 38, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0),
              (PartialShape{2 * 12 * 100 * 2, {(100 + 3 + 5) / 12, -1}, (1024 + 38 + 38) / 100, (3 + 1) / 2}));

    auto batch_to_space = make_shared<op::v1::BatchToSpace>(space_to_batch, block_shape, pads_begin, pads_end);
    ASSERT_EQ(batch_to_space->get_element_type(), element::f32);
    ASSERT_EQ(batch_to_space->get_output_partial_shape(0), (PartialShape{2, {100, -1}, 1024, 3}));
}

TEST(type_prop, space_to_batch_when_space_is_static) {
    auto data_shape = PartialShape{{2, 5}, 100, 1024, 3};
    set_shape_symbols(data_shape);
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 12, 100, 2});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 38, 1});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 5, 38, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    EXPECT_EQ(
        space_to_batch->get_output_partial_shape(0),
        (PartialShape{{2 * 12 * 100 * 2, 5 * 12 * 100 * 2}, (100 + 3 + 5) / 12, (1024 + 38 + 38) / 100, (3 + 1) / 2}));
    EXPECT_THAT(get_shape_symbols(space_to_batch->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, space_to_batch_when_data_dynamic_) {
    auto data_shape = PartialShape{{2, 5}, {5, 100}, {100, 1024}, {3, 10}};
    auto symbols = set_shape_symbols(data_shape);
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 1, 1, 1});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 0, 2, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 0, 3, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    EXPECT_EQ(space_to_batch->get_output_partial_shape(0),
              PartialShape({{2, 5}, {5, 100}, {(100 + 2 + 3) / 1, (1024 + 2 + 3) / 1}, {3, 10}}));
    EXPECT_THAT(get_shape_symbols(space_to_batch->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, symbols[3]));
}

TEST(type_prop, space_to_batch_when_space_is_dynamic) {
    auto data_shape = PartialShape{{2, 5}, {5, 100}, {100, 1024}, {3, 10}};
    set_shape_symbols(data_shape);
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 12, 100, 2});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 38, 1});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 5, 38, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    EXPECT_EQ(space_to_batch->get_output_partial_shape(0),
              (PartialShape{{2 * 12 * 100 * 2, 5 * 12 * 100 * 2},
                            {DIV_ROUND_UP((5 + 5 + 3), 12), (100 + 5 + 3) / 12},
                            {DIV_ROUND_UP((100 + 38 + 38), 100), (1024 + 38 + 38) / 100},
                            {DIV_ROUND_UP((3 + 1), 2), (10 + 1) / 2}}));
    EXPECT_THAT(get_shape_symbols(space_to_batch->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, space_to_batch_dynamic_shape_static_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(4));
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape({-1, {1, -1}, {1, -1}, -1}));
}

TEST(type_prop, space_to_batch_dynamic_shape_dynamic_rank) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 10, 5, 1});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, space_to_batch_dynamic_rank_shape_block_and_pads_not_const) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto block_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto pads_begin = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto pads_end = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    ASSERT_EQ(space_to_batch->get_element_type(), element::f32);
    ASSERT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, space_to_batch_default_ctor) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{2, 5}, 100, {100, -1}, 3});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 2, 4, 1});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 1, 2, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 1, 6, 0});

    auto space_to_batch = make_shared<op::v1::SpaceToBatch>();
    space_to_batch->set_arguments(OutputVector{data, block_shape, pads_begin, pads_end});
    space_to_batch->validate_and_infer_types();

    EXPECT_EQ(space_to_batch->get_input_size(), 4);
    EXPECT_EQ(space_to_batch->get_output_size(), 1);
    EXPECT_EQ(space_to_batch->get_output_element_type(0), element::f32);
    EXPECT_EQ(space_to_batch->get_output_partial_shape(0),
              PartialShape({{2 * 2 * 4, 5 * 2 * 4}, (100 + 2) / 2, {(100 + 2 + 6) / 4, -1}, 3}));
}

TEST(type_prop, space_to_batch_non_const_inputs) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{100, 7, 13, 3});

    auto block_shape = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{4});
    auto pads_begin = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{4});
    auto pads_end = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{4});
    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    EXPECT_EQ(space_to_batch->get_element_type(), element::f32);
    EXPECT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, space_to_batch_block_non_constant_only) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{100, 7, 13, 3});
    auto block_shape = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{4});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 0});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 0, 0});
    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    EXPECT_EQ(space_to_batch->get_element_type(), element::f32);
    EXPECT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape::dynamic(4));
}

TEST(type_prop, space_to_batch_crops_non_constant_only) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{100, 7, 13, 3});

    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{4}, vector<int64_t>{1, 2, 5, 1});
    auto pads_begin = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{4});
    auto pads_end = make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{4});
    auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    EXPECT_EQ(space_to_batch->get_element_type(), element::f32);
    EXPECT_EQ(space_to_batch->get_output_partial_shape(0), PartialShape({1000, -1, -1, -1}));
}

TEST(type_prop, space_to_batch_invalid_element_type_block_shape) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<float>{0, 2});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});

    try {
        auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid f32 element type for block_shape not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "block_shape must be an integral number");
    } catch (...) {
        FAIL() << "Integral element type node validation check failed for unexpected reason";
    }
}

TEST(type_prop, space_to_batch_invalid_element_type_pads_begin) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, vector<float>{0, 2});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});

    try {
        auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid f32 element type for pads_begin not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "pads_begin must be an integral number but got");
    } catch (...) {
        FAIL() << "Integral element type node validation check failed for unexpected reason";
    }
}

TEST(type_prop, space_to_batch_invalid_element_type_pads_end) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i16, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, vector<float>{0, 0});

    try {
        auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid f32 element type for pads_end not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "pads_end must be an integral number but got");
    } catch (...) {
        FAIL() << "Integral element type node validation check failed for unexpected reason";
    }
}

TEST(type_prop, space_to_batch_invalid_value_block_shape) {
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{-1, -5});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<float>{0, 0});

    try {
        auto space_to_batch = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);
        // Input element type is float32
        FAIL() << "Invalid block_shape value not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "block_shape values must be greater than 0");
    } catch (...) {
        FAIL() << "block_shape value node validation check failed for unexpected reason";
    }
}
