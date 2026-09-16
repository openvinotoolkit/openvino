// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pa_kv_reorder.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::test {

class TypePropPaKVReorderTest : public TypePropOpTest<op::internal::PaKVReorder> {};

TEST_F(TypePropPaKVReorderTest, static_shapes) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    const auto op = make_op(key_cache,
                            value_cache,
                            block_indices,
                            block_indices_begins,
                            block_update_indices,
                            block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
    EXPECT_EQ(op->get_output_size(), 1);
}

TEST_F(TypePropPaKVReorderTest, dynamic_batch) {
    const auto key_cache =
        std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 8, 64, 16});
    const auto value_cache =
        std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    const auto block_indices_begins =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    const auto block_update_indices =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    const auto block_update_indices_begins =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    const auto op = make_op(key_cache,
                            value_cache,
                            block_indices,
                            block_indices_begins,
                            block_update_indices,
                            block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

TEST_F(TypePropPaKVReorderTest, fully_dynamic_shapes) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto block_update_indices_begins =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    const auto op = make_op(key_cache,
                            value_cache,
                            block_indices,
                            block_indices_begins,
                            block_update_indices,
                            block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

TEST_F(TypePropPaKVReorderTest, dynamic_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    const auto op = make_op(key_cache,
                            value_cache,
                            block_indices,
                            block_indices_begins,
                            block_update_indices,
                            block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

TEST_F(TypePropPaKVReorderTest, invalid_key_cache_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    OV_EXPECT_THROW(std::ignore = make_op(key_cache,
                                          value_cache,
                                          block_indices,
                                          block_indices_begins,
                                          block_update_indices,
                                          block_update_indices_begins),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[0].rank().compatible(4)"));
}

TEST_F(TypePropPaKVReorderTest, invalid_value_cache_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    OV_EXPECT_THROW(std::ignore = make_op(key_cache,
                                          value_cache,
                                          block_indices,
                                          block_indices_begins,
                                          block_update_indices,
                                          block_update_indices_begins),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[1].rank().compatible(4)"));
}

TEST_F(TypePropPaKVReorderTest, invalid_indices_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5, 10});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    OV_EXPECT_THROW(std::ignore = make_op(key_cache,
                                          value_cache,
                                          block_indices,
                                          block_indices_begins,
                                          block_update_indices,
                                          block_update_indices_begins),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[i].rank().compatible(1)"));
}

TEST_F(TypePropPaKVReorderTest, invalid_indices_type) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    OV_EXPECT_THROW(std::ignore = make_op(key_cache,
                                          value_cache,
                                          block_indices,
                                          block_indices_begins,
                                          block_update_indices,
                                          block_update_indices_begins),
                    NodeValidationFailure,
                    testing::HasSubstr("(indices) must be i32"));
}

TEST_F(TypePropPaKVReorderTest, i8_cache) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    const auto op = make_op(key_cache,
                            value_cache,
                            block_indices,
                            block_indices_begins,
                            block_update_indices,
                            block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

}  // namespace ov::test
