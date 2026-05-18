// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/pa_kv_reorder.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"

namespace ov {
namespace testing {

TEST(type_prop, pa_kv_reorder_static_shapes) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    const auto op = std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
    EXPECT_EQ(op->get_output_size(), 1);
}

TEST(type_prop, pa_kv_reorder_dynamic_batch) {
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

    const auto op = std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

TEST(type_prop, pa_kv_reorder_fully_dynamic_shapes) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto block_update_indices_begins =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    const auto op = std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

TEST(type_prop, pa_kv_reorder_dynamic_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    const auto op = std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

TEST(type_prop, pa_kv_reorder_invalid_key_cache_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    EXPECT_THROW(std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                             value_cache,
                                                             block_indices,
                                                             block_indices_begins,
                                                             block_update_indices,
                                                             block_update_indices_begins),
                 ov::NodeValidationFailure);
}

TEST(type_prop, pa_kv_reorder_invalid_value_cache_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    EXPECT_THROW(std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                             value_cache,
                                                             block_indices,
                                                             block_indices_begins,
                                                             block_update_indices,
                                                             block_update_indices_begins),
                 ov::NodeValidationFailure);
}

TEST(type_prop, pa_kv_reorder_invalid_indices_rank) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5, 10});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    EXPECT_THROW(std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                             value_cache,
                                                             block_indices,
                                                             block_indices_begins,
                                                             block_update_indices,
                                                             block_update_indices_begins),
                 ov::NodeValidationFailure);
}

TEST(type_prop, pa_kv_reorder_invalid_indices_type) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    EXPECT_THROW(std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                             value_cache,
                                                             block_indices,
                                                             block_indices_begins,
                                                             block_update_indices,
                                                             block_update_indices_begins),
                 ov::NodeValidationFailure);
}

TEST(type_prop, pa_kv_reorder_i8_cache) {
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{100, 8, 64, 16});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{100, 8, 16, 64});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});
    const auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{50});
    const auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{4});

    const auto op = std::make_shared<op::internal::PaKVReorder>(key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins);

    EXPECT_EQ(op->get_output_element_type(0), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1}));
}

}  // namespace testing
}  // namespace ov
