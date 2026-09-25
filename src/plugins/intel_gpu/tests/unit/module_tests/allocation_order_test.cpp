// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "allocation_order.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace cldnn;

namespace {

struct allocation_input {
    layout output_layout;
    size_t unique_id;
};

// Reference comparator from program::get_allocating_order before key caching.
bool reference_less(const std::vector<allocation_input>& inputs, size_t lhs, size_t rhs) {
    auto lhs_layout = inputs[lhs].output_layout;
    auto rhs_layout = inputs[rhs].output_layout;
    if (lhs_layout.is_dynamic() && lhs_layout.has_upper_bound()) {
        lhs_layout.set_tensor(lhs_layout.get_tensor());
    }
    if (rhs_layout.is_dynamic() && rhs_layout.has_upper_bound()) {
        rhs_layout.set_tensor(rhs_layout.get_tensor());
    }
    if (rhs_layout.is_dynamic() && !rhs_layout.has_upper_bound() && lhs_layout.is_dynamic() && !lhs_layout.has_upper_bound()) {
        return lhs < rhs;
    }
    if (rhs_layout.is_dynamic() && !lhs_layout.is_dynamic())
        return true;
    if (lhs_layout.is_dynamic() && !rhs_layout.is_dynamic())
        return false;
    if (lhs_layout.bytes_count() == rhs_layout.bytes_count()) {
        return inputs[lhs].unique_id < inputs[rhs].unique_id;
    }
    return lhs_layout.bytes_count() > rhs_layout.bytes_count();
}

std::vector<size_t> sorted_ids(const std::vector<allocation_input>& inputs) {
    std::vector<std::pair<allocation_order_key, size_t>> keyed;
    for (size_t i = 0; i < inputs.size(); ++i) {
        keyed.emplace_back(allocation_order_key(inputs[i].output_layout, inputs[i].unique_id, i), inputs[i].unique_id);
    }
    std::sort(keyed.begin(), keyed.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    std::vector<size_t> ids;
    for (const auto& entry : keyed) {
        ids.push_back(entry.second);
    }
    return ids;
}

layout plain(ov::PartialShape shape, data_types type = data_types::f32) {
    return layout(shape, type, format::bfyx);
}

}  // namespace

TEST(allocation_order, empty_and_single_node) {
    EXPECT_TRUE(sorted_ids({}).empty());
    EXPECT_EQ(sorted_ids({{plain({1, 2, 3, 4}), 7}}), std::vector<size_t>({7}));
}

TEST(allocation_order, descending_bytes_then_unique_id) {
    const std::vector<allocation_input> inputs = {
        {plain({1, 1, 1, 8}, data_types::f16), 40},
        {plain({1, 1, 1, 8}, data_types::f32), 30},
        {plain({1, 1, 1, 16}, data_types::u8), 20},
        {plain({1, 1, 1, 0}), 10},
    };
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({30, 20, 40, 10}));
}

TEST(allocation_order, bounded_layout_uses_normalized_upper_bound_without_mutating_input) {
    auto bounded = plain({1, 1, ov::Dimension(1, 4), 4});
    auto original = bounded;
    const std::vector<allocation_input> inputs = {
        {plain({1, 1, 1, 8}), 1},
        {bounded, 3},
        {plain({1, 1, 4, 4}), 2},
    };
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({2, 3, 1}));
    EXPECT_EQ(bounded, original);
    EXPECT_TRUE(inputs[1].output_layout.is_dynamic());
    EXPECT_EQ(inputs[1].output_layout.get_partial_shape(), original.get_partial_shape());
}

TEST(allocation_order, unbounded_layouts_follow_processing_order_after_static_and_bounded) {
    const auto unbounded = plain({1, 1, ov::Dimension::dynamic(), 4});
    const std::vector<allocation_input> inputs = {
        {unbounded, 90},
        {plain({1, 1, ov::Dimension(1, 8), 4}), 30},
        {unbounded, 10},
        {plain({1, 1, 2, 4}), 20},
    };
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({30, 20, 90, 10}));
    // Computing a byte size for this layout would throw.
    EXPECT_NO_THROW((allocation_order_key(unbounded, 0, 0)));
}

TEST(allocation_order, blocked_format_and_padding_affect_allocation_size) {
    const layout blocked({1, 17, 1, 1}, data_types::f16, format::b_fs_yx_fsv16);
    const layout padded({1, 1, 2, 2}, data_types::f32, format::bfyx, padding({0, 0, 1, 1}, {0, 0, 1, 1}));
    ASSERT_EQ(blocked.bytes_count(), 64u);
    ASSERT_EQ(padded.bytes_count(), 64u);
    const std::vector<allocation_input> inputs = {
        {plain({1, 24, 1, 1}, data_types::f16), 10},
        {blocked, 30},
        {padded, 20},
    };
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({20, 30, 10}));
}

TEST(allocation_order, custom_format_preserves_descriptor_size) {
    auto traits = format(format::bfyx).traits();
    traits.str = "custom";
    traits.desc_size = 128;
    const layout custom({1, 1, 2, 2}, data_types::f32, format(traits));
    ASSERT_EQ(custom.bytes_count(), 128u);
    const std::vector<allocation_input> inputs = {
        {plain({1, 1, 1, 16}), 1},
        {custom, 2},
    };
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({2, 1}));
}

TEST(allocation_order, subbyte_types_round_up_to_bytes) {
    const std::vector<allocation_input> inputs = {
        {plain({1, 1, 1, 3}, data_types::i4), 20},
        {plain({1, 1, 1, 2}, data_types::u8), 10},
        {plain({1, 1, 1, 2}, data_types::i4), 30},
    };
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({10, 20, 30}));
}

TEST(allocation_order, keys_are_rebuilt_for_changed_layouts_and_processing_order) {
    const auto dynamic = plain({1, 1, ov::Dimension::dynamic(), 1});
    std::vector<allocation_input> inputs = {{dynamic, 9}, {dynamic, 1}};
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({9, 1}));
    std::reverse(inputs.begin(), inputs.end());
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({1, 9}));
    inputs[1].output_layout = plain({1, 1, 1, 16});
    EXPECT_EQ(sorted_ids(inputs), std::vector<size_t>({9, 1}));
}

TEST(allocation_order, randomized_orders_match_reference) {
    std::mt19937 random(20260921);
    for (size_t trial = 0; trial < 12; ++trial) {
        std::vector<allocation_input> inputs;
        for (size_t i = 0; i < 256; ++i) {
            auto shape = ov::PartialShape{1, static_cast<int64_t>(1 + random() % 64), 2, 4};
            if (i % 3 == 0)
                shape[2] = ov::Dimension(1, 8);
            else if (i % 3 == 1)
                shape[2] = ov::Dimension::dynamic();
            auto fmt = i % 2 == 0 ? format::bfyx : format::b_fs_yx_fsv16;
            auto type = i % 2 == 0 ? data_types::f32 : data_types::f16;
            inputs.push_back({layout(shape, type, fmt), i + 1});
        }
        std::shuffle(inputs.begin(), inputs.end(), random);
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
            return reference_less(inputs, lhs, rhs);
        });
        std::vector<size_t> expected;
        for (auto index : indices)
            expected.push_back(inputs[index].unique_id);
        EXPECT_EQ(sorted_ids(inputs), expected) << "trial " << trial;
        EXPECT_EQ(sorted_ids(inputs), expected) << "repeated trial " << trial;
    }
}

TEST(allocation_order, comparator_is_strict_weak_ordering) {
    const std::vector<allocation_input> inputs = {
        {plain({1, 1, 1, 4}), 1},
        {plain({1, 1, 1, 8}), 2},
        {plain({1, 1, ov::Dimension(1, 2), 4}), 3},
        {plain({1, 1, ov::Dimension::dynamic(), 4}), 4},
        {plain({1, 1, ov::Dimension::dynamic(), 4}), 5},
    };
    std::vector<allocation_order_key> keys;
    for (size_t i = 0; i < inputs.size(); ++i)
        keys.emplace_back(inputs[i].output_layout, inputs[i].unique_id, i);
    // Include an equivalent key to test equivalence transitivity, too.
    keys.push_back(keys.front());
    for (const auto& a : keys) {
        EXPECT_FALSE(a < a);
        for (const auto& b : keys) {
            if (a < b) {
                EXPECT_FALSE(b < a);
            }
            for (const auto& c : keys) {
                if (a < b && b < c) {
                    EXPECT_TRUE(a < c);
                }
                if (!(a < b) && !(b < a) && !(b < c) && !(c < b)) {
                    EXPECT_FALSE(a < c);
                    EXPECT_FALSE(c < a);
                }
            }
        }
    }
}
