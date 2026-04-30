// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/tensorflow_lite/sparsity_info.hpp"

#include <gtest/gtest.h>

#include "openvino/core/type/element_type.hpp"

using ov::frontend::tensorflow_lite::SparsityInfo;

// Direct unit tests for SparsityInfo::enable() and the disabled-state
// contract used by TensorLitePlace and the TFLite frontend's get_sparsity()
// factory.  See src/frontends/tensorflow_lite/src/utils.cpp.

namespace {

constexpr uint8_t kDummyValues[16] = {0};

}  // namespace

// Default ctor leaves m_disabled = false. enable() is what re-evaluates the
// flag based on whether all four required fields are populated. This is the
// exact bug behind the regression: get_sparsity() used to skip the explicit
// enable() call, leaving m_disabled = false on incomplete metadata.
TEST(SparsityInfoTest, DefaultConstructedNotDisabledUntilEnable) {
    SparsityInfo s;
    EXPECT_FALSE(s.is_disabled()) << "Default-constructed SparsityInfo must report not-disabled before enable() runs.";
    s.enable();
    EXPECT_TRUE(s.is_disabled()) << "After enable() with all fields empty, SparsityInfo must report disabled.";
}

TEST(SparsityInfoTest, EnableDisablesWhenShapeEmpty) {
    SparsityInfo s;
    s.set_traversal_order({0, 1});
    s.set_block_map({0});
    s.set_dim_format({0, 1});
    // shape intentionally left empty
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

TEST(SparsityInfoTest, EnableDisablesWhenTraversalOrderEmpty) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_block_map({0});
    s.set_dim_format({0, 1});
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

TEST(SparsityInfoTest, EnableDisablesWhenBlockMapEmpty) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_dim_format({0, 1});
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

// HandsLandmarkFull-style: tf_sparsity is non-null but dim_metadata is empty,
// so dim_format ends up empty in get_sparsity().
TEST(SparsityInfoTest, EnableDisablesWhenDimFormatEmpty) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_block_map({0});
    // dim_format intentionally left empty
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

TEST(SparsityInfoTest, EnableEnablesWhenAllFourPopulated) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_block_map({0});
    s.set_dim_format({0, 1});
    s.enable();
    EXPECT_FALSE(s.is_disabled());
}

TEST(SparsityInfoTest, FullCtorAlreadyEnabled) {
    SparsityInfo::SparsityDataDesc desc{};
    SparsityInfo s({2, 2}, {0, 1}, {0}, {0, 1}, {desc, desc}, ov::element::f32, kDummyValues, sizeof(kDummyValues));
    EXPECT_FALSE(s.is_disabled()) << "Full ctor must call enable() and find all fields populated.";
}

// The full ctor calls enable() at the end (sparsity_info.hpp:47). With an
// empty dim_format the result must be disabled.
TEST(SparsityInfoTest, FullCtorWithEmptyDimFormatDisables) {
    SparsityInfo s({2, 2},
                   {0, 1},
                   {0},
                   {},  // dim_format empty
                   {},
                   ov::element::f32,
                   kDummyValues,
                   sizeof(kDummyValues));
    EXPECT_TRUE(s.is_disabled());
}

// dense_data() must throw when m_disabled is true. This is the contract that
// TensorLitePlace::ctor relies on via the is_disabled() short-circuit.
TEST(SparsityInfoTest, DenseDataThrowsWhenDisabled) {
    SparsityInfo s;
    s.enable();
    ASSERT_TRUE(s.is_disabled());
    EXPECT_THROW(s.dense_data(), std::exception);
}

// disable()/enable() round-trip: explicit disable() forces disabled; a
// subsequent enable() re-evaluates against the populated fields. Lock-down
// against future regressions that turn enable() into a one-way switch.
TEST(SparsityInfoTest, DisableSetsFlagAndEnableReevaluates) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_block_map({0});
    s.set_dim_format({0, 1});

    s.disable();
    EXPECT_TRUE(s.is_disabled());

    s.enable();
    EXPECT_FALSE(s.is_disabled()) << "enable() must re-evaluate from scratch, not stay disabled.";
}

// is_copyable() is fixed to false for runtime attributes — sparsity is bound
// to the original tensor. Locks the contract used by RuntimeAttribute.
TEST(SparsityInfoTest, IsCopyableReturnsFalse) {
    SparsityInfo s;
    EXPECT_FALSE(s.is_copyable());
}

// Setters and getters round-trip values without side effects. Cheap sanity
// to make sure none of the setters accidentally re-disable the object.
TEST(SparsityInfoTest, SettersRoundTripPreserveValues) {
    SparsityInfo s;
    s.set_shape({4, 8});
    s.set_traversal_order({1, 0});
    s.set_block_map({0});
    s.set_dim_format({0, 1});
    s.set_target_type(ov::element::i8);
    s.set_values(kDummyValues, sizeof(kDummyValues));

    EXPECT_EQ(s.get_shape(), std::vector<int32_t>({4, 8}));
    EXPECT_EQ(s.get_traversal_order(), std::vector<int32_t>({1, 0}));
    EXPECT_EQ(s.get_block_map(), std::vector<int32_t>({0}));
    EXPECT_EQ(s.get_dim_format(), std::vector<uint16_t>({0, 1}));
    EXPECT_EQ(s.get_target_type(), ov::element::i8);
    EXPECT_EQ(s.get_values(), kDummyValues);
    EXPECT_EQ(s.get_values_size(), sizeof(kDummyValues));
}
