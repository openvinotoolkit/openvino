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
constexpr uint8_t kSegmentsSentinel[1] = {0};
constexpr uint8_t kIndicesSentinel[1] = {0};

// Standard-CSR data_desc for a (DENSE, SPARSE_CSR) dim_format. The DENSE
// entry's segments/indices are unused (densify() never indexes them); the
// SPARSE_CSR entry must carry non-null payloads for enable() to admit it.
inline std::vector<SparsityInfo::SparsityDataDesc> make_dense_then_sparse_data_desc() {
    return {
        SparsityInfo::SparsityDataDesc{0, nullptr, 0, nullptr},
        SparsityInfo::SparsityDataDesc{0, kSegmentsSentinel, 0, kIndicesSentinel},
    };
}

}  // namespace

// enable() must disable a SparsityInfo whose required fields are all empty.
// The end-to-end coverage that get_sparsity() actually invokes enable() is in
// convert_sparse_incomplete.cpp; this test pins down the enable() semantics
// in isolation.
TEST(SparsityInfoTest, EnableOnEmptyObjectDisables) {
    SparsityInfo s;
    s.enable();
    EXPECT_TRUE(s.is_disabled()) << "enable() with all required fields empty must disable.";
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

TEST(SparsityInfoTest, EnableDoesNotDisableWhenBlockMapEmpty) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_dim_format({0, 1});
    s.set_data_desc(make_dense_then_sparse_data_desc());
    // block_map is intentionally absent — valid for standard CSR tensors
    s.enable();
    EXPECT_FALSE(s.is_disabled());
}

// Block-sparse layouts (traversal_order longer than shape rank) require
// block_map. With block_map missing the metadata is malformed and enable()
// must disable the tensor so densify() is never called on it.
TEST(SparsityInfoTest, EnableDisablesWhenBlockSparseMissingBlockMap) {
    SparsityInfo s;
    s.set_shape({2, 2});                  // rank = 2
    s.set_traversal_order({0, 1, 2, 3});  // length = 4 → block-sparse with k=2
    s.set_dim_format({0, 1, 1, 1});
    // block_map intentionally absent — invalid for block-sparse
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

// traversal_order shorter than the tensor rank — densify() would later loop
// `for (dim = 0; dim < m_shape.size(); ++dim) m_dim_format[dim]` and read
// past the end of m_dim_format on untrusted model input. enable() must
// catch this size mismatch up front.
TEST(SparsityInfoTest, EnableDisablesWhenTraversalOrderShorterThanRank) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0});  // length 1 < rank 2
    s.set_dim_format({0});
    s.set_data_desc({SparsityInfo::SparsityDataDesc{0, nullptr, 0, nullptr}});
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

// dim_format must carry exactly one DimensionMetadata entry per
// traversal-order position. A length mismatch is malformed; enable() must
// disable so the dispatch in densify() never indexes m_dim_format past its
// real end.
TEST(SparsityInfoTest, EnableDisablesWhenDimFormatLengthMismatchesTraversal) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_dim_format({0});  // length 1 != traversal_order length 2
    s.set_data_desc({SparsityInfo::SparsityDataDesc{0, nullptr, 0, nullptr}});
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

// A SPARSE_CSR DimensionMetadata with absent array_segments or array_indices
// is recorded as a null pointer in m_data_desc. densify()'s dispatcher casts
// those pointers to flatbuffers vector types and calls ->values() without a
// null check, which would dereference null. enable() must disable the
// tensor in that case so the raw-buffer fallback runs instead.
TEST(SparsityInfoTest, EnableDisablesWhenSparseCsrDimMissesSegments) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    s.set_dim_format({0, 1});  // dim 1 is SPARSE_CSR
    s.set_data_desc({
        SparsityInfo::SparsityDataDesc{0, nullptr, 0, nullptr},
        SparsityInfo::SparsityDataDesc{0, nullptr, 0, kIndicesSentinel},  // segments absent
    });
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

// HandsLandmarkFull-style: tf_sparsity is non-null but dim_metadata is empty,
// so dim_format ends up empty in get_sparsity().
TEST(SparsityInfoTest, EnableDisablesWhenDimFormatEmpty) {
    SparsityInfo s;
    s.set_shape({2, 2});
    s.set_traversal_order({0, 1});
    // block_map left empty (consistent with non-block-sparse layout)
    // dim_format intentionally left empty
    s.enable();
    EXPECT_TRUE(s.is_disabled());
}

// Block-sparse positive: traversal_order length = rank + #block dims,
// block_map lists the block dims, dim_format and data_desc match.
TEST(SparsityInfoTest, EnableEnabledOnBlockSparseLayout) {
    SparsityInfo s;
    s.set_shape({2, 2});                  // rank = 2
    s.set_traversal_order({0, 1, 2, 3});  // rank + 2 block dims
    s.set_block_map({0, 1});              // 2 block dims appended after rank
    s.set_dim_format({0, 0, 1, 1});
    s.set_data_desc({
        SparsityInfo::SparsityDataDesc{0, nullptr, 0, nullptr},
        SparsityInfo::SparsityDataDesc{0, nullptr, 0, nullptr},
        SparsityInfo::SparsityDataDesc{0, kSegmentsSentinel, 0, kIndicesSentinel},
        SparsityInfo::SparsityDataDesc{0, kSegmentsSentinel, 0, kIndicesSentinel},
    });
    s.enable();
    EXPECT_FALSE(s.is_disabled());
}

TEST(SparsityInfoTest, FullCtorAlreadyEnabled) {
    SparsityInfo::SparsityDataDesc dense_desc{};
    SparsityInfo::SparsityDataDesc sparse_desc{0, kSegmentsSentinel, 0, kIndicesSentinel};
    SparsityInfo
        s({2, 2}, {0, 1}, {}, {0, 1}, {dense_desc, sparse_desc}, ov::element::f32, kDummyValues, sizeof(kDummyValues));
    EXPECT_FALSE(s.is_disabled()) << "Full ctor must call enable() and find all fields populated.";
}

// The full ctor calls enable() at the end (sparsity_info.hpp). With an
// empty dim_format the result must be disabled.
TEST(SparsityInfoTest, FullCtorWithEmptyDimFormatDisables) {
    SparsityInfo s({2, 2},
                   {0, 1},
                   {},  // block_map empty (non-block-sparse)
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
    s.set_dim_format({0, 1});
    s.set_data_desc(make_dense_then_sparse_data_desc());

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
