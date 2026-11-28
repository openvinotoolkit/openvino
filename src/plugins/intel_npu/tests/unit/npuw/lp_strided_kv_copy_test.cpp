// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "infer_request_utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace ov::test::npuw {

namespace {

namespace uu = ov::npuw::util;

constexpr uint8_t k_untouched = 0xAA;

// KV-cache like tensor [N, H, S, E] living in a buffer whose rows are padded to twice their packed size,
// i.e. the same layout an ROI of a larger zero-copy allocation has. With head_gap, heads are padded as well.
struct PaddedTensor {
    PaddedTensor(const ov::element::Type& type, const ov::Shape& shape, bool pattern, bool head_gap = false)
        : shape(shape),
          row_bytes(ov::util::get_memory_size(type, shape.back())),
          strides{shape[1] * shape[2] * 2 * row_bytes * (head_gap ? 2 : 1),
                  shape[2] * 2 * row_bytes * (head_gap ? 2 : 1),
                  2 * row_bytes,
                  type.size()} {
        bytes.resize(shape[0] * strides[0]);
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = pattern ? static_cast<uint8_t>(i * 37 + 11) : k_untouched;
        }
        tensor = ov::SoPtr<ov::ITensor>(ov::make_tensor(type, shape, bytes.data(), strides), {});
    }

    size_t row_offset(size_t h, size_t row) const {
        return h * strides[1] + row * strides[2];
    }

    ov::Shape shape;
    size_t row_bytes;
    ov::Strides strides;
    std::vector<uint8_t> bytes;
    ov::SoPtr<ov::ITensor> tensor;
};

class LpStridedKvCopyTest : public testing::TestWithParam<ov::element::Type> {
protected:
    static constexpr size_t H = 2;
    static constexpr size_t E = 16;
    static constexpr size_t S_MAX = 8;
};

// kv_dim == 2: [1, H, S, E], sequence axis is not the packed one
TEST_P(LpStridedKvCopyTest, PaddedSequenceSliceCopyMatchesReference) {
    const auto type = GetParam();
    PaddedTensor src(type, {1, H, S_MAX, E}, true);
    PaddedTensor dst(type, {1, H, S_MAX, E}, false);

    constexpr size_t src_from = 1, dst_from = 4, len = 3;
    auto src_view = uu::view(src.tensor, 2, src_from, len);
    auto dst_view = uu::view(dst.tensor, 2, dst_from, len);
    ASSERT_EQ(src_view->get_shape(), (ov::Shape{1, H, len, E}));
    ASSERT_FALSE(src_view->is_continuous());

    auto expected = dst.bytes;
    uu::copy_tensor_by_dim(src_view, dst_view, 2, 2);

    for (size_t h = 0; h < H; ++h) {
        for (size_t r = 0; r < len; ++r) {
            std::copy_n(src.bytes.begin() + src.row_offset(h, src_from + r),
                        src.row_bytes,
                        expected.begin() + dst.row_offset(h, dst_from + r));
        }
    }
    EXPECT_EQ(dst.bytes, expected);
}

// kv_dim == 3: [1, H, E, S], sequence axis is the packed (innermost) one
TEST_P(LpStridedKvCopyTest, PaddedInnermostSliceCopyMatchesReference) {
    const auto type = GetParam();
    PaddedTensor src(type, {1, H, E, S_MAX * 2}, true);
    PaddedTensor dst(type, {1, H, E, S_MAX * 2}, false);

    // Offsets are multiples of 8 elements, so they sit on a byte boundary for every sub-byte type
    constexpr size_t src_from = 8, dst_from = 0, len = 8;
    auto src_view = uu::view(src.tensor, 3, src_from, len);
    auto dst_view = uu::view(dst.tensor, 3, dst_from, len);
    ASSERT_EQ(src_view->get_shape(), (ov::Shape{1, H, E, len}));

    auto expected = dst.bytes;
    uu::copy_tensor_by_dim(src_view, dst_view, 3, 3);

    const auto chunk = ov::util::get_memory_size(type, len);
    for (size_t h = 0; h < H; ++h) {
        for (size_t e = 0; e < E; ++e) {
            std::copy_n(src.bytes.begin() + src.row_offset(h, e) + ov::util::get_memory_size(type, src_from),
                        chunk,
                        expected.begin() + dst.row_offset(h, e) + ov::util::get_memory_size(type, dst_from));
        }
    }
    EXPECT_EQ(dst.bytes, expected);
}

// Padded source rows copied into a densely packed destination must not spill padding into it
TEST_P(LpStridedKvCopyTest, PaddedSourceToDenseDestinationMatchesReference) {
    const auto type = GetParam();
    PaddedTensor src(type, {1, H, S_MAX, E}, true);
    constexpr size_t src_from = 2, len = 4;
    auto src_view = uu::view(src.tensor, 2, src_from, len);

    const ov::Shape dst_shape{1, H, len, E};
    std::vector<uint8_t> dst_bytes(H * len * src.row_bytes, k_untouched);
    auto dst = ov::SoPtr<ov::ITensor>(ov::make_tensor(type, dst_shape, dst_bytes.data()), {});

    uu::copy_tensor_by_dim(src_view, dst, 2, 2);

    std::vector<uint8_t> expected(dst_bytes.size(), k_untouched);
    for (size_t h = 0; h < H; ++h) {
        for (size_t r = 0; r < len; ++r) {
            std::copy_n(src.bytes.begin() + src.row_offset(h, src_from + r),
                        src.row_bytes,
                        expected.begin() + (h * len + r) * src.row_bytes);
        }
    }
    EXPECT_EQ(dst_bytes, expected);
}

TEST_P(LpStridedKvCopyTest, HeadStrideGapCopyMatchesReference) {
    const auto type = GetParam();
    constexpr size_t rows = 2;
    PaddedTensor src(type, {1, H, rows, E}, true, true);

    std::vector<uint8_t> dst_bytes(H * rows * src.row_bytes, k_untouched);
    auto dst = ov::SoPtr<ov::ITensor>(ov::make_tensor(type, {1, H, rows, E}, dst_bytes.data()), {});

    uu::copy_tensor_by_dim(src.tensor, dst, 3, 3);

    std::vector<uint8_t> expected(dst_bytes.size(), k_untouched);
    for (size_t h = 0; h < H; ++h) {
        for (size_t r = 0; r < rows; ++r) {
            std::copy_n(src.bytes.begin() + src.row_offset(h, r),
                        src.row_bytes,
                        expected.begin() + (h * rows + r) * src.row_bytes);
        }
    }
    EXPECT_EQ(dst_bytes, expected);
}

// The batch stride is honored as well
TEST_P(LpStridedKvCopyTest, BatchStrideCopyMatchesReference) {
    const auto type = GetParam();
    constexpr size_t batch = 2, rows = 2;
    PaddedTensor src(type, {batch, H, rows, E}, true);

    std::vector<uint8_t> dst_bytes(batch * H * rows * src.row_bytes, k_untouched);
    auto dst = ov::SoPtr<ov::ITensor>(ov::make_tensor(type, {batch, H, rows, E}, dst_bytes.data()), {});

    uu::copy_tensor_by_dim(src.tensor, dst, 3, 3);

    std::vector<uint8_t> expected(dst_bytes.size(), k_untouched);
    for (size_t n = 0; n < batch; ++n) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t r = 0; r < rows; ++r) {
                std::copy_n(src.bytes.begin() + n * src.strides[0] + src.row_offset(h, r),
                            src.row_bytes,
                            expected.begin() + ((n * H + h) * rows + r) * src.row_bytes);
            }
        }
    }
    EXPECT_EQ(dst_bytes, expected);
}

TEST_P(LpStridedKvCopyTest, ViewOffsetsFollowBitWidth) {
    const auto type = GetParam();
    PaddedTensor t(type, {1, H, E, S_MAX * 2}, true);
    const auto* base = static_cast<const uint8_t*>(t.tensor->data());

    auto seq_view = uu::view(t.tensor, 2, 3, 2);
    EXPECT_EQ(static_cast<const uint8_t*>(seq_view->data()), base + 3 * t.strides[2]);
    EXPECT_EQ(seq_view->get_strides(), t.tensor->get_strides());

    auto inner_view = uu::view(t.tensor, 3, 8, 8);
    EXPECT_EQ(static_cast<const uint8_t*>(inner_view->data()), base + ov::util::get_memory_size(type, 8));
    EXPECT_EQ(inner_view->get_strides(), t.tensor->get_strides());
}

INSTANTIATE_TEST_SUITE_P(NpuwLpKv,
                         LpStridedKvCopyTest,
                         testing::Values(ov::element::u1,
                                         ov::element::u2,
                                         ov::element::u4,
                                         ov::element::i4,
                                         ov::element::nf4,
                                         ov::element::f4e2m1,
                                         ov::element::f16),
                         [](const testing::TestParamInfo<ov::element::Type>& info) {
                             return info.param.get_type_name();
                         });

void expect_non_dense_innermost_copy(uint32_t kv_dim) {
    const ov::Shape shape{1, 2, 2, 4};
    const ov::Strides strides{64, 32, 16, 4};
    std::vector<uint8_t> src_bytes(128);
    for (size_t i = 0; i < src_bytes.size(); ++i) {
        src_bytes[i] = static_cast<uint8_t>(i * 37 + 11);
    }
    std::vector<uint8_t> dst_bytes(src_bytes.size(), k_untouched);
    auto src = ov::SoPtr<ov::ITensor>(ov::make_tensor(ov::element::f16, shape, src_bytes.data(), strides), {});
    auto dst = ov::SoPtr<ov::ITensor>(ov::make_tensor(ov::element::f16, shape, dst_bytes.data(), strides), {});

    uu::copy_tensor_by_dim(src, dst, kv_dim, kv_dim);

    std::vector<uint8_t> expected(dst_bytes.size(), k_untouched);
    for (size_t h = 0; h < 2; ++h) {
        for (size_t r = 0; r < 2; ++r) {
            for (size_t e = 0; e < 4; ++e) {
                const size_t offset = h * 32 + r * 16 + e * 4;
                std::copy_n(src_bytes.begin() + offset, 2, expected.begin() + offset);
            }
        }
    }
    EXPECT_EQ(dst_bytes, expected);
}

TEST(LpStridedKvCopyGuardsTest, CopyByPlanesKeepsElementsOfNonDenseInnermostStride) {
    expect_non_dense_innermost_copy(2);
}

TEST(LpStridedKvCopyGuardsTest, CopyColumnsKeepsElementsOfNonDenseInnermostStride) {
    expect_non_dense_innermost_copy(3);
}

TEST(LpStridedKvCopyGuardsTest, UnalignedSubByteInnermostViewThrows) {
    PaddedTensor t(ov::element::u4, {1, 2, 4, 16}, true);
    EXPECT_ANY_THROW(uu::view(t.tensor, 3, 1, 8));  // starts mid-byte
    EXPECT_ANY_THROW(uu::view(t.tensor, 3, 0, 3));  // ends mid-byte
}

TEST(LpStridedKvCopyGuardsTest, CopyByPlanesRejectsDestinationWithFewerRows) {
    PaddedTensor src(ov::element::u4, {1, 2, 4, 16}, true);
    PaddedTensor dst(ov::element::u4, {1, 2, 2, 16}, false);
    EXPECT_ANY_THROW(uu::copy_by_planes(src.tensor, dst.tensor));
}

TEST(LpStridedKvCopyGuardsTest, SubBytePermuteIsRejectedInsteadOfCorrupting) {
    PaddedTensor src(ov::element::u4, {1, 2, 4, 4}, true);
    PaddedTensor dst(ov::element::u4, {1, 2, 4, 4}, false);
    EXPECT_ANY_THROW(uu::copy_tensor_by_dim(src.tensor, dst.tensor, 2, 3));
    EXPECT_EQ(dst.bytes, PaddedTensor(ov::element::u4, {1, 2, 4, 4}, false).bytes);
}

}  // namespace

}  // namespace ov::test::npuw
