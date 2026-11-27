// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_longrope_kv.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "openvino/runtime/make_tensor.hpp"

namespace {

namespace lr = ov::npuw::longrope;

constexpr size_t kInvFreq = 4;                   // planes per rotate_half row
constexpr size_t kRotaryNdims = kInvFreq * 2;    // rotated channels
constexpr size_t kHeadDim = kRotaryNdims + 2;    // two pass-through channels on top
constexpr size_t kHeads = 2;
constexpr size_t kSeqLen = 16;
constexpr uint32_t kSeqDim = 2;  // [batch, heads, seq, head_dim]

ov::npuw::patterns::pre_compute::LongRopeCosSin make_tables(bool has_long) {
    ov::npuw::patterns::pre_compute::LongRopeCosSin tables;
    tables.max_len = kSeqLen;
    tables.rotary_ndims = kRotaryNdims;
    tables.has_long = has_long;
    tables.inv_freq_short = {1.0f, 0.31f, 0.09f, 0.027f};
    tables.inv_freq_long = has_long ? std::vector<float>{0.7f, 0.2f, 0.05f, 0.013f} : tables.inv_freq_short;
    tables.rebuild_tables();
    return tables;
}

// A deterministic, non-degenerate raw (pre-RoPE) key cache.
std::vector<float> make_raw_keys() {
    std::vector<float> raw(kHeads * kSeqLen * kHeadDim);
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = std::sin(static_cast<float>(i) * 0.37f) * 1.5f + 0.25f;
    }
    return raw;
}

// Rotates raw keys the way the graph does: rotate_half over the rotary channels only,
// with the f16 coefficients of the requested mode.
std::vector<float> rotate(const std::vector<float>& raw,
                          ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                          bool is_long) {
    auto cos = tables.cos_rows(tables.max_len, is_long);
    auto sin = tables.sin_rows(tables.max_len, is_long);
    const auto* pcos = cos.data<ov::float16>();
    const auto* psin = sin.data<ov::float16>();

    std::vector<float> out = raw;
    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t p = 0; p < kSeqLen; ++p) {
            float* row = out.data() + (h * kSeqLen + p) * kHeadDim;
            const float* src = raw.data() + (h * kSeqLen + p) * kHeadDim;
            for (size_t j = 0; j < kInvFreq; ++j) {
                const float c = static_cast<float>(pcos[p * kRotaryNdims + j]);
                const float s = static_cast<float>(psin[p * kRotaryNdims + j]);
                row[j] = src[j] * c - src[j + kInvFreq] * s;
                row[j + kInvFreq] = src[j + kInvFreq] * c + src[j] * s;
            }
        }
    }
    return out;
}

ov::SoPtr<ov::ITensor> as_tensor(std::vector<float>& data) {
    return ov::get_tensor_impl(
        ov::Tensor(ov::element::f32, ov::Shape{1, kHeads, kSeqLen, kHeadDim}, data.data()));
}

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected, float tol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tol) << "at element " << i;
    }
}

}  // anonymous namespace

// Re-rotating a short-mode cache must land on the same keys the graph would have
// produced had it rotated the raw keys with the long factors from the start.
TEST(LongRopeRerotate, ShortCacheBecomesLongCache) {
    auto tables = make_tables(true);
    const auto raw = make_raw_keys();
    auto cache = rotate(raw, tables, false);
    const auto expected = rotate(raw, tables, true);

    const auto delta = lr::make_mode_delta(tables, 0, kSeqLen, true);
    ASSERT_EQ(delta.half, kInvFreq);

    auto tensor = as_tensor(cache);
    lr::rerotate_keys(tensor, kSeqDim, kSeqLen, delta);

    expect_close(cache, expected, 2e-3f);
}

// The pass-through channels of a partial-rotary model were never rotated.
TEST(LongRopeRerotate, PassThroughChannelsUntouched) {
    auto tables = make_tables(true);
    const auto raw = make_raw_keys();
    auto cache = rotate(raw, tables, false);
    const auto before = cache;

    auto tensor = as_tensor(cache);
    lr::rerotate_keys(tensor, kSeqDim, kSeqLen, lr::make_mode_delta(tables, 0, kSeqLen, true));

    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t p = 0; p < kSeqLen; ++p) {
            for (size_t d = kRotaryNdims; d < kHeadDim; ++d) {
                const size_t idx = (h * kSeqLen + p) * kHeadDim + d;
                EXPECT_FLOAT_EQ(cache[idx], before[idx]) << "at head " << h << " pos " << p << " ch " << d;
            }
        }
    }
}

// Rows beyond the cached token count belong to no position and must stay as they are.
TEST(LongRopeRerotate, RowsBeyondCachedTokensUntouched) {
    constexpr uint32_t kCached = 5;
    auto tables = make_tables(true);
    const auto raw = make_raw_keys();
    auto cache = rotate(raw, tables, false);
    const auto before = cache;

    auto tensor = as_tensor(cache);
    lr::rerotate_keys(tensor, kSeqDim, kCached, lr::make_mode_delta(tables, 0, kCached, true));

    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t p = kCached; p < kSeqLen; ++p) {
            for (size_t d = 0; d < kHeadDim; ++d) {
                const size_t idx = (h * kSeqLen + p) * kHeadDim + d;
                EXPECT_FLOAT_EQ(cache[idx], before[idx]) << "at head " << h << " pos " << p;
            }
        }
    }
}

// Going long and back must return the cache to where it started.
TEST(LongRopeRerotate, RoundTripRestoresTheCache) {
    auto tables = make_tables(true);
    const auto raw = make_raw_keys();
    auto cache = rotate(raw, tables, false);
    const auto before = cache;

    auto tensor = as_tensor(cache);
    lr::rerotate_keys(tensor, kSeqDim, kSeqLen, lr::make_mode_delta(tables, 0, kSeqLen, true));
    lr::rerotate_keys(tensor, kSeqDim, kSeqLen, lr::make_mode_delta(tables, 0, kSeqLen, false));

    expect_close(cache, before, 2e-3f);
}

// Cached rows do not have to start at position zero.
TEST(LongRopeRerotate, DeltaFollowsTheFirstPositionId) {
    constexpr int64_t kFirstPos = 3;
    auto tables = make_tables(true);

    const auto from_zero = lr::make_mode_delta(tables, 0, kSeqLen, true);
    const auto shifted = lr::make_mode_delta(tables, kFirstPos, kSeqLen - kFirstPos, true);

    ASSERT_EQ(shifted.cos.size(), (kSeqLen - kFirstPos) * kInvFreq);
    for (size_t i = 0; i < shifted.cos.size(); ++i) {
        EXPECT_FLOAT_EQ(shifted.cos[i], from_zero.cos[kFirstPos * kInvFreq + i]);
        EXPECT_FLOAT_EQ(shifted.sin[i], from_zero.sin[kFirstPos * kInvFreq + i]);
    }
}

// A model whose two factor sets coincide never needs a re-rotation.
TEST(LongRopeRerotate, NoLongModeYieldsEmptyDelta) {
    auto tables = make_tables(false);
    const auto delta = lr::make_mode_delta(tables, 0, kSeqLen, true);
    EXPECT_EQ(delta.half, 0u);

    const auto raw = make_raw_keys();
    auto cache = rotate(raw, tables, false);
    const auto before = cache;

    auto tensor = as_tensor(cache);
    lr::rerotate_keys(tensor, kSeqDim, kSeqLen, delta);
    expect_close(cache, before, 0.0f);
}

// Positions the coefficient tables do not cover must be rejected, not silently wrapped.
TEST(LongRopeRerotate, PositionsOutsideTheTablesThrow) {
    auto tables = make_tables(true);
    EXPECT_THROW(lr::make_mode_delta(tables, 1, kSeqLen, true), ov::Exception);
}

// A quantized cache cannot be turned without dequantizing it, and a half-turned cache is
// worse than none - so it is refused rather than warned about and skipped.
TEST(LongRopeRerotate, UnsupportedElementTypeThrows) {
    auto tables = make_tables(true);
    const auto delta = lr::make_mode_delta(tables, 0, kSeqLen, true);

    std::vector<int8_t> quantized(kHeads * kSeqLen * kHeadDim, 1);
    auto tensor = ov::get_tensor_impl(
        ov::Tensor(ov::element::i8, ov::Shape{1, kHeads, kSeqLen, kHeadDim}, quantized.data()));

    EXPECT_THROW(lr::rerotate_keys(tensor, kSeqDim, kSeqLen, delta), ov::Exception);
}

// Rows are addressed from one base pointer, so anything but a canonically packed tensor
// has to be refused. A view that crops only the sequence axis is the trap: its own
// sequence stride still looks dense while every plane after the first sits elsewhere.
TEST(LongRopeRerotate, StridedTensorThrows) {
    auto tables = make_tables(true);
    const auto delta = lr::make_mode_delta(tables, 0, kSeqLen, true);

    ov::Tensor parent(ov::element::f32, ov::Shape{1, kHeads, kSeqLen * 2, kHeadDim});
    std::fill_n(parent.data<float>(), parent.get_size(), 0.5f);
    ov::Tensor cropped(parent,
                       ov::Coordinate{0, 0, 0, 0},
                       ov::Coordinate{1, kHeads, kSeqLen, kHeadDim});
    ASSERT_EQ(cropped.get_shape(), (ov::Shape{1, kHeads, kSeqLen, kHeadDim}));

    auto tensor = ov::get_tensor_impl(cropped);
    EXPECT_THROW(lr::rerotate_keys(tensor, kSeqDim, kSeqLen, delta), ov::Exception);
}

// An f16 cache is rewritten by the SIMD kernel in util_xarch, an f32 one by the scalar
// loop; the two must agree. A rotary width of 20 gives 10 planes per row, which covers
// both the 8-wide vector body and the 2-plane scalar tail of the SIMD path.
TEST(LongRopeRerotate, F16PathMatchesTheF32Path) {
    constexpr size_t kWideInvFreq = 10;
    constexpr size_t kWideRotary = kWideInvFreq * 2;
    constexpr size_t kWideHeadDim = kWideRotary + 4;
    const ov::Shape shape{1, kHeads, kSeqLen, kWideHeadDim};

    ov::npuw::patterns::pre_compute::LongRopeCosSin tables;
    tables.max_len = kSeqLen;
    tables.rotary_ndims = kWideRotary;
    tables.has_long = true;
    for (size_t i = 0; i < kWideInvFreq; ++i) {
        tables.inv_freq_short.push_back(1.0f / std::pow(3.0f, static_cast<float>(i)));
        tables.inv_freq_long.push_back(0.7f / std::pow(3.1f, static_cast<float>(i)));
    }
    tables.rebuild_tables();

    // Both caches start from the very same values, exactly representable in f16.
    std::vector<ov::float16> f16_cache(kHeads * kSeqLen * kWideHeadDim);
    std::vector<float> f32_cache(f16_cache.size());
    for (size_t i = 0; i < f16_cache.size(); ++i) {
        f16_cache[i] = ov::float16(std::sin(static_cast<float>(i) * 0.23f) * 1.5f);
        f32_cache[i] = static_cast<float>(f16_cache[i]);
    }
    const auto before = f16_cache;

    const auto delta = lr::make_mode_delta(tables, 0, kSeqLen, true);
    ASSERT_EQ(delta.half, kWideInvFreq);

    auto f16_tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f16, shape, f16_cache.data()));
    auto f32_tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape, f32_cache.data()));
    lr::rerotate_keys(f16_tensor, kSeqDim, kSeqLen, delta);
    lr::rerotate_keys(f32_tensor, kSeqDim, kSeqLen, delta);

    for (size_t h = 0; h < kHeads; ++h) {
        for (size_t p = 0; p < kSeqLen; ++p) {
            for (size_t d = 0; d < kWideHeadDim; ++d) {
                const size_t idx = (h * kSeqLen + p) * kWideHeadDim + d;
                if (d < kWideRotary) {
                    EXPECT_NEAR(static_cast<float>(f16_cache[idx]), f32_cache[idx], 2e-3f)
                        << "at head " << h << " pos " << p << " ch " << d;
                } else {
                    EXPECT_EQ(f16_cache[idx].to_bits(), before[idx].to_bits())
                        << "pass-through channel " << d << " was touched";
                }
            }
        }
    }
}
