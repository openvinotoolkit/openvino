// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ov::intel_gpu::op {

/// \brief Quantization of one paged QSA main-KV cache (K or V).
///
/// A quantized cache is a u8 tensor [blocks, Hkv, total_rows, row_bytes]. Per (block, head) page the
/// first `page_size` rows hold the token-major data (4-bit values are packed two per byte, the low nibble
/// holds the even channel); the scale/zero-point parameters follow in the remaining rows:
///   BY_TOKEN   : per token `[scale f16][zp f16]` (the zp half is unused when symmetric)
///   BY_CHANNEL : per group of `group_size` tokens `[scale f16 x head_size][zp f16 x head_size]`
/// Dequantization: x = (q - zp) * scale. A symmetric cache stores biased codes (zp = 128 / 8).
struct KVQuant {
    enum class Mode : uint8_t { NONE = 0, BY_TOKEN = 1, BY_CHANNEL = 2 };

    Mode mode = Mode::NONE;
    uint8_t bits = 16;        ///< 8 or 4 when quantized
    bool symmetric = false;   ///< no zero point stored, biased code
    uint16_t group_size = 0;  ///< tokens per parameter group (BY_CHANNEL only)

    bool enabled() const {
        return mode != Mode::NONE;
    }
    bool operator==(const KVQuant& rhs) const {
        return mode == rhs.mode && bits == rhs.bits && symmetric == rhs.symmetric && group_size == rhs.group_size;
    }
    bool operator!=(const KVQuant& rhs) const {
        return !(*this == rhs);
    }
};

/// Byte geometry of one (block, head) page of a cache.
struct KVPageLayout {
    size_t row_bytes = 0;     ///< bytes of one token row (last dim of the cache tensor)
    size_t param_bytes = 0;   ///< bytes of scale/zp parameters per page
    size_t extra_rows = 0;    ///< rows appended to the page to hold the parameters
    size_t total_rows = 0;    ///< page_size + extra_rows (third dim of the cache tensor)
    size_t page_bytes = 0;    ///< total_rows * row_bytes
    size_t param_offset = 0;  ///< byte offset of the parameters inside the page (== page_size * row_bytes)
};

inline KVPageLayout kv_page_layout(const KVQuant& q, size_t page_size, size_t head_size) {
    KVPageLayout l;
    if (!q.enabled()) {
        l.row_bytes = head_size * 2;  // f16
        l.total_rows = page_size;
        l.page_bytes = page_size * l.row_bytes;
        l.param_offset = l.page_bytes;
        return l;
    }
    l.row_bytes = head_size * q.bits / 8;
    const size_t values_per_param = q.symmetric ? 1 : 2;
    if (q.mode == KVQuant::Mode::BY_TOKEN) {
        l.param_bytes = page_size * 4;  // [scale f16][zp f16], the zp slot is unused when symmetric
    } else {
        l.param_bytes = (page_size / q.group_size) * head_size * values_per_param * 2;
    }
    l.extra_rows = (l.param_bytes + l.row_bytes - 1) / l.row_bytes;
    l.total_rows = page_size + l.extra_rows;
    l.page_bytes = l.total_rows * l.row_bytes;
    l.param_offset = page_size * l.row_bytes;
    return l;
}

/// Checks a cache quantization against the geometry it is applied to. `is_key` enables BY_CHANNEL.
inline bool validate_kv_quant(const KVQuant& q, bool is_key, size_t page_size, size_t head_size, size_t key_block_size, std::string& reason) {
    if (!q.enabled()) {
        return true;
    }
    if (q.bits != 8 && q.bits != 4) {
        reason = "quantized cache must use 8 or 4 bits";
        return false;
    }
    if (page_size == 0) {
        reason = "a quantized cache needs an explicit page_size";
        return false;
    }
    if (head_size < 32 || (head_size & (head_size - 1)) != 0) {
        reason = "a quantized cache needs a power-of-two head size of at least 32";
        return false;
    }
    if (q.mode == KVQuant::Mode::BY_CHANNEL) {
        if (!is_key) {
            reason = "BY_CHANNEL quantization is only supported for the key cache";
            return false;
        }
        if (q.group_size != 16 && q.group_size != 32) {
            reason = "BY_CHANNEL group_size must be 16 or 32";
            return false;
        }
        if (page_size % q.group_size != 0 || (key_block_size != 0 && q.group_size % key_block_size != 0)) {
            reason = "BY_CHANNEL group_size must be a multiple of key_block_size and divide the page size";
            return false;
        }
    } else if (q.group_size != 0) {
        reason = "group_size only applies to BY_CHANNEL quantization";
        return false;
    }
    return true;
}

}  // namespace ov::intel_gpu::op
