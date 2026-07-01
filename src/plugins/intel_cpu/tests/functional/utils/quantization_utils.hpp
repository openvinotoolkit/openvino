// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <unordered_map>

namespace CPUTestUtils {

struct QuantizationData {
    QuantizationData(float il, float ih, float ol, float oh, int levels)
        : il(il),
          ih(ih),
          ol(ol),
          oh(oh),
          levels(levels) {}

    QuantizationData(float il, float ih, int levels) : il(il), ih(ih), ol(il), oh(ih), levels(levels) {}

    float il;
    float ih;
    float ol;
    float oh;
    int levels;
};

struct QuantizationInfo {
    std::unordered_map<size_t, QuantizationData> inputs;
    std::unordered_map<size_t, QuantizationData> outputs;

    bool empty() const {
        return inputs.empty() && outputs.empty();
    }
};

inline std::ostream& operator<<(std::ostream& os, const QuantizationData& qdata) {
    os << qdata.il << "_" << qdata.ih << "_" << qdata.ol << "_" << qdata.oh << "_levels=" << qdata.levels;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const QuantizationInfo& qinfo) {
    os << "QuantizationInfo_[";
    os << "inputs_[";
    if (!qinfo.inputs.empty()) {
        for (const auto& [inputId, qData] : qinfo.inputs) {
            os << inputId << "_[" << qData << "]_";
        }
    }
    os << "]_";

    os << "outputs_[";
    if (!qinfo.outputs.empty()) {
        for (const auto& [outputId, qData] : qinfo.inputs) {
            os << outputId << "_[" << qData << "]_";
        }
    }
    os << "]";

    return os;
}

// Input: n float values in src.
// Output: u8 values in dst, plus scale and zp.
// Algorithm: min/max affine quantization to 255 levels.
inline void scalar_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    float max_val = -std::numeric_limits<float>::max();
    float min_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; i++) {
        max_val = std::max(max_val, src[i]);
        min_val = std::min(min_val, src[i]);
    }
    scale = (max_val - min_val) / 255.0f;
    if (scale == 0.0f)
        scale = 0.0001f;
    zp = -min_val / scale;
    for (size_t i = 0; i < n; i++) {
        dst[i] = static_cast<uint8_t>(std::round(std::max(src[i] / scale + zp, 0.0f)));
    }
}

// Input: n u8 values in src, with scale and zp.
// Output: float values in dst.
// Algorithm: inverse affine dequantization.
inline void scalar_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = (static_cast<float>(src[i]) - zp) * scale;
    }
}

// Input: float matrix src with shape [seq_dim, hidden_dims].
// Output: per-channel u8 rows in dst, plus scale and zp per channel.
// Algorithm: channel-wise min/max affine quantization.
inline void scalar_quant_u8_by_channel(const float* src,
                                       uint8_t* dst,
                                       size_t seq_dim,
                                       size_t hidden_dims,
                                       size_t dst_stride,
                                       float* scale,
                                       float* zp) {
    for (size_t j = 0; j < hidden_dims; j++) {
        float max_val = -std::numeric_limits<float>::max();
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < seq_dim; i++) {
            float v = src[i * hidden_dims + j];
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
        scale[j] = (max_val - min_val) / 255.0f;
        if (scale[j] == 0.0f)
            scale[j] = 0.0001f;
        zp[j] = -min_val / scale[j];
    }
    for (size_t i = 0; i < seq_dim; i++) {
        for (size_t j = 0; j < hidden_dims; j++) {
            dst[i * dst_stride + j] =
                static_cast<uint8_t>(std::round(std::max(src[i * hidden_dims + j] / scale[j] + zp[j], 0.0f)));
        }
    }
}

// Input: per-channel u8 matrix in src, with scale and zp arrays.
// Output: float matrix in dst.
// Algorithm: channel-wise inverse affine dequantization.
inline void scalar_dequant_u8_by_channel(const uint8_t* src,
                                         float* dst,
                                         size_t seq_dim,
                                         size_t hidden_dims,
                                         size_t src_stride,
                                         const float* scale,
                                         const float* zp) {
    for (size_t i = 0; i < seq_dim; i++) {
        for (size_t j = 0; j < hidden_dims; j++) {
            dst[i * hidden_dims + j] = (static_cast<float>(src[i * src_stride + j]) - zp[j]) * scale[j];
        }
    }
}

// Input: n float values in src.
// Output: i8 values in dst, plus scale.
// Algorithm: symmetric quantization using max absolute value.
inline void scalar_quant_i8(const float* src, int8_t* dst, size_t n, float& scale) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        max_abs = std::max(max_abs, std::abs(src[i]));
    }
    scale = max_abs / 127.0f;
    if (scale == 0.0f)
        scale = 0.0001f;
    for (size_t i = 0; i < n; i++) {
        float tmp = std::round(src[i] / scale);
        tmp = std::max(tmp, -128.0f);
        tmp = std::min(tmp, 127.0f);
        dst[i] = static_cast<int8_t>(tmp);
    }
}

// Input: n i8 values in src, with scale.
// Output: float values in dst.
// Algorithm: symmetric dequantization.
inline void scalar_dequant_i8(const int8_t* src, float* dst, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// Input: a packed byte, one u4 value, and the target nibble selector.
// Output: the byte with the selected nibble updated.
// Algorithm: shift and OR the 4-bit payload into place.
inline uint8_t insert_u4(uint8_t dst_byte, uint8_t val, bool high_half) {
    uint8_t shift = high_half ? 0 : 4;
    return dst_byte | static_cast<uint8_t>(val << shift);
}

// Input: a packed byte and the nibble selector.
// Output: the selected u4 value.
// Algorithm: shift and mask the chosen nibble.
inline uint8_t extract_u4(uint8_t byte, bool high_half) {
    uint8_t shift = high_half ? 0 : 4;
    return static_cast<uint8_t>((byte >> shift) & 0x0F);
}

// Input: n float values in src.
// Output: packed u4 values in dst, plus scale and zp.
// Algorithm: min/max affine quantization to 16 levels, packed two per byte.
inline void scalar_quant_u4(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    float max_val = -std::numeric_limits<float>::max();
    float min_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < n; i++) {
        max_val = std::max(max_val, src[i]);
        min_val = std::min(min_val, src[i]);
    }
    scale = (max_val - min_val) / 15.0f;
    if (scale == 0.0f)
        scale = 0.0001f;
    zp = -min_val / scale;
    for (size_t i = 0; i < n; i++) {
        uint8_t val = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::round(src[i] / scale + zp))));
        uint8_t dst_val = (i % 2 == 0) ? 0 : dst[i / 2];
        dst[i / 2] = insert_u4(dst_val, val, static_cast<bool>(i % 2));
    }
}

// Input: packed u4 data in src, with scale and zp.
// Output: float values in dst.
// Algorithm: unpack and inverse affine dequantization.
inline void scalar_dequant_u4(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
    for (size_t i = 0; i < n; i++) {
        uint8_t val = extract_u4(src[i / 2], static_cast<bool>(i % 2));
        dst[i] = (static_cast<float>(val) - zp) * scale;
    }
}

// Input: float matrix src with shape [seq_dim, hidden_dims].
// Output: per-channel packed u4 rows in dst, plus scale and zp per channel.
// Algorithm: channel-wise min/max affine quantization to 16 levels.
inline void scalar_quant_u4_by_channel(const float* src,
                                       uint8_t* dst,
                                       size_t seq_dim,
                                       size_t hidden_dims,
                                       size_t dst_stride,
                                       float* scale,
                                       float* zp) {
    for (size_t j = 0; j < hidden_dims; j++) {
        float max_val = -std::numeric_limits<float>::max();
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < seq_dim; i++) {
            float v = src[i * hidden_dims + j];
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
        scale[j] = (max_val - min_val) / 15.0f;
        if (scale[j] == 0.0f)
            scale[j] = 0.0001f;
        zp[j] = -min_val / scale[j];
    }
    for (size_t i = 0; i < seq_dim; i++) {
        for (size_t j = 0; j < hidden_dims; j++) {
            uint8_t val = static_cast<uint8_t>(
                std::min(15.0f, std::max(0.0f, std::round(src[i * hidden_dims + j] / scale[j] + zp[j]))));
            size_t byte_idx = i * dst_stride + j / 2;
            uint8_t dst_val = (j % 2 == 0) ? 0 : dst[byte_idx];
            dst[byte_idx] = insert_u4(dst_val, val, static_cast<bool>(j % 2));
        }
    }
}

// Input: per-channel packed u4 rows in src, with scale and zp arrays.
// Output: float matrix in dst.
// Algorithm: unpack and channel-wise inverse affine dequantization.
inline void scalar_dequant_u4_by_channel(const uint8_t* src,
                                         float* dst,
                                         size_t seq_dim,
                                         size_t hidden_dims,
                                         size_t src_stride,
                                         const float* scale,
                                         const float* zp) {
    for (size_t i = 0; i < seq_dim; i++) {
        for (size_t j = 0; j < hidden_dims; j++) {
            uint8_t val = extract_u4(src[i * src_stride + j / 2], static_cast<bool>(j % 2));
            dst[i * hidden_dims + j] = (static_cast<float>(val) - zp[j]) * scale[j];
        }
    }
}

}  // namespace CPUTestUtils
