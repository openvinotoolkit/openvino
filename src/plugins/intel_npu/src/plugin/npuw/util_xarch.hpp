// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace util {
namespace XARCH {

void unpack_i4i8(const ov::SoPtr<ov::ITensor>& from,
                 const ov::SoPtr<ov::ITensor>& to,
                 const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u4i8(const ov::SoPtr<ov::ITensor>& from,
                 const ov::SoPtr<ov::ITensor>& to,
                 const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_i4f16(const ov::SoPtr<ov::ITensor>& from,
                  const ov::SoPtr<ov::ITensor>& to,
                  const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_i4f16_scale(const ov::SoPtr<ov::ITensor>& from,
                        const ov::SoPtr<ov::ITensor>& scale,
                        const ov::SoPtr<ov::ITensor>& to,
                        const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_i4f16_z(const ov::SoPtr<ov::ITensor>& from,
                    const ov::SoPtr<ov::ITensor>& scale,
                    const ov::SoPtr<ov::ITensor>& to,
                    const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u4f16(const ov::SoPtr<ov::ITensor>& from,
                  const ov::SoPtr<ov::ITensor>& to,
                  const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u4f16_scale_zp(const ov::SoPtr<ov::ITensor>& from,
                           const ov::SoPtr<ov::ITensor>& zerop,
                           const ov::SoPtr<ov::ITensor>& scale,
                           const ov::SoPtr<ov::ITensor>& to,
                           const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u4f16_asymm_zp(const ov::SoPtr<ov::ITensor>& from,
                           const ov::SoPtr<ov::ITensor>& zerop,
                           const ov::SoPtr<ov::ITensor>& scale,
                           const ov::SoPtr<ov::ITensor>& to,
                           const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u4f16_z(const ov::SoPtr<ov::ITensor>& from,
                    const ov::SoPtr<ov::ITensor>& zerop,
                    const ov::SoPtr<ov::ITensor>& scale,
                    const ov::SoPtr<ov::ITensor>& to,
                    const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u4f32(const ov::SoPtr<ov::ITensor>& from,
                  const ov::SoPtr<ov::ITensor>& to,
                  const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_i8f16(const ov::SoPtr<ov::ITensor>& from,
                  const ov::SoPtr<ov::ITensor>& to,
                  const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_i8f16_scale(const ov::SoPtr<ov::ITensor>& from,
                        const ov::SoPtr<ov::ITensor>& scale,
                        const ov::SoPtr<ov::ITensor>& to,
                        const ov::npuw::util::UnpackOptions& unpack_options);

void unpack_u8f16(const ov::SoPtr<ov::ITensor>& from,
                  const ov::SoPtr<ov::ITensor>& zerop,
                  const ov::SoPtr<ov::ITensor>& scale,
                  const ov::SoPtr<ov::ITensor>& to,
                  const ov::npuw::util::UnpackOptions& _options);

ov::Tensor to_f16(const ov::Tensor& t);

void copy_row_as_column(const ov::SoPtr<ov::ITensor>& from, const ov::SoPtr<ov::ITensor>& to);

void transpose_i4(const uint8_t* src, uint8_t* dst, size_t rows, size_t cols);
void transpose_f16(const uint16_t* src, uint16_t* dst, size_t rows, size_t cols);
void transpose_f32(const float* src, float* dst, size_t rows, size_t cols);

void unpack_f8f16_scale(const ov::SoPtr<ov::ITensor>& from,
                        const ov::SoPtr<ov::ITensor>& scale,
                        const ov::SoPtr<ov::ITensor>& to,
                        const ov::npuw::util::UnpackOptions& unpack_options);

// Turns the leading num_tokens rows of one densely packed f16 key plane by a per-token
// rotation, in rotate_half layout: components j and j + half of every head_dim-wide row
// become (a * cos - b * sin, b * cos + a * sin), with (cos, sin) taken from row t of
// delta_cos/delta_sin. Channels at and beyond 2 * half are left untouched.
//
// Sequence entry t starts at plane[t * seq_stride] and spans rows_per_token rows.
void rerotate_f16_rows(uint16_t* plane,
                       size_t num_tokens,
                       size_t seq_stride,
                       size_t rows_per_token,
                       size_t head_dim,
                       const float* delta_cos,
                       const float* delta_sin,
                       size_t half);

}  // namespace XARCH
}  // namespace util
}  // namespace npuw
}  // namespace ov
