// Copyright (C) 2024 Intel Corporation
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

void transpose_i4_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_ROWS, size_t IN_COLS);
void transpose_f32_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_ROWS, size_t IN_COLS);

void permute021_i4_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_PLAS, size_t IN_ROWS, size_t IN_COLS);
void permute021_f32_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_PLAS, size_t IN_ROWS, size_t IN_COLS);
void permute021_f16_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_PLAS, size_t IN_ROWS, size_t IN_COLS);

void permute102_i4_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_PLAS, size_t IN_ROWS, size_t IN_COLS);
void permute102_f16_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_PLAS, size_t IN_ROWS, size_t IN_COLS);

void permute120_f16_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_ROWS, size_t IN_COLS);
void permute120_f32_avx2(const ov::Tensor& t, ov::Tensor& tnew, size_t IN_ROWS, size_t IN_COLS);

void memcpy_avx2(uint8_t* dst, const uint8_t* src, size_t len);

}  // namespace XARCH
}  // namespace util
}  // namespace npuw
}  // namespace ov
