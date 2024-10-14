// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif

#include "util_xarch.hpp"
#include "unpack_kernel.hpp"

void ov::npuw::util::XARCH::unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& to) {
              unpack_impl(from, to);
            }

void ov::npuw::util::XARCH::unpack_scale(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to) {
              unpack_scale_impl(from, scale, to);
            }

void ov::npuw::util::XARCH::unpack_scale_zp(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& zerop,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to) {
              unpack_scale_zp_impl(from, zerop, scale, to);
            }

ov::Tensor ov::npuw::util::XARCH::to_f16(const ov::Tensor& t) {
    return to_f16_impl(t);
}
