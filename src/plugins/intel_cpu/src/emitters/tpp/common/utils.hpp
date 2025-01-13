// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "emitters/utils.hpp"
#include "libxsmm.h"

namespace ov {
namespace intel_cpu {
namespace tpp {

inline libxsmm_datatype ov_to_xsmm_dtype(ov::element::Type_t elemet_type) {
    switch (elemet_type) {
    case ov::element::Type_t::f32:
        return LIBXSMM_DATATYPE_F32;
    case ov::element::Type_t::bf16:
        return LIBXSMM_DATATYPE_BF16;
    case ov::element::Type_t::f16:
        return LIBXSMM_DATATYPE_F16;
    case ov::element::Type_t::i8:
        return LIBXSMM_DATATYPE_I8;
    case ov::element::Type_t::u8:
        return LIBXSMM_DATATYPE_U8;
    default:
        OV_CPU_JIT_EMITTER_THROW("Attempt to convert unsupported ov data type");
        return LIBXSMM_DATATYPE_IMPLICIT;
    }
}

}  // namespace tpp
}  // namespace intel_cpu
}  // namespace ov
