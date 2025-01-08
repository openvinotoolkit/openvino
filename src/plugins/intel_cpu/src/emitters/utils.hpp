// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "libxsmm.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

std::string jit_emitter_pretty_name(const std::string& pretty_func);

#ifdef __GNUC__
#    define OV_CPU_JIT_EMITTER_NAME jit_emitter_pretty_name(__PRETTY_FUNCTION__)
#else /* __GNUC__ */
#    define OV_CPU_JIT_EMITTER_NAME jit_emitter_pretty_name(__FUNCSIG__)
#endif /* __GNUC__ */

#define OV_CPU_JIT_EMITTER_THROW(...)        OPENVINO_THROW(OV_CPU_JIT_EMITTER_NAME, ": ", __VA_ARGS__)
#define OV_CPU_JIT_EMITTER_ASSERT(cond, ...) OPENVINO_ASSERT((cond), OV_CPU_JIT_EMITTER_NAME, ": ", __VA_ARGS__)

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

}  // namespace ov::intel_cpu
