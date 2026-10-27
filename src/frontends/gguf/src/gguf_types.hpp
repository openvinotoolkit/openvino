// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Mapping between the canonical on-disk GGML tensor type enum (the integer value stored in a
// .gguf tensor-info record) and the corresponding ov::element::Type.
//
// IMPORTANT: these integer values are the *file-format* GGML type values taken from ggml.h
// (`enum ggml_type`). They are NOT the internal enum of any particular GGUF reader library.
// Do not reorder or remap them.

#pragma once

#include <cstdint>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Canonical on-disk GGML tensor type values (enum ggml_type in ggml.h).
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 removed (Q4_2, Q4_3)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // 31, 32, 33 removed (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8)
    TQ1_0 = 34,
    TQ2_0 = 35,
};

// Translate an on-disk GGML type value to the corresponding ov::element::Type.
// Throws for unknown / unsupported type ids - the frontend never silently falls back.
inline ov::element::Type ggml_type_to_ov_element_type(uint32_t ggml_type) {
    switch (static_cast<GGMLType>(ggml_type)) {
    case GGMLType::F32:
        return ov::element::f32;
    case GGMLType::F16:
        return ov::element::f16;
    case GGMLType::BF16:
        return ov::element::bf16;
    case GGMLType::F64:
        return ov::element::f64;
    case GGMLType::I8:
        return ov::element::i8;
    case GGMLType::I16:
        return ov::element::i16;
    case GGMLType::I32:
        return ov::element::i32;
    case GGMLType::I64:
        return ov::element::i64;
    case GGMLType::Q4_0:
        return ov::element::gguf_q4_0;
    case GGMLType::Q4_1:
        return ov::element::gguf_q4_1;
    case GGMLType::Q5_0:
        return ov::element::gguf_q5_0;
    case GGMLType::Q5_1:
        return ov::element::gguf_q5_1;
    case GGMLType::Q8_0:
        return ov::element::gguf_q8_0;
    case GGMLType::Q8_1:
        return ov::element::gguf_q8_1;
    case GGMLType::Q2_K:
        return ov::element::gguf_q2_k;
    case GGMLType::Q3_K:
        return ov::element::gguf_q3_k;
    case GGMLType::Q4_K:
        return ov::element::gguf_q4_k;
    case GGMLType::Q5_K:
        return ov::element::gguf_q5_k;
    case GGMLType::Q6_K:
        return ov::element::gguf_q6_k;
    case GGMLType::Q8_K:
        return ov::element::gguf_q8_k;
    case GGMLType::IQ2_XXS:
        return ov::element::gguf_iq2_xxs;
    case GGMLType::IQ2_XS:
        return ov::element::gguf_iq2_xs;
    case GGMLType::IQ3_XXS:
        return ov::element::gguf_iq3_xxs;
    case GGMLType::IQ1_S:
        return ov::element::gguf_iq1_s;
    case GGMLType::IQ4_NL:
        return ov::element::gguf_iq4_nl;
    case GGMLType::IQ3_S:
        return ov::element::gguf_iq3_s;
    case GGMLType::IQ2_S:
        return ov::element::gguf_iq2_s;
    case GGMLType::IQ4_XS:
        return ov::element::gguf_iq4_xs;
    case GGMLType::IQ1_M:
        return ov::element::gguf_iq1_m;
    case GGMLType::TQ1_0:
        return ov::element::gguf_tq1_0;
    case GGMLType::TQ2_0:
        return ov::element::gguf_tq2_0;
    default:
        OPENVINO_THROW("[GGUF Frontend] Unsupported GGML tensor type id: ", ggml_type);
    }
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
