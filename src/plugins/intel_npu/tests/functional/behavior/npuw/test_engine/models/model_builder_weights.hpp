// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace npuw {

struct FloatWeight {
    ov::element::Type storage_type;

    FloatWeight(ov::element::Type st = ov::element::f32) : storage_type(st) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

using FP32Weight = FloatWeight;
struct FP16Weight : FloatWeight {
    FP16Weight() : FloatWeight(ov::element::f16) {}
};

/// Decompression pattern for CompressedWeight, matching DCOFF recognition.
///
/// After NPUW partitioning, Constants become Parameters.  The patterns below
/// describe the graph that DCOFF will see *before* partitioning transforms it.
///
///   Pattern         | Chain (f16/f32 = decomp type)             | DCOFF class matched
///   ----------------+-------------------------------------------+------------------------------
///   SYMM_NO_ZP      | Cvt(f16) → Mul(f16 scale) [→ Reshape]    | Reshape3 / Reshape4
///   SYMM_NO_ZP_F32  | Cvt(f32) → Mul(f32 scale) [→ Reshape]    | SymmNoZP::MatMul / Reshape4
///   SYMM_ZP         | Cvt(f16) → Sub(Const u4→Cvt f16) → Mul   | Reshape1 / Convert1
///   GPTQ            | Cvt(f32) → Sub(Const f32) → Mul(f32)     | Reshape2
///   ASYMM_ZP        | Cvt(f16) → Sub(varying u4→Cvt f16) → Mul | AsymmZP::Reshape
enum class DCOffPattern {
    SYMM_NO_ZP,      ///< f16 chain, no zero point.  i4/i8/u4 storage.
    SYMM_NO_ZP_F32,  ///< f32 chain, no zero point.  i4/i8/nf4 storage.
    SYMM_ZP,         ///< f16 chain, uniform u4 zero point (Constant after partitioning).  u4 storage.
    GPTQ,            ///< f32 chain, uniform f32 zero point (no Convert on ZP).  u4 storage.
    ASYMM_ZP,        ///< f16 chain, per-layer varying u4 zero point (Parameter after partitioning).  u4 storage.
};

/// Compressed (quantized) weight with configurable DCOFF decompression pattern.
/// group_size > 0 = per-group scale (3D weight → decompress → Reshape 2D).
/// group_size = 0 = per-channel scale (2D weight, no Reshape).
/// Note: GPTQ and ASYMM_ZP require group_size > 0 (no per-channel DCOFF pass exists).
struct CompressedWeight {
    ov::element::Type storage_type;
    size_t group_size;     ///< 0 = per-channel scale, >0 = per-group scale
    DCOffPattern pattern;  ///< Decompression pattern to generate

    explicit CompressedWeight(ov::element::Type st, size_t gs = 0, DCOffPattern pat = DCOffPattern::SYMM_NO_ZP)
        : storage_type(st),
          group_size(gs),
          pattern(pat) {}

    ov::Output<ov::Node> operator()(const std::string& name,
                                    const ov::Shape& shape,
                                    ov::element::Type compute_precision) const;
};

struct INT8Weight : CompressedWeight {
    INT8Weight() : CompressedWeight(ov::element::i8) {}
};

struct INT4Weight : CompressedWeight {
    INT4Weight() : CompressedWeight(ov::element::i4) {}
};

struct INT4GroupWeight : CompressedWeight {
    explicit INT4GroupWeight(size_t gs = 128) : CompressedWeight(ov::element::i4, gs) {}
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
