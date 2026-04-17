// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CacheCodec — identifies the quantization / encoding scheme of one KV cache side.
// Lightweight header with no dependencies beyond <cstdint>.

#pragma once

#include <cstdint>

namespace ov::Extensions::Cpu {

enum class CacheCodec : uint8_t {
    U8,
    U4,
    U8_BY_CHANNEL,
    RAW_F32,
    RAW_F16,
    RAW_BF16,
};

}  // namespace ov::Extensions::Cpu
