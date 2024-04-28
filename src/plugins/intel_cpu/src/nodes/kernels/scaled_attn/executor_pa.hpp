// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>
#include "cpu_memory.h"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

struct PagedAttentionExecutor {
    // PagedAttention input index
    static const size_t ID_Q = 0;
    static const size_t ID_K = 1;
    static const size_t ID_V = 2;
    static const size_t ID_KCACHE = 3;
    static const size_t ID_VCACHE = 4;
    static const size_t ID_IS_PROMPT = 5;
    static const size_t ID_SLOT_MAPPING = 6;
    static const size_t ID_MAX_CONTEXT_LEN = 7;
    static const size_t ID_CONTEXT_LENS = 8;
    static const size_t ID_BLOCK_TABLES = 9;
    static const size_t ID_SCALE = 10;
    static const size_t ID_ALIBI_SLOPES = 11;
    static const size_t ID_SLIDING_WINDOW = 12;
    static const size_t ID_SUBSEQUENCE_LENS = 13;
    virtual void execute(const std::vector<ov::intel_cpu::MemoryPtr>& inputs, const ov::intel_cpu::MemoryPtr output) = 0;
};

std::shared_ptr<ov::Extensions::Cpu::XARCH::PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type, ov::element::Type kvcache_type);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov