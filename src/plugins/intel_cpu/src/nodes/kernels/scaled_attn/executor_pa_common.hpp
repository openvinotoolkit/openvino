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
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {

// this file will contain features that do not require multiple instantiation

struct PagedAttentionExecutor {
    // PagedAttention input index
    static const size_t ID_Q = 0;                           // [B_token, H * S], float
    static const size_t ID_K = 1;                           // [B_token, Hk * S], float
    static const size_t ID_V = 2;                           // [B_token, Hk * S], float
    static const size_t ID_KCACHE = 3;                      // [block_number, H, block_size, S], float
    static const size_t ID_VCACHE = 4;                      // [block_number, H, block_size, S], float
    static const size_t ID_PAST_LENS = 5;                   // [B_seq]
    static const size_t ID_SUBSEQUENCE_BEGINS = 6;          // [B_seq+1]
    static const size_t ID_BLOCK_INDICES = 7;               // [num_blocks]
    static const size_t ID_BLOCK_INDICES_BEGINS = 8;        // [B_seq+1]
    static const size_t ID_SCALE = 9;                       // [], float
    static const size_t ID_SLIDING_WINDOW = 10;             // []
    static const size_t ID_ALIBI_SLOPES = 11;               // [H|0], float
    static const size_t ID_MAX_CONTEXT_LEN = 12;            // []
    virtual void execute(const std::vector<ov::intel_cpu::MemoryPtr>& inputs, const std::vector<ov::intel_cpu::MemoryPtr> outputs) = 0;
    virtual ~PagedAttentionExecutor() = default;
};

#ifdef OPENVINO_ARCH_X86_64

// w = query * Key
//
// query: [1,      S]
// Key  : [block_size, S]
// w    : [1, block_size]
//
// S is known at compile time
struct TileConfig {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    void reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes);
};

class TileConfiger : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(TileConfiger)
    TileConfiger();
    void generate() override;
};

class JitMatMulVecAMX : public dnnl::impl::cpu::x64::jit_generator {
    void operator=(const JitMatMulVecAMX&);

public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitMatMulVecAMX)
    int m_head_size;
    int m_block_size;
    ov::element::Type m_amx_prec;
    TileConfiger m_tile_configer;
    TileConfig m_tile_cfg;
    JitMatMulVecAMX(int head_size, int block_size, ov::element::Type amx_prec);

    void tile_config() {
        m_tile_configer(&m_tile_cfg);
    }
    void tile_release() {
        m_tile_configer(nullptr);
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    static constexpr auto abi_param_regs = dnnl::impl::cpu::x64::abi_param_regs;
    Xbyak::Reg64 reg_q_addr = abi_param1;
    Xbyak::Reg64 reg_k_addr = abi_param2;
    Xbyak::Reg64 reg_dst_addr = abi_param3;
    Xbyak::Reg64 reg_stride_A = rax;
    Xbyak::Reg64 reg_stride_BC = r9;

    Xbyak::Tmm tmmC = tmm0;
    Xbyak::Tmm tmmA = tmm1;
    Xbyak::Tmm tmmB0 = tmm2;
    Xbyak::Tmm tmmB1 = tmm3;
    Xbyak::Tmm tmmB2 = tmm4;
    Xbyak::Tmm tmmB3 = tmm5;
    Xbyak::Tmm tmmB4 = tmm6;
    Xbyak::Tmm tmmB5 = tmm7;

    void generate() override;
};

#endif

}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov