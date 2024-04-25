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
    TileConfiger m_tile_configer;
    TileConfig m_tile_cfg;
    JitMatMulVecAMX(int head_size, int block_size);

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