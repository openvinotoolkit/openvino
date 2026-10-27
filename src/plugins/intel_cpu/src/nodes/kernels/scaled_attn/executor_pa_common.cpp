// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(XBYAK_RISCV_V) && !defined(OV_CPU_WITH_ACL)
#    include "executor_pa_common.hpp"
#endif

#include <cfloat>
#include <cmath>
#include <cstring>

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <xbyak/xbyak.h>

#    include <cpu/x64/jit_generator.hpp>
#    include <cstdint>
#    include <stdexcept>
#    include <utility>
#    include <vector>

#    include "openvino/core/type/element_type.hpp"
#endif

namespace ov::Extensions::Cpu {

using namespace ov;

#ifdef OPENVINO_ARCH_X86_64
using namespace ov::intel_cpu;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

void TileConfig::reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
    palette_id = palette;
    startRow = _startRow;
    uint64_t i = 0;
    for (; i < 14; i++) {
        reserved[i] = 0;
    }
    for (i = 0; i < _rows_columnsBytes.size(); i++) {
        rows[i] = _rows_columnsBytes[i].first;
        cols[i] = _rows_columnsBytes[i].second;
    }
    for (; i < 16; i++) {
        cols[i] = 0;
        rows[i] = 0;
    }
}

TileConfiger::TileConfiger() : jit_generator_t(jit_name()) {
    create_kernel();
}

void TileConfiger::generate() {
    Xbyak::Label release;
    test(abi_param1, abi_param1);
    jz(release);
    ldtilecfg(ptr[abi_param1]);
    ret();
    L(release);
    tilerelease();
    ret();
}

JitMatMulVecAMX::JitMatMulVecAMX(int head_size, int block_size, ov::element::Type amx_prec)
    : jit_generator_t(jit_name()),
      m_head_size(head_size),
      m_block_size(block_size),
      m_amx_prec(amx_prec) {
    if (m_head_size <= 0 || (m_head_size % 32) != 0) {
        throw std::runtime_error("head size is not a positive multiple of 32");
    }
    if (m_head_size > kMaxHeadSize) {
        throw std::runtime_error("head size is bigger than the supported maximum");
    }
    if (m_block_size <= 0 || (m_block_size % 16) != 0) {
        throw std::runtime_error("block size is not a positive multiple of 16");
    }
    create_kernel();
    m_tile_cfg.reset(1,
                     0,
                     {
                         {16, 4},   // C:0   M x 1     (4b)
                         {16, 64},  // A:1   M x 32/64 (64b)
                         {16, 4},   // B:2   32/64 x 1 (4b)
                         {16, 4},   // B:3
                         {16, 4},   // B:4
                         {16, 4},   // B:5
                         {16, 4},   // B:6
                         {16, 4},   // B:7
                     });
}

void JitMatMulVecAMX::generate() {
    mov(reg_stride_A, m_head_size * 2);
    mov(reg_stride_BC, 4);
    const int kStep = 32;
    // C and A take two of the eight tile registers, the other six hold query (B) tiles.
    constexpr int kMaxResidentBTiles = 6;
    auto num_B_tiles = m_head_size / kStep;

    /*
                                B(query)    head_size x 1
    A(key) matrix : block_size x head_size  C(dst) block_size x 1
    */
    if (num_B_tiles <= kMaxResidentBTiles) {
        // The whole query fits in the B tiles: load it once, reuse it for every block of 16 keys.
        for (int i = 0; i < num_B_tiles; i++) {
            tileloadd(Xbyak::Tmm(tmmB0.getIdx() + i), ptr[reg_q_addr + reg_stride_BC + i * 64]);
        }

        for (int m = 0; m < m_block_size; m += 16) {
            tilezero(tmmC);
            for (int i = 0; i < num_B_tiles; i++) {
                tileloadd(tmmA, ptr[reg_k_addr + reg_stride_A + i * 64]);
                if (m_amx_prec == ov::element::bf16) {
                    tdpbf16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
                } else if (m_amx_prec == ov::element::f16) {
                    tdpfp16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
                }
            }
            tilestored(ptr[reg_dst_addr + reg_stride_BC + m * sizeof(float)], tmmC);
            add(reg_k_addr, m_head_size * 2 * 16);
        }
        ret();
        return;
    }

    // More B tiles than registers: loop over K chunks of at most kMaxResidentBTiles tiles, balanced (224 = 7 tiles ->
    // 4 + 3, 256 -> 4 + 4, 512 = 16 tiles -> 6 + 5 + 5). Every chunk runs the loop over the key blocks with its slice
    // of the query resident and accumulates its partial scores into dst: the first chunk starts from zero, the others
    // from the scores the previous chunks stored. Each key tile is still loaded exactly once. The key loop advances
    // reg_k_addr past the whole key matrix, so every chunk restarts from the preserved reg_k_base; a 32-element tile is
    // 64 bytes in both the query and a key row, so one cumulative offset serves both.
    mov(reg_k_base, reg_k_addr);
    const int num_chunks = (num_B_tiles + kMaxResidentBTiles - 1) / kMaxResidentBTiles;
    const int base_tiles = num_B_tiles / num_chunks;
    const int remainder = num_B_tiles % num_chunks;
    int first_tile = 0;
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_tiles = base_tiles + (chunk < remainder ? 1 : 0);
        for (int i = 0; i < chunk_tiles; i++) {
            tileloadd(Xbyak::Tmm(tmmB0.getIdx() + i), ptr[reg_q_addr + reg_stride_BC + (first_tile + i) * 64]);
        }
        lea(reg_k_addr, ptr[reg_k_base + first_tile * 64]);
        for (int m = 0; m < m_block_size; m += 16) {
            if (chunk == 0) {
                tilezero(tmmC);
            } else {
                tileloadd(tmmC, ptr[reg_dst_addr + reg_stride_BC + m * sizeof(float)]);
            }
            for (int i = 0; i < chunk_tiles; i++) {
                tileloadd(tmmA, ptr[reg_k_addr + reg_stride_A + i * 64]);
                if (m_amx_prec == ov::element::bf16) {
                    tdpbf16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
                } else if (m_amx_prec == ov::element::f16) {
                    tdpfp16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
                }
            }
            tilestored(ptr[reg_dst_addr + reg_stride_BC + m * sizeof(float)], tmmC);
            add(reg_k_addr, m_head_size * 2 * 16);
        }
        first_tile += chunk_tiles;
    }
    ret();
}

#endif

}  // namespace ov::Extensions::Cpu
