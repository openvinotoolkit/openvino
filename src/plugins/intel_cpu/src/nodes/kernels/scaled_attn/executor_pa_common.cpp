// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/parallel.hpp"
#include "executor_pa_common.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {

using namespace ov;
using namespace ov::intel_cpu;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

#ifdef OPENVINO_ARCH_X86_64

void TileConfig::reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
    palette_id = palette;
    startRow = _startRow;
    unsigned long i;
    for (i = 0; i < 14; i++) {
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

TileConfiger::TileConfiger() : jit_generator(jit_name()) {
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

JitMatMulVecAMX::JitMatMulVecAMX(int head_size, int block_size, ov::element::Type amx_prec) :
    jit_generator(jit_name()), m_head_size(head_size), m_block_size(block_size), m_amx_prec(amx_prec) {
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
    if ((m_head_size % 32) != 0)
        throw std::runtime_error("head size is not multiple of 32");
    if ((m_block_size % 16) != 0)
        throw std::runtime_error("block size is not multiple of 16");
    auto num_B_tiles = m_head_size / kStep;
    if (num_B_tiles > 6)
        throw std::runtime_error("number of B tiles is bigger than 6");

    /*
                                B(query)    head_size x 1
    A(key) matrix : block_size x head_size  C(dst) block_size x 1
    */
    // load query into B tiles
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
}

#endif

}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov