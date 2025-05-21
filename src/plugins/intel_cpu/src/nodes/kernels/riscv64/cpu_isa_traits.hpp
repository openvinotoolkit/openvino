// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ov {
namespace intel_cpu {
namespace riscv64 {

// Maximum number of features + hints that can be specified via bits
static constexpr int cpu_isa_total_bits = sizeof(unsigned) * 8;

enum cpu_isa_bit_t : unsigned {
    i_bit = 1u << 0,
    m_bit = 1u << 1,
    a_bit = 1u << 2,
    f_bit = 1u << 3,
    d_bit = 1u << 4,
    c_bit = 1u << 5,
    v_bit = 1u << 6,  // rvv 1.0

    last_bit = 1u << (cpu_isa_total_bits - 1),    
};

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    g = i_bit | m_bit | a_bit | f_bit | d_bit, // G = IMAFD
    gv = g | v_bit,
    isa_all = ~0u & ~last_bit
};

std::string isa2str(cpu_isa_t isa);

bool mayiuse(const cpu_isa_t cpu_isa);

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
