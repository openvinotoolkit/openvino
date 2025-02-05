// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "xbyak_riscv/xbyak_riscv_util.hpp"

#include "openvino/core/except.hpp"


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
    v_bit = 1u << 6,

    last_bit = 1u << (cpu_isa_total_bits - 1),    
};

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    imafdc = i_bit | m_bit | a_bit | f_bit | d_bit | c_bit,
    imafdcv = imafdc | v_bit,
    isa_all = ~0u & ~last_bit
};

static std::string isa2str(cpu_isa_t isa) {
    switch(isa) {
    case cpu_isa_t::isa_undef: return "undef";
    case cpu_isa_t::imafdc: return "imafdc";
    case cpu_isa_t::imafdcv: return "imafdcv";
    case cpu_isa_t::isa_all: return "all";
    default: OPENVINO_THROW("Uknown ISA");
    }
}

static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    using namespace Xbyak_riscv;

    const auto cpu = CPU();
    switch (cpu_isa) {
        case imafdc: return cpu.hasExtension(RISCVExtension::I) &&
                            cpu.hasExtension(RISCVExtension::M) &&
                            cpu.hasExtension(RISCVExtension::A) &&
                            cpu.hasExtension(RISCVExtension::F) &&
                            cpu.hasExtension(RISCVExtension::D) &&
                            cpu.hasExtension(RISCVExtension::C);
        case imafdcv: return mayiuse(imafdc) && cpu.hasExtension(RISCVExtension::V);
        case isa_all: return false;
        case isa_undef: return true;
    }
    return false;
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
