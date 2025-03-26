// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_isa_traits.hpp"

#include "xbyak_riscv/xbyak_riscv_util.hpp"

#include "openvino/core/except.hpp"

#include <csignal>
#include <csetjmp>

using namespace Xbyak_riscv;

namespace ov {
namespace intel_cpu {
namespace riscv64 {

namespace {

struct RVVGenerator : public CodeGenerator {
    RVVGenerator() : CodeGenerator(8) {
        // vsetivli is appeared in RVV 1.0
        vsetivli(a0, 10, SEW::e32);
        ret();
    }
};

static thread_local sigjmp_buf jmpbuf;

static bool can_compile_rvv100() {
#if defined(__linux__)
    __sighandler_t signal_handler = [](int signal) {
        siglongjmp(jmpbuf, 1);
    };

    struct sigaction new_sa, old_sa;
    new_sa.sa_handler = signal_handler;
    sigemptyset(&new_sa.sa_mask);
    new_sa.sa_flags = SA_RESETHAND;
    sigaction(SIGILL, &new_sa, &old_sa);

    bool status = false;
    if (sigsetjmp(jmpbuf, 1) == 0) {
        RVVGenerator gen;
        gen.ready();
        const auto caller = gen.getCode<uint32_t (*)()>();
        status = static_cast<bool>(caller());
    }

    // Restore original signal handler
    sigaction(SIGILL, &old_sa, nullptr);

    return status;
#else
    return false;
#endif
}

}  // namespace


bool mayiuse(const cpu_isa_t cpu_isa) {
    const auto cpu = CPU::getInstance();
    switch (cpu_isa) {
        case g: return cpu.hasExtension(RISCVExtension::I) &&
                       cpu.hasExtension(RISCVExtension::M) &&
                       cpu.hasExtension(RISCVExtension::A) &&
                       cpu.hasExtension(RISCVExtension::F) &&
                       cpu.hasExtension(RISCVExtension::D);
        // cpu.hasExtension(RISCVExtension::V) checks only RVV support on the device.
        // To figure out RVV version, we try to execute code with RVV1.0 instructions.
        // If there is no `SEGILL`, the device supports RVV1.0.
        // Otherwise we consider that there is no RVV support
        // [TODO] If needed, support other RVV versions
        case gv: return mayiuse(g) && cpu.hasExtension(RISCVExtension::V) && can_compile_rvv100();
        case isa_all: return false;
        case isa_undef: return true;
    }
    return false;
}

std::string isa2str(cpu_isa_t isa) {
    switch(isa) {
    case cpu_isa_t::isa_undef: return "undef";
    case cpu_isa_t::g: return "g";
    case cpu_isa_t::gv: return "gv";
    case cpu_isa_t::isa_all: return "all";
    default: OPENVINO_THROW("Uknown ISA");
    }
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
