// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_isa_traits.hpp"

#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <string>

#include "openvino/core/except.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"
#include "xbyak_riscv/xbyak_riscv_util.hpp"

using namespace Xbyak_riscv;

namespace ov::intel_cpu::riscv64 {

namespace {

struct RVVGenerator : public CodeGenerator {
    RVVGenerator() : CodeGenerator(8) {
        // vsetivli is appeared in RVV 1.0
        vsetivli(a0, 10, SEW::e32);
        ret();
    }
};

struct ZvfhGenerator : public CodeGenerator {
    ZvfhGenerator() : CodeGenerator(32) {
        // Probe Zvfh instructions used by Snippets Convert emitters.
        vsetivli(a0, 1, SEW::e16, LMUL::mf2);
        vfwcvt_f_f_v(v0, v0);
        vfncvt_f_f_w(v0, v0);
        li(a0, 1);
        ret();
    }
};

// NOLINTBEGIN(misc-include-cleaner) bug in clang-tidy
template <typename Generator>
bool can_execute_generated_code() {
#if defined(__linux__)
    static thread_local sigjmp_buf jmpbuf;
    __sighandler_t signal_handler = []([[maybe_unused]] int signal) {
        siglongjmp(jmpbuf, 1);
    };

    struct sigaction new_sa {
    }, old_sa{};
    new_sa.sa_handler = signal_handler;
    sigemptyset(&new_sa.sa_mask);
    new_sa.sa_flags = SA_RESETHAND;
    sigaction(SIGILL, &new_sa, &old_sa);

    bool status = false;
    if (sigsetjmp(jmpbuf, 1) == 0) {
        Generator gen;
        gen.ready();
        const auto caller = gen.template getCode<uint32_t (*)()>();
        status = static_cast<bool>(caller());
    }

    // Restore original signal handler
    sigaction(SIGILL, &old_sa, nullptr);

    return status;
// NOLINTEND(misc-include-cleaner) bug in clang-tidy
#else
    return false;
#endif
}

bool can_compile_rvv100() {
    static const bool status = can_execute_generated_code<RVVGenerator>();
    return status;
}

bool can_compile_zvfh() {
    static const bool status = can_execute_generated_code<ZvfhGenerator>();
    return status;
}

}  // namespace

bool mayiuse(const cpu_isa_t cpu_isa) {
    const auto cpu = CPU::getInstance();
    switch (cpu_isa) {
    case g:
        return cpu.hasExtension(RISCVExtension::I) && cpu.hasExtension(RISCVExtension::M) &&
               cpu.hasExtension(RISCVExtension::A) && cpu.hasExtension(RISCVExtension::F) &&
               cpu.hasExtension(RISCVExtension::D);
    // cpu.hasExtension(RISCVExtension::V) checks only RVV support on the device.
    // To figure out RVV version, we try to execute code with RVV1.0 instructions.
    // If there is no `SEGILL`, the device supports RVV1.0.
    // Otherwise we consider that there is no RVV support
    // [TODO] If needed, support other RVV versions
    case gv:
        return mayiuse(g) && cpu.hasExtension(RISCVExtension::V) && can_compile_rvv100();
    case gv_zvfh:
        return mayiuse(gv) && can_compile_zvfh();
    case isa_all:
        return false;
    case isa_undef:
        return true;
    }
    return false;
}

bool has_zvfh_support() {
    return mayiuse(cpu_isa_t::gv_zvfh);
}

std::string isa2str(cpu_isa_t isa) {
    switch (isa) {
    case cpu_isa_t::isa_undef:
        return "undef";
    case cpu_isa_t::g:
        return "g";
    case cpu_isa_t::gv:
        return "gv";
    case cpu_isa_t::gv_zvfh:
        return "gv_zvfh";
    case cpu_isa_t::isa_all:
        return "all";
    default:
        OPENVINO_THROW("Uknown ISA");
    }
}

}  // namespace ov::intel_cpu::riscv64
