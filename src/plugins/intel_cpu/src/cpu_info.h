// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <cpu/x64/xbyak/xbyak_util.h>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/amx_tile_configure.hpp>
using Xbyak::Xmm;
using Xbyak::Ymm;
using Xbyak::Zmm;
using Xbyak::Tmm;
using Xbyak::util::Cpu;
#include <cpu/platform.hpp>

#include "ie_precision.hpp"

namespace ov {
namespace intel_cpu {
class CPUInfo {
public:
    CPUInfo();
    float getPeakGOPSImpl(InferenceEngine::Precision precision);
    void printDetails();

private:
    enum class ISA {
        sse,
        sse2,
        sse3,
        ssse3,
        sse4_1,
        sse4_2,
        avx,
        avx2,
        fma,
        avx512_common,
        avx512_core,
        avx512_mic,
        avx512_mic_4ops,
        avx512_vnni,
        amx_int8,
        amx_bf16,
    };

    void init();
    bool checkIsaSupport(ISA cpu_isa) {
        switch (cpu_isa) {
        case ISA::sse:
            return cpu.has(Cpu::tSSE);
        case ISA::sse2:
            return cpu.has(Cpu::tSSE2);
        case ISA::sse3:
            return cpu.has(Cpu::tSSE3);
        case ISA::ssse3:
            return cpu.has(Cpu::tSSSE3);
        case ISA::sse4_1:
            return cpu.has(Cpu::tSSE41);
        case ISA::sse4_2:
            return cpu.has(Cpu::tSSE42);
        case ISA::avx:
            return cpu.has(Cpu::tAVX);
        case ISA::avx2:
            return cpu.has(Cpu::tAVX2);
        case ISA::fma:
            return cpu.has(Cpu::tFMA);
        case ISA::avx512_common:
            return cpu.has(Cpu::tAVX512F);
        case ISA::avx512_core:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) &&
                   cpu.has(Cpu::tAVX512DQ);
        case ISA::avx512_mic:
            return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) && cpu.has(Cpu::tAVX512ER) &&
                   cpu.has(Cpu::tAVX512PF);
        case ISA::avx512_mic_4ops:
            return checkIsaSupport(ISA::avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) && cpu.has(Cpu::tAVX512_4VNNIW);
        case ISA::avx512_vnni:
            return cpu.has(Cpu::tAVX512F);  //&&
                                            // cpu.has(Cpu::tAVX512_VNNI);
        case ISA::amx_bf16:
            return cpu.has(Cpu::tAMX_BF16);
        case ISA::amx_int8:
            return cpu.has(Cpu::tAMX_INT8);
        }

        return false;
    }
    bool haveSSE() {
        return have_sse;
    }
    bool haveSSEX() {
        return have_sse2 && have_ssse3 && have_sse4_1 && have_sse4_2;
    }
    bool haveAVX() {
        return have_avx && have_avx2;
    }
    bool haveAVX512() {
        return have_avx512f;
    }
    bool haveAMXBF16() {
        return have_amx_bf16;
    }
    bool haveAMXINT8() {
        return have_amx_int8;
    }

    float calcComputeBlockIPC(InferenceEngine::Precision precision);

    float getFrequency(const std::string path);
    float getMaxCPUFreq(size_t core_id);
    float getMinCPUFreq(size_t core_id);
    bool isFrequencyFixed();

#ifdef WIN32
    float currGHz = 1.0f;
#endif

    bool have_sse = false;
    bool have_sse2 = false;
    bool have_ssse3 = false;
    bool have_sse4_1 = false;
    bool have_sse4_2 = false;
    bool have_avx = false;
    bool have_avx2 = false;
    bool have_fma = false;
    bool have_avx512f = false;
    bool have_vnni = false;
    bool have_amx_int8 = false;
    bool have_amx_bf16 = false;

    // Micro architecture level
    uint32_t simd_size = 1;
    float instructions_per_cycle = 1.0f;
    uint32_t operations_per_compute_block = 1;

    // Machine architecture level
    float freqGHz = 1.0f;
    uint32_t cores_per_socket = 1;
    uint32_t sockets_per_node = 1;
    std::string ISA_detailed;
    Xbyak::util::Cpu cpu;
};

template <typename T>
struct RegMap;

template <>
struct RegMap<Xbyak::Xmm> {
    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        g->movaps(g->ptr[g->rsp + off], Xmm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        g->movaps(Xmm(idx), g->ptr[g->rsp + off]);
    }

    void killdep(Xbyak::CodeGenerator* g, int idx) {
        g->xorps(Xmm(idx), Xmm(idx));
    }
};

template <>
struct RegMap<Xbyak::Ymm> {
    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(g->ptr[g->rsp + off], Ymm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(Ymm(idx), g->ptr[g->rsp + off]);
    }
    void killdep(Xbyak::CodeGenerator* g, int idx) {
        g->vxorps(Ymm(idx), Ymm(idx), Ymm(idx));
    }
};

template <>
struct RegMap<Xbyak::Zmm> {
    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(g->ptr[g->rsp + off], Zmm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        g->vmovaps(Zmm(idx), g->ptr[g->rsp + off]);
    }

    void killdep(Xbyak::CodeGenerator* g, int idx) {
        g->vpxorq(Zmm(idx), Zmm(idx), Zmm(idx));
    }
};

template <>
struct RegMap<Xbyak::Tmm> {
    RegMap<Xbyak::Tmm>() {
        dnnl::impl::cpu::x64::palette_config_t tconf = {0};
        auto palette_id = dnnl::impl::cpu::x64::amx::get_target_palette();
        std::cout << "[WangYang] AMX palette ID: " << palette_id << std::endl;
        tconf.palette_id = palette_id;
        for (int index = 0; index < sizeof(tconf.rows) / sizeof(tconf.rows[0]); index++) {
            tconf.rows[index] = dnnl::impl::cpu::x64::amx::get_max_rows(tconf.palette_id);
            std::cout << "[WangYang] Max rows per register: " << tconf.rows[index] << std::endl;
        }
        for (int index = 0; index < sizeof(tconf.cols) / sizeof(tconf.cols[0]); index++) {
            tconf.cols[index] = dnnl::impl::cpu::x64::amx::get_max_column_bytes(tconf.palette_id);
            std::cout << "[WangYang] Max number of bytes per rows: " << tconf.cols[index] << std::endl;
        }
        // configura AMX tiles
        dnnl::impl::cpu::x64::amx_tile_configure((const char*)&tconf);
    }

    ~RegMap<Xbyak::Tmm>() {
        dnnl::impl::cpu::x64::amx_tile_release();
    }

    void save(Xbyak::CodeGenerator* g, int idx, int off) {
        // No need to save AMX registers here and juse clean registers.
        auto palette_id = dnnl::impl::cpu::x64::amx::get_target_palette();
        idx = idx % dnnl::impl::cpu::x64::amx::get_max_tiles(palette_id);
        g->tilezero(Tmm(idx));
    }

    void restore(Xbyak::CodeGenerator* g, int idx, int off) {
        // No need to restore AMX registers here and juse clean registers.
        auto palette_id = dnnl::impl::cpu::x64::amx::get_target_palette();
        idx = idx % dnnl::impl::cpu::x64::amx::get_max_tiles(palette_id);
        g->tilezero(Tmm(idx));
    }

    void killdep(Xbyak::CodeGenerator* g, int idx) {
        return;
    }
};

template <typename RegType, typename Gen, typename F>
struct ThroughputGenerator {
    void operator()(Gen* g, RegMap<RegType>& rm, F f, int num_insn) {
        for (int j = 0; j < num_insn / 12; j++)
            for (int i = 0; i < 12; i++)
                f(g, 4 + i, 4 + i);
    }
};

template <typename RegType, typename F>
struct Generator : public Xbyak::CodeGenerator {
    Generator(F f, int num_loop, int num_insn) {
        RegMap<RegType> rm;

        int reg_size = 64;
        int num_reg = 12;

        push(rbp);
        mov(rbp, rsp);
        and_(rsp, -(Xbyak::sint64)64);
        sub(rsp, reg_size * (num_reg + 1));

        for (int i = 0; i < num_reg; i++)
            rm.save(this, 4 + i, -reg_size * (12 - i));

        for (int i = 0; i < num_reg; i++)
            rm.killdep(this, 4 + i);

        mov(rcx, num_loop);

        align(16);
        L("@@");
        ThroughputGenerator<RegType, Generator, F>()(this, rm, f, num_insn);
        dec(rcx);
        jnz("@b");

        for (int i = 0; i < num_reg; i++)
            rm.restore(this, 4 + i, -reg_size * (12 - i));

        mov(rsp, rbp);
        pop(rbp);
        ret();
    }
};
}  // namespace intel_cpu
}  // namespace ov