// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define XBYAK64
#define XBYAK_NO_OP_NAMES
#define XBYAK_USE_MMAP_ALLOCATOR
#include <chrono>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#ifdef WIN32
#    include <powrprof.h>
#    pragma comment(lib, "powrprof.lib")
#endif

#include <cmath>
#include <set>
#include "cpu_info.h"

#ifndef WIN32
static const float Hz_IN_GHz = 1e6f;
#else
static const float MHz_IN_GHz = 1e3f;
#endif

using Xbyak::Xmm;
using Xbyak::Ymm;
using Xbyak::Zmm;

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

namespace ov {
namespace intel_cpu {

bool CPUInfo::haveSSE() {
    return have_sse && have_sse2 && have_ssse3 && have_sse4_1 && have_sse4_2;
}

bool CPUInfo::haveAVX() {
    return have_avx && have_avx2;
}

bool CPUInfo::haveAVX512() {
    return have_avx512f;
}

bool CPUInfo::checkIsaSupport(ISA cpu_isa) {
    using namespace Xbyak::util;
    static Cpu cpu;

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
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case ISA::avx512_mic:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) && cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
    case ISA::avx512_mic_4ops:
        return checkIsaSupport(ISA::avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) && cpu.has(Cpu::tAVX512_4VNNIW);
    case ISA::avx512_vnni:
        return cpu.has(Cpu::tAVX512F);//&&
                   //cpu.has(Cpu::tAVX512_VNNI);
    }

    return false;
}

float CPUInfo::calcComputeBlockIPC(Precision precision) {
    const int NUM_LOOP = 16384 * 8;
    const int NUM_INSN = 36;
    const int NUM_ITER = 100;

    Xbyak::CodeGenerator* g = nullptr;
    if (haveAVX512()) {
        auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
            g->vfmadd132ps(Zmm(dst_reg), Zmm(src_reg), Zmm(src_reg));
        };

        g = new Generator<Zmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);

    } else if (haveAVX()) {
        if (precision == Precision::FP32) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vfmadd132ps(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
            };
            g = new Generator<Ymm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        } else if (precision == Precision::INT8) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vpmaddubsw(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
                g->vpmaddwd(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
                g->vpaddd(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
            };
            g = new Generator<Ymm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        } else if (precision == Precision::INT1) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vpxor(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
                g->vandps(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpsrld(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vandnps(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpshufb(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpshufb(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpaddb(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpmaddubsw(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpmaddwd(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
                g->vpaddd(Ymm(dst_reg), Ymm(src_reg), Ymm(dst_reg));
            };
            g = new Generator<Ymm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        }

    } else if (haveSSE()) {
        auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
            g->mulps(Xmm(dst_reg), Xmm(src_reg));
            g->addps(Xmm(dst_reg), Xmm(src_reg));
        };

        g = new Generator<Xmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
    }
    typedef void (*func_t)(void);
    func_t exec = (func_t)g->getCode();

    using clock_type = std::chrono::high_resolution_clock;
    using duration = clock_type::duration;

    float res = 0;
    for (int i = 0; i < NUM_ITER; i++) {
        duration b1 = clock_type::now().time_since_epoch();
        exec();
        duration e1 = clock_type::now().time_since_epoch();

        res = std::max(res, (NUM_INSN * NUM_LOOP) / ((e1.count() - b1.count()) * freqGHz));
    }

    delete g;

    return res;
}

float CPUInfo::getFrequency(const std::string path) {
#ifndef WIN32
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("CPUInfo: unable to open " + path + " file\n");
    }

    std::string freq;
    file >> freq;

    return std::stof(freq) / Hz_IN_GHz;
#else
    return freqGHz;
#endif
}

float CPUInfo::getMaxCPUFreq(size_t core_id) {
#ifndef WIN32
    std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(core_id) + "/cpufreq/scaling_max_freq";
    return getFrequency(path);
#else
    return getFrequency(std::string());
#endif
}

#ifndef WIN32
float CPUInfo::getMinCPUFreq(size_t core_id) {
    std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(core_id) + "/cpufreq/scaling_min_freq";
    return getFrequency(path);
}
#endif

bool CPUInfo::isFrequencyFixed() {
    // Try to detect if CPU frequency wasn't fixed
    for (size_t i = 0; i < cores_per_socket; i++) {
#ifndef WIN32
        if (freqGHz != getMinCPUFreq(i) || freqGHz != getMaxCPUFreq(i)) {
            return false;
        }
#else
        if (freqGHz != currGHz) {
            return false;
        }
#endif
    }

    return true;
}

#ifdef WIN32
static uint32_t getNumPhysicalCores(void) {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = nullptr;
    DWORD bufSize = 0;
    DWORD processorCoreCount = 0;
    DWORD byteOffset = 0;
    DWORD rc = 0;

    // get required buffer size
    GetLogicalProcessorInformation(nullptr, &bufSize);

    DWORD errcode = GetLastError();
    if (ERROR_INSUFFICIENT_BUFFER != errcode) {
        std::string errmsg = std::string("\nError ") + std::to_string(errcode) + std::string("\n");
        throw std::runtime_error(errmsg);
    }

    std::vector<BYTE> buf(bufSize);

    rc = GetLogicalProcessorInformation((PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)buf.data(), &bufSize);
    if (FALSE == rc) {
        std::string errmsg = std::string("\nError ") + std::to_string(GetLastError()) + std::string("\n");
        throw std::runtime_error(errmsg);
    }

    ptr = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)buf.data();

    while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= bufSize) {
        switch (ptr->Relationship) {
        case RelationProcessorCore:
            processorCoreCount++;
            break;

        default:
            break;
        }
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

    return processorCoreCount;
}
#endif

void CPUInfo::init() {
#ifndef WIN32
    auto getFeatureValue = [](std::string& line) -> std::string {
        std::istringstream iss(line);
        std::string res;
        while (std::getline(iss, res, ':')) {}
        return res;
    };

    std::string path = "/proc/cpuinfo";
    std::ifstream cpuinfo(path);
    if (!cpuinfo.is_open()) {
        throw std::runtime_error("CPUInfo: unable to open /proc/cpuinfo file\n");
    }

    std::set<uint32_t> unique_core_ids;
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.find("cpu cores") != std::string::npos) {
            cores_per_socket = static_cast<uint32_t>(std::stoi(getFeatureValue(line)));
        }
        if (line.find("physical id") != std::string::npos) {
            unique_core_ids.insert(static_cast<uint32_t>(std::stoi(getFeatureValue(line))));
        }
    }

    sockets_per_node = static_cast<uint32_t>(unique_core_ids.size());

    cpuinfo.close();
#else
    typedef struct _PROCESSOR_POWER_INFORMATION {
        ULONG Number;
        ULONG MaxMhz;
        ULONG CurrentMhz;
        ULONG MhzLimit;
        ULONG MaxIdleState;
        ULONG CurrentIdleState;
    } PROCESSOR_POWER_INFORMATION, *PPROCESSOR_POWER_INFORMATION;

    // get the number or processors
    SYSTEM_INFO si = {0};
    ::GetSystemInfo(&si);

    // returns num of cores (excluding HT cores)
    cores_per_socket = getNumPhysicalCores();

    // allocate buffer to get info for each processor
    const int size = si.dwNumberOfProcessors * sizeof(PROCESSOR_POWER_INFORMATION);
    std::vector<BYTE> buf(size);

    NTSTATUS status = ::CallNtPowerInformation(ProcessorInformation, nullptr, 0, buf.data(), size);
    if (0 == status) {
        // get processor frequency (only the first core for now)
        PPROCESSOR_POWER_INFORMATION ppi = (PPROCESSOR_POWER_INFORMATION)buf.data();

        freqGHz = ppi->MaxMhz / MHz_IN_GHz;
        currGHz = ppi->CurrentMhz / MHz_IN_GHz;
    } else {
        std::string errmsg = std::string("CallNtPowerInformation failed. Status: ") + std::to_string(status);
        throw std::runtime_error(errmsg);
    }
    // Need to add correct detection of sockets count for win
    sockets_per_node = 1;
#endif
}

CPUInfo::CPUInfo() {
    have_sse = checkIsaSupport(ISA::sse);
    have_sse2 = checkIsaSupport(ISA::sse2);
    have_ssse3 = checkIsaSupport(ISA::ssse3);
    have_sse4_1 = checkIsaSupport(ISA::sse4_1);
    have_sse4_2 = checkIsaSupport(ISA::sse4_2);
    have_avx = checkIsaSupport(ISA::avx);
    have_avx2 = checkIsaSupport(ISA::avx2);
    have_fma = checkIsaSupport(ISA::fma);
    have_avx512f = checkIsaSupport(ISA::avx512_common);
    have_vnni = checkIsaSupport(ISA::avx512_vnni);

    try {
        init();
        freqGHz = getMaxCPUFreq(0);
        if (!isFrequencyFixed()) {
            std::cout << "WARNING: CPU frequency is not fixed. Result may be incorrect. "
                      << "Max frequency (" << freqGHz << "GHz) will be used." << std::endl;
        }
    } catch (std::exception&) {
        throw;
    }
}

float CPUInfo::getPeakGOPSImpl(Precision precision) {
    if (precision != Precision::FP32 && precision != Precision::INT8 && precision != Precision::INT1) {
        throw std::invalid_argument("Get GOPS: Unsupported precision");
    }

    uint32_t data_type_bit_size;
    switch (precision) {
    case Precision::FP32:
        data_type_bit_size = sizeof(float) * 8;
        break;
    case Precision::INT8:
        data_type_bit_size = sizeof(int8_t) * 8;
        break;
    case Precision::INT1:
        data_type_bit_size = 1;
        break;
    default:
        break;
    }

    if (haveAVX512())
        simd_size = 512 / data_type_bit_size;
    else if (haveAVX())
        simd_size = 256 / data_type_bit_size;
    else if (haveSSE())
        simd_size = 128 / data_type_bit_size;
    else
        simd_size = 1;

    operations_per_compute_block = 2 * simd_size;

    instructions_per_cycle = calcComputeBlockIPC(precision);

    return std::round(instructions_per_cycle * operations_per_compute_block) * freqGHz * cores_per_socket *
           sockets_per_node;
}

void CPUInfo::printDetails() {
    std::cout << "ops per compute block:           " << operations_per_compute_block << std::endl;
    std::cout << "IPC of the compute block:        " << instructions_per_cycle << std::endl;
    std::cout << "cycles per second (freq in GHz): " << freqGHz << std::endl;
    std::cout << "cores per socket:                " << cores_per_socket << std::endl;
    std::cout << "sockets count:                   " << sockets_per_node << std::endl;
}
} // namespace intel_cpu
} // namespace ov