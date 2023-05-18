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
#include <powrprof.h>
#pragma comment(lib, "powrprof.lib")
#endif

#include <cmath>
#include <set>

#include "cpu_info.h"

#ifndef WIN32
static const float Hz_IN_GHz = 1e6f;
#else
static const float MHz_IN_GHz = 1e3f;
#endif

namespace ov {
namespace intel_cpu {

float CPUInfo::calcComputeBlockIPC(InferenceEngine::Precision precision) {
    const int NUM_LOOP = 16384 * 8;
    const int NUM_INSN = 36;
    const int NUM_ITER = 100;
    using Xbyak::Tmm;
    using Xbyak::Xmm;
    using Xbyak::Ymm;
    using Xbyak::Zmm;
    Xbyak::CodeGenerator* g = NULL;
    if (haveAMXBF16() || haveAMXINT8()) {
        if (haveAMXBF16() && precision == InferenceEngine::Precision::BF16) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->tdpbf16ps(Tmm(dst_reg), Tmm(src_reg), Tmm(dst_reg));
            };
            g = new Generator<Tmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        } else if (haveAMXINT8() && precision == InferenceEngine::Precision::I8) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->tdpbssd(Tmm(dst_reg), Tmm(src_reg), Tmm(dst_reg));
            };
            g = new Generator<Tmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        }
    } else if (haveAVX512()) {
        if (precision == InferenceEngine::Precision::FP32) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vfmadd132ps(Zmm(dst_reg), Zmm(src_reg), Zmm(src_reg));
            };
            g = new Generator<Zmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        } else if (precision == InferenceEngine::Precision::FP16) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vfmadd132ph(Zmm(dst_reg), Zmm(src_reg), Zmm(src_reg));
            };
            g = new Generator<Zmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        }
    } else if (haveAVX()) {
        if (precision == InferenceEngine::Precision::FP32) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vfmadd132ps(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
            };
            g = new Generator<Ymm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        } else if (precision == InferenceEngine::Precision::I8) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->vpmaddubsw(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
                g->vpmaddwd(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
                g->vpaddd(Ymm(dst_reg), Ymm(src_reg), Ymm(src_reg));
            };
            g = new Generator<Ymm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        } else if (precision == InferenceEngine::Precision::BIN) {
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
    } else if (haveSSE() || haveSSEX()) {
        if (precision == InferenceEngine::Precision::FP32 || precision == InferenceEngine::Precision::FP16) {
            auto gen = [](Xbyak::CodeGenerator* g, int dst_reg, int src_reg) {
                g->mulps(Xmm(dst_reg), Xmm(src_reg));
                g->addps(Xmm(dst_reg), Xmm(src_reg));
            };
            g = new Generator<Xmm, decltype(gen)>(gen, NUM_LOOP, NUM_INSN);
        }
    }

    float res = 0;
    typedef void (*func_t)(void);
    if (g) {
        func_t exec = (func_t)g->getCode();

        using clock_type = std::chrono::high_resolution_clock;
        using duration = clock_type::duration;

        for (int i = 0; i < NUM_ITER; i++) {
            duration b1 = clock_type::now().time_since_epoch();
            exec();
            duration e1 = clock_type::now().time_since_epoch();

            res = std::max(res, (NUM_INSN * NUM_LOOP) / ((e1.count() - b1.count()) * freqGHz));
        }
        delete g;
    }
    return res;
}

float CPUInfo::getFrequency(const std::string path) {
#ifndef WIN32
    std::string freq;
    try {
        std::ifstream file(path);
        file >> freq;
    } catch (std::ios_base::failure& e) {
        throw std::runtime_error("CPUInfo: unable to open " + path + " file: " + std::string(e.what()) + "\n");
    }
    if (freq.empty())
        return freqGHz;
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
        while (std::getline(iss, res, ':')) {
        }
        return res;
    };

    std::string path = "/proc/cpuinfo";
    std::ifstream cpuinfo;
    try {
        cpuinfo.open(path);
    } catch (std::ios_base::failure& e) {
        throw std::runtime_error("CPUInfo: unable to open " + path + " file: " + std::string(e.what()) + "\n");
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

    auto status = ::CallNtPowerInformation(ProcessorInformation, nullptr, 0, buf.data(), size);
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
    ISA_detailed = dnnl::impl::cpu::platform::get_isa_info();
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
            std::cout << "WARNING: CPU frequency is not fixed. Result may be incorrect. \n"
                      << "Max frequency (" << freqGHz << "GHz) will be used." << std::endl;
        }
        std::cout << "Initialize CPU info for calculating GOPS successfully!" << std::endl;
    } catch (std::exception& e) {
        std::string msg{e.what()};
        IE_THROW() << "Failed to initialize CPU info for calculating GOPS: " << msg;
    }
}

float CPUInfo::getPeakGOPSImpl(InferenceEngine::Precision precision) {
    uint32_t data_type_bit_size = 1;
    switch (precision) {
    case InferenceEngine::Precision::FP32:
        data_type_bit_size = sizeof(float) * 8;
        break;
    case InferenceEngine::Precision::FP16:
        data_type_bit_size = sizeof(float) * 8 * 2;
        break;
    case InferenceEngine::Precision::I8:
        data_type_bit_size = sizeof(int8_t) * 8;
        break;
    case InferenceEngine::Precision::BIN:
        data_type_bit_size = 1;
        break;
    default:
        throw std::invalid_argument("Get GOPS: Unsupported precision");
        break;
    }

    simd_size = 1;
    if (haveAMXBF16() || haveAMXINT8())
        simd_size = 1024 / data_type_bit_size;
    if (haveAVX512())
        simd_size = 512 / data_type_bit_size;
    if (haveAVX())
        simd_size = 256 / data_type_bit_size;
    if (haveSSE())
        simd_size = 128 / data_type_bit_size;
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
    std::cout << "ISA information:                 " << ISA_detailed << std::endl;
}
}  // namespace intel_cpu
}  // namespace ov