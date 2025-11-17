// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gemmv_force_isa.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <string>

namespace ov::intel_cpu::x64::gemmv_jit {

namespace {
std::atomic<gemmv_force_isa_t> g_force_mode{gemmv_force_isa_t::auto_mode};

std::string to_upper(const char* value) {
    if (!value) return {};
    std::string tmp(value);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return tmp;
}
} // namespace

gemmv_force_isa_t get_gemmv_force_isa() {
    return g_force_mode.load(std::memory_order_relaxed);
}

void set_gemmv_force_isa(gemmv_force_isa_t mode) {
    g_force_mode.store(mode, std::memory_order_relaxed);
}

const char* gemmv_force_isa_to_cstr(gemmv_force_isa_t mode) {
    switch (mode) {
    case gemmv_force_isa_t::auto_mode:   return "AUTO";
    case gemmv_force_isa_t::avx512_fp32: return "AVX512_FP32";
    case gemmv_force_isa_t::avx2_fp32:   return "AVX2_FP32";
    case gemmv_force_isa_t::ref_fp32:    return "REF";
    case gemmv_force_isa_t::vnni:        return "VNNI";
    case gemmv_force_isa_t::amx_int8:    return "AMX_INT8";
    case gemmv_force_isa_t::amx_bf16:    return "AMX_BF16";
    default:                             return "AUTO";
    }
}

gemmv_force_isa_t gemmv_force_isa_from_cstr(const char* value) {
    const std::string upper = to_upper(value);
    if (upper.empty() || upper == "AUTO") return gemmv_force_isa_t::auto_mode;
    if (upper == "AVX512_FP32") return gemmv_force_isa_t::avx512_fp32;
    if (upper == "AVX2_FP32")   return gemmv_force_isa_t::avx2_fp32;
    if (upper == "REF")         return gemmv_force_isa_t::ref_fp32;
    if (upper == "VNNI")        return gemmv_force_isa_t::vnni;
    if (upper == "AMX_INT8")    return gemmv_force_isa_t::amx_int8;
    if (upper == "AMX_BF16")    return gemmv_force_isa_t::amx_bf16;
    return gemmv_force_isa_t::auto_mode;
}

} // namespace ov::intel_cpu::x64::gemmv_jit
