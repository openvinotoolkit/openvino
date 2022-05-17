/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstring>
#include <mutex>

#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
#ifdef DNNL_ENABLE_MAX_CPU_ISA
cpu_isa_t init_max_cpu_isa() {
    cpu_isa_t max_cpu_isa_val = isa_all;
    char buf[64];
    if (getenv("DNNL_MAX_CPU_ISA", buf, sizeof(buf)) > 0) {

#define IF_HANDLE_CASE(cpu_isa) \
    if (std::strcmp(buf, cpu_isa_traits<cpu_isa>::user_option_env) == 0) \
    max_cpu_isa_val = cpu_isa
#define ELSEIF_HANDLE_CASE(cpu_isa) else IF_HANDLE_CASE(cpu_isa)

        IF_HANDLE_CASE(isa_all);
        ELSEIF_HANDLE_CASE(sse41);
        ELSEIF_HANDLE_CASE(avx);
        ELSEIF_HANDLE_CASE(avx2);
        ELSEIF_HANDLE_CASE(avx2_vnni);
        ELSEIF_HANDLE_CASE(avx512_mic);
        ELSEIF_HANDLE_CASE(avx512_mic_4ops);
        ELSEIF_HANDLE_CASE(avx512_core);
        ELSEIF_HANDLE_CASE(avx512_core_vnni);
        ELSEIF_HANDLE_CASE(avx512_core_bf16);
        ELSEIF_HANDLE_CASE(avx512_core_amx);

#undef IF_HANDLE_CASE
#undef ELSEIF_HANDLE_CASE
    }
    return max_cpu_isa_val;
}

set_once_before_first_get_setting_t<cpu_isa_t> &max_cpu_isa() {
    static set_once_before_first_get_setting_t<cpu_isa_t> max_cpu_isa_setting(
            init_max_cpu_isa());
    return max_cpu_isa_setting;
}
#endif

#ifdef DNNL_ENABLE_CPU_ISA_HINTS
dnnl_cpu_isa_hints_t init_cpu_isa_hints() {
    dnnl_cpu_isa_hints_t cpu_isa_hints_val = dnnl_cpu_isa_no_hints;
    char buf[64];
    if (getenv("DNNL_CPU_ISA_HINTS", buf, sizeof(buf)) > 0) {
        if (std::strcmp(buf, "PREFER_YMM") == 0)
            cpu_isa_hints_val = dnnl_cpu_isa_prefer_ymm;
    }
    return cpu_isa_hints_val;
}

set_once_before_first_get_setting_t<dnnl_cpu_isa_hints_t> &cpu_isa_hints() {
    static set_once_before_first_get_setting_t<dnnl_cpu_isa_hints_t>
            cpu_isa_hints_setting(init_cpu_isa_hints());
    return cpu_isa_hints_setting;
}
#endif
} // namespace

struct isa_info_t {
    isa_info_t(cpu_isa_t aisa) : isa(aisa) {};

    // this converter is needed as code base defines certain ISAs
    // that the library does not expose (eg avx512_common),
    // so the internal and external enum types do not coincide.
    dnnl_cpu_isa_t convert_to_public_enum(void) const {
        switch (isa) {
            case avx512_core_amx: return dnnl_cpu_isa_avx512_core_amx;
            case avx512_core_bf16_amx_bf16: // fallback to avx512_core_bf16
            case avx512_core_bf16_amx_int8: // fallback to avx512_core_bf16
            case avx512_core_bf16_ymm: // fallback to avx512_core_bf16
            case avx512_core_bf16: return dnnl_cpu_isa_avx512_core_bf16;
            case avx512_core_vnni: return dnnl_cpu_isa_avx512_core_vnni;
            case avx512_core: return dnnl_cpu_isa_avx512_core;
            case avx512_mic_4ops: return dnnl_cpu_isa_avx512_mic_4ops;
            case avx512_mic: return dnnl_cpu_isa_avx512_mic;
            case avx2_vnni: return dnnl_cpu_isa_avx2_vnni;
            case avx2: return dnnl_cpu_isa_avx2;
            case avx: return dnnl_cpu_isa_avx;
            case sse41: return dnnl_cpu_isa_sse41;
            default: return dnnl_cpu_isa_all;
        }
    }

    const char *get_name() const {
        switch (isa) {
            case avx512_core_amx:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support "
                       "and Intel AMX with bfloat16 and 8-bit integer support";
            case avx512_core_bf16_amx_bf16:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support "
                       "and Intel AMX with bfloat16 support";
            case avx512_core_bf16_amx_int8:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support "
                       "and Intel AMX with 8-bit integer support";
            case avx512_core_bf16_ymm:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support "
                       "on Ymm/Zmm";
            case avx512_core_bf16:
                return "Intel AVX-512 with Intel DL Boost and bfloat16 support";
            case avx512_core_vnni: return "Intel AVX-512 with Intel DL Boost";
            case avx512_core:
                return "Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ "
                       "extensions";
            case avx512_mic_4ops:
                return "Intel AVX-512 with AVX512_4FMAPS and AVX512_4VNNIW "
                       "extensions";
            case avx512_mic:
                return "Intel AVX-512 with AVX512CD, AVX512ER, and AVX512PF "
                       "extensions";
            case avx512_common: return "Intel AVX-512";
            case avx2_vnni: return "Intel AVX2 with Intel DL Boost";
            case avx2: return "Intel AVX2";
            case avx: return "Intel AVX";
            case sse41: return "Intel SSE4.1";
            default: return "Intel 64";
        }
    }

    cpu_isa_t isa;
};

static isa_info_t get_isa_info_t(void) {
    // descending order due to mayiuse check
#define HANDLE_CASE(cpu_isa) \
    if (mayiuse(cpu_isa)) return isa_info_t(cpu_isa);
    HANDLE_CASE(avx512_core_amx);
    HANDLE_CASE(avx512_core_bf16_amx_bf16);
    HANDLE_CASE(avx512_core_bf16_amx_int8);
    HANDLE_CASE(avx512_core_bf16_ymm);
    HANDLE_CASE(avx512_core_bf16);
    HANDLE_CASE(avx512_core_vnni);
    HANDLE_CASE(avx512_core);
    HANDLE_CASE(avx512_mic_4ops);
    HANDLE_CASE(avx512_mic);
    HANDLE_CASE(avx512_common);
    HANDLE_CASE(avx2_vnni);
    HANDLE_CASE(avx2);
    HANDLE_CASE(avx);
    HANDLE_CASE(sse41);
#undef HANDLE_CASE
    return isa_info_t(isa_any);
}

const char *get_isa_info() {
    return get_isa_info_t().get_name();
}

cpu_isa_t get_max_cpu_isa() {
    return get_isa_info_t().isa;
}

cpu_isa_t get_max_cpu_isa_mask(bool soft) {
    MAYBE_UNUSED(soft);
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    return max_cpu_isa().get(soft);
#else
    return isa_all;
#endif
}

dnnl_cpu_isa_hints_t get_cpu_isa_hints(bool soft) {
    MAYBE_UNUSED(soft);
#ifdef DNNL_ENABLE_CPU_ISA_HINTS
    return cpu_isa_hints().get(soft);
#else
    return dnnl_cpu_isa_no_hints;
#endif
}

status_t set_max_cpu_isa(dnnl_cpu_isa_t isa) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    cpu_isa_t isa_to_set = isa_any;
#define HANDLE_CASE(cpu_isa) \
    case cpu_isa_traits<cpu_isa>::user_option_val: isa_to_set = cpu_isa; break;
    switch (isa) {
        HANDLE_CASE(isa_all);
        HANDLE_CASE(sse41);
        HANDLE_CASE(avx);
        HANDLE_CASE(avx2);
        HANDLE_CASE(avx2_vnni);
        HANDLE_CASE(avx512_mic);
        HANDLE_CASE(avx512_mic_4ops);
        HANDLE_CASE(avx512_core);
        HANDLE_CASE(avx512_core_vnni);
        HANDLE_CASE(avx512_core_bf16);
        HANDLE_CASE(avx512_core_amx);
        default: return invalid_arguments;
    }
    assert(isa_to_set != isa_any);
#undef HANDLE_CASE

    if (max_cpu_isa().set(isa_to_set))
        return success;
    else
        return invalid_arguments;
#else
    return unimplemented;
#endif
}

dnnl_cpu_isa_t get_effective_cpu_isa() {
    return get_isa_info_t().convert_to_public_enum();
}

status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_CPU_ISA_HINTS
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    if (cpu_isa_hints().set(isa_hints))
        return success;
    else
        return runtime_error;
#else
    return unimplemented;
#endif
}

namespace amx {

int get_max_palette() {
    if (mayiuse(amx_tile)) {
        unsigned int data[4] = {};
        const unsigned int &EAX = data[0];
        Xbyak::util::Cpu::getCpuidEx(0x1D, 0, data);
        return EAX;
    } else {
        return 0;
    }
}

int get_max_tiles(int palette) {
    if (mayiuse(amx_tile)) {
        if (palette > get_max_palette() || palette <= 0) return -1;

        unsigned int data[4] = {};
        const unsigned int &EBX = data[1];
        Xbyak::util::Cpu::getCpuidEx(0x1D, palette, data);

        return EBX >> 16;
    } else {
        return 0;
    }
}

int get_max_column_bytes(int palette) {
    if (mayiuse(amx_tile)) {
        if (palette > get_max_palette() || palette <= 0) return -1;

        unsigned int data[4] = {};
        const unsigned int &EBX = data[1];
        Xbyak::util::Cpu::getCpuidEx(0x1D, palette, data);

        return (EBX << 16) >> 16;
    } else {
        return 0;
    }
}

int get_max_rows(int palette) {
    if (mayiuse(amx_tile)) {
        if (palette > get_max_palette() || palette <= 0) return -1;

        unsigned int data[4] = {};
        const unsigned int &ECX = data[2];
        Xbyak::util::Cpu::getCpuidEx(0x1D, palette, data);

        return (ECX << 16) >> 16;
    } else {
        return 0;
    }
}

} // namespace amx

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
