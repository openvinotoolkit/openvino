/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef TEST_ISA_COMMON_HPP
#define TEST_ISA_COMMON_HPP

#include <cassert>
#include <map>
#include <set>
#include <utility>
#include <type_traits>

// prevent earlier inclusion of dnnl_threads
#include "test_thread.hpp"

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"

#include "src/cpu/platform.hpp"

#if DNNL_X64
#include "src/cpu/x64/cpu_isa_traits.hpp"
#endif

namespace dnnl {

#if DNNL_X64

inline impl::cpu::x64::cpu_isa_t cvt_to_internal_cpu_isa(cpu_isa input_isa) {
#define HANDLE_ISA(isa) \
    case cpu_isa::isa: return impl::cpu::x64::cpu_isa_t::isa; break

    switch (input_isa) {
        HANDLE_ISA(sse41);
        HANDLE_ISA(avx);
        HANDLE_ISA(avx2);
        HANDLE_ISA(avx512_mic);
        HANDLE_ISA(avx512_mic_4ops);
        HANDLE_ISA(avx512_core);
        HANDLE_ISA(avx512_core_vnni);
        HANDLE_ISA(avx512_core_bf16);
        HANDLE_ISA(avx512_core_amx);
        HANDLE_ISA(avx2_vnni);
        default:
            assert(input_isa == cpu_isa::all);
            return impl::cpu::x64::cpu_isa_t::isa_all;
            break;
    }
#undef HANDLE_ISA
}

// There is no 1-1 correspondence between cpu_isa and internal cpu_isa_t
// In particular, individual cpu_isa can correspond to more than one cpu_isa_t
// distinguished by varying combination of feature bits.
//
// Note that, for two cpu_isa namely, isa_1 and isa_2 such that isa_1 <= isa_2
// the masked_internal_cpu_isa list corresponding to isa_1 will be disjoint from
// the masked_internal_cpu_isa list for isa_2. This is done so as to avoid
// the unnecessary duplication.
//
// Moreover, by default we don't differentiate internal cpu_isa_t according to
// the CPU ISA hints
inline std::set<impl::cpu::x64::cpu_isa_t> masked_internal_cpu_isa(
        cpu_isa isa) {
    using namespace impl::cpu::x64;
    cpu_isa_t internal_isa = cvt_to_internal_cpu_isa(isa);

    static const std::set<cpu_isa_t> amx_isa_list {avx512_core_amx,
            avx512_core_bf16_amx_bf16, avx512_core_bf16_amx_int8, amx_bf16,
            amx_int8, amx_tile};

    switch (internal_isa) {
        case avx512_mic: return {avx512_mic, avx512_common}; break;
        case avx512_core: return {avx512_core, avx512_common}; break;
        case avx512_core_amx: return amx_isa_list; break;
        default: return {internal_isa}; break;
    }
}

inline std::set<std::pair<impl::cpu::x64::cpu_isa_t, impl::cpu::x64::cpu_isa_t>>
hints_masked_internal_cpu_isa(cpu_isa_hints hints) {
    using namespace impl::cpu::x64;
    using mask_pair = std::pair<cpu_isa_t, cpu_isa_t>;

    switch (hints) {
        case cpu_isa_hints::no_hints: return std::set<mask_pair> {}; break;
        case cpu_isa_hints::prefer_ymm:
            return std::set<mask_pair> {
                    {avx512_core_bf16, avx512_core_bf16_ymm}};
            break;
        default:
            assert(!"unknown CPU ISA hint");
            return std::set<mask_pair> {};
            break;
    }
}

inline const std::set<cpu_isa> &cpu_isa_all() {
    static const std::set<cpu_isa> isa_all {cpu_isa::sse41, cpu_isa::avx,
            cpu_isa::avx2, cpu_isa::avx2_vnni, cpu_isa::avx512_mic,
            cpu_isa::avx512_mic_4ops, cpu_isa::avx512_core,
            cpu_isa::avx512_core_vnni, cpu_isa::avx512_core_bf16,
            cpu_isa::avx512_core_amx, cpu_isa::all};

    return isa_all;
}

inline const std::set<cpu_isa> &compatible_cpu_isa(cpu_isa input_isa) {
    // Do not use internal `is_superset` routine as this is used to verify
    // the correctness of cpu_isa_traits routines
    static const std::map<cpu_isa, const std::set<cpu_isa>> isa_cmpt_info {
            {cpu_isa::sse41, {cpu_isa::sse41}},
            {cpu_isa::avx, {cpu_isa::avx, cpu_isa::sse41}},
            {cpu_isa::avx2, {cpu_isa::avx2, cpu_isa::avx, cpu_isa::sse41}},
            {cpu_isa::avx2_vnni,
                    {cpu_isa::avx2_vnni, cpu_isa::avx2, cpu_isa::avx,
                            cpu_isa::sse41}},
            {cpu_isa::avx512_mic,
                    {cpu_isa::avx512_mic, cpu_isa::avx2, cpu_isa::avx,
                            cpu_isa::sse41}},
            {cpu_isa::avx512_mic_4ops,
                    {cpu_isa::avx512_mic_4ops, cpu_isa::avx512_mic,
                            cpu_isa::avx2, cpu_isa::avx, cpu_isa::sse41}},
            {cpu_isa::avx512_core,
                    {cpu_isa::avx512_core, cpu_isa::avx2, cpu_isa::avx,
                            cpu_isa::sse41}},
            {cpu_isa::avx512_core_vnni,
                    {cpu_isa::avx512_core_vnni, cpu_isa::avx512_core,
                            cpu_isa::avx2, cpu_isa::avx, cpu_isa::sse41}},
            {cpu_isa::avx512_core_bf16,
                    {cpu_isa::avx512_core_bf16, cpu_isa::avx512_core_vnni,
                            cpu_isa::avx512_core, cpu_isa::avx2, cpu_isa::avx,
                            cpu_isa::sse41}},
            {cpu_isa::avx512_core_amx,
                    {cpu_isa::avx512_core_amx, cpu_isa::avx512_core_bf16,
                            cpu_isa::avx512_core_vnni, cpu_isa::avx512_core,
                            cpu_isa::avx2, cpu_isa::avx, cpu_isa::sse41}},
            {cpu_isa::all,
                    {cpu_isa::all, cpu_isa::avx512_core_amx,
                            cpu_isa::avx512_core_bf16,
                            cpu_isa::avx512_core_vnni, cpu_isa::avx512_core,
                            cpu_isa::avx512_mic_4ops, cpu_isa::avx512_mic,
                            cpu_isa::avx2_vnni, cpu_isa::avx2, cpu_isa::avx,
                            cpu_isa::sse41}}};

    auto iter = isa_cmpt_info.find(input_isa);
    assert(iter != isa_cmpt_info.end());
    return iter->second;
}

inline bool is_superset(cpu_isa isa_1, cpu_isa isa_2) {
    const auto &cmpt_list = compatible_cpu_isa(isa_1);
    return cmpt_list.find(isa_2) != cmpt_list.end();
}

inline bool is_superset(dnnl_cpu_isa_t isa_1, dnnl_cpu_isa_t isa_2) {
    return is_superset(
            static_cast<cpu_isa>(isa_1), static_cast<cpu_isa>(isa_2));
}

inline bool mayiuse(impl::cpu::x64::cpu_isa_t isa, bool soft = false) {
    return impl::cpu::x64::mayiuse(isa, soft);
}

inline bool mayiuse(cpu_isa isa, bool soft = false) {
    return mayiuse(cvt_to_internal_cpu_isa(isa), soft);
}

inline bool mayiuse(dnnl_cpu_isa_t isa, bool soft = false) {
    return mayiuse(static_cast<cpu_isa>(isa), soft);
}

inline cpu_isa get_max_cpu_isa(bool soft = false) {
    for (auto it = cpu_isa_all().crbegin(); it != cpu_isa_all().crend(); it++) {
        if (mayiuse(*it, soft)) return *it;
    }

    return cpu_isa::all;
}

inline impl::cpu::x64::cpu_isa_t get_max_cpu_isa_mask(bool soft = false) {
    return impl::cpu::x64::get_max_cpu_isa_mask(soft);
}

#else

inline bool is_superset(dnnl_cpu_isa_t isa_1, dnnl_cpu_isa_t isa_2) {
    return false;
}

#endif

} // namespace dnnl
#endif
