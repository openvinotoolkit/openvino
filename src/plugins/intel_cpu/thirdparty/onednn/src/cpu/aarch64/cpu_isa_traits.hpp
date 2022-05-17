/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_CPU_ISA_TRAITS_HPP
#define CPU_AARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR

#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h"
#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

/* The following enum is temporal implementation.
   It should be made in dnnl_types.h,
   but an RFC is requird to modify dnnl_types.h.
   The following values are used with
   static_cast<dnnl_cpu_isa_t>, the same values
   defined in dnnl_cpu_isa_t are temporaly used. */
/// CPU instruction set flags
enum {
    /// AARCH64 Advanced SIMD & floating-point
    dnnl_cpu_isa_asimd = 0x1,
    /// AARCH64 SVE 128 bits
    dnnl_cpu_isa_sve_128 = 0x3,
    /// AARCH64 SVE 256 bits
    dnnl_cpu_isa_sve_256 = 0x7,
    /// AARCH64 SVE 384 bits
    dnnl_cpu_isa_sve_384 = 0xf,
    /// AARCH64 SVE 512 bits
    dnnl_cpu_isa_sve_512 = 0x1f,
};

enum cpu_isa_bit_t : unsigned {
    asimd_bit = 1u << 0,
    sve_128_bit = 1u << 1,
    sve_256_bit = 1u << 2,
    sve_384_bit = 1u << 3,
    sve_512_bit = 1u << 4,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    asimd = asimd_bit,
    sve_128 = sve_128_bit | asimd,
    sve_256 = sve_256_bit | asimd,
    sve_384 = sve_384_bit | asimd,
    sve_512 = sve_512_bit | asimd,
    isa_all = ~0u,
};

const char *get_isa_info();

cpu_isa_t get_max_cpu_isa();
cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<asimd> {
    typedef Xbyak_aarch64::VReg TReg;
    typedef Xbyak_aarch64::VReg16B TRegB;
    typedef Xbyak_aarch64::VReg8H TRegH;
    typedef Xbyak_aarch64::VReg4S TRegS;
    typedef Xbyak_aarch64::VReg2D TRegD;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_asimd);
    static constexpr const char *user_option_env = "ADVANCED_SIMD";
};

template <>
struct cpu_isa_traits<sve_512> {
    typedef Xbyak_aarch64::ZReg TReg;
    typedef Xbyak_aarch64::ZRegB TRegB;
    typedef Xbyak_aarch64::ZRegH TRegH;
    typedef Xbyak_aarch64::ZRegS TRegS;
    typedef Xbyak_aarch64::ZRegD TRegD;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_512);
    static constexpr const char *user_option_env = "SVE_512";
};

inline const Xbyak_aarch64::util::Cpu &cpu() {
    const static Xbyak_aarch64::util::Cpu cpu_;
    return cpu_;
}

namespace {

static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak_aarch64::util;

    unsigned cpu_isa_mask = aarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case asimd: return cpu().has(Cpu::tADVSIMD);
        case sve_128:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_128;
        case sve_256:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_256;
        case sve_384:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_384;
        case sve_512:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_512;
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

static inline uint64_t get_sve_length() {
    return cpu().getSveLen();
}

static inline bool mayiuse_atomic() {
    using namespace Xbyak_aarch64::util;
    return cpu().isAtomicSupported();
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return false;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == asimd ? prefix STRINGIFY(asimd) : \
    ((isa) == sve_512 ? prefix STRINGIFY(sve_512) : \
    prefix suffix_if_any)))
/* clang-format on */

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
