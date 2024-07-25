// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {

enum impl_desc_type : int64_t {
    unknown = 0x00000000,
    undef,
    // Optimization approach
    simple  = 1<<6,
    ref     = 1<<7,
    jit     = 1<<8,
    gemm    = 1<<9,
    brgconv = 1<<10,
    brgemm  = 1<<11,
    // CPU version
    sse42  = 1<<12,
    avx    = 1<<13,
    avx2   = 1<<14,
    avx512 = 1<<15,
    amx    = 1<<16,
    blas   = 1<<17,
    any    = 1<<18,
    uni    = 1<<19,
    acl    = 1<<20,
    // Other specificator
    _1x1    = 1<<21,
    _dw     = 1<<22,
    // Other info
    reorder = 1<<23,
    // winograd
    winograd = 1<<24,
    // sparse
    sparse = 1<<25,
    //mlas backend
    mlas = 1<<26,

    asimd  = 1<<27,
    sve128 = 1<<28,
    sve256 = 1<<29,
    sve384 = 1<<30,
    sve512 = 1<<31,

    // shl backend
    shl = 1ll<<32,

    // real types
    ref_any             = ref  | any,

    gemm_any            = gemm | any,
    gemm_blas           = gemm | blas,
    gemm_avx512         = gemm | avx512,
    gemm_avx2           = gemm | avx2,
    gemm_avx            = gemm | avx,
    gemm_sse42          = gemm | sse42,
    jit_gemm            = jit | gemm,

    jit_avx512_winograd = jit  | avx512 | winograd,
    jit_avx512          = jit  | avx512,
    jit_avx2            = jit  | avx2,
    jit_avx             = jit  | avx,
    jit_sse42           = jit  | sse42,
    jit_uni             = jit  | uni,
    jit_avx512_amx      = jit  | avx512 | amx,

    jit_avx512_1x1      = jit  | avx512 | _1x1,
    jit_avx2_1x1        = jit  | avx2   | _1x1,
    jit_avx_1x1         = jit  | avx    | _1x1,
    jit_sse42_1x1       = jit  | sse42  | _1x1,
    jit_uni_1x1         = jit  | uni    | _1x1,
    jit_avx512_amx_1x1  = jit  | avx512 | amx | _1x1,

    jit_avx512_dw       = jit  | avx512 | _dw,
    jit_avx2_dw         = jit  | avx2   | _dw,
    jit_avx_dw          = jit  | avx    | _dw,
    jit_sse42_dw        = jit  | sse42  | _dw,
    jit_uni_dw          = jit  | uni    | _dw,
    jit_avx512_amx_dw   = jit  | avx512 | amx | _dw,

    brgconv_avx512      = brgconv  | avx512,
    brgconv_avx2        = brgconv  | avx2,
    brgconv_avx         = brgconv  | avx,
    brgconv_sse42       = brgconv  | sse42,
    brgconv_uni         = brgconv  | uni,
    brgconv_avx512_amx  = brgconv  | avx512 | amx,

    brgconv_avx512_1x1      = brgconv  | avx512 | _1x1,
    brgconv_avx2_1x1        = brgconv  | avx2 | _1x1,
    brgconv_avx_1x1         = brgconv  | avx | _1x1,
    brgconv_sse42_1x1       = brgconv  | sse42 | _1x1,
    brgconv_uni_1x1         = brgconv  | uni | _1x1,
    brgconv_avx512_amx_1x1  = brgconv  | avx512 | amx | _1x1,

    brgemm_avx512      = brgemm  | avx512,
    brgemm_avx2        = brgemm  | avx2,
    brgemm_avx         = brgemm  | avx,
    brgemm_sse42       = brgemm  | sse42,
    brgemm_uni         = brgemm  | uni,
    brgemm_avx512_amx  = brgemm  | avx512 | amx,
    brgemm_sparse_avx512_amx = brgemm | sparse | avx512 | amx,

    dw_acl             = _dw | acl,
    gemm_acl           = gemm | acl,
    winograd_acl       = winograd | acl,
    gemm_mlas          = gemm | mlas,

    jit_asimd          = jit | asimd,
    jit_sve128        = jit | sve128,
    jit_sve256        = jit | sve256,
    jit_sve384        = jit | sve384,
    jit_sve512        = jit | sve512,

    gemm_shl          = gemm | shl
};

std::vector<std::string> extractTypeAndImplName(const std::string& priority);
const char * impl_type_to_string(impl_desc_type type);
impl_desc_type parse_impl_name(std::string impl_desc_name);
bool contains(const std::vector<impl_desc_type>& priorities, const impl_desc_type impl_type_str);

}   // namespace intel_cpu
}   // namespace ov
