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

#include "cpu/cpu_engine.hpp"

#include "cpu/gemm_inner_product.hpp"
#include "cpu/gemm_x8s8s32x_inner_product.hpp"
#include "cpu/ref_inner_product.hpp"
#include "cpu/ref_inner_product_int8.hpp"

#if DNNL_X64
#include "cpu/x64/gemm_bf16_inner_product.hpp"
#include "cpu/x64/jit_brgemm_inner_product.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

#if DNNL_AARCH64 && DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/acl_inner_product.hpp"
using namespace dnnl::impl::cpu::aarch64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;

struct ip_impl_key_t {
    prop_kind_t kind;
    data_type_t src_dt, wei_dt, dst_dt;

    bool operator<(const ip_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    enum { MAX_DT_NUM = 10 };
    size_t value() const {
        return (((size_t)kind * MAX_DT_NUM + (size_t)src_dt) * MAX_DT_NUM
                       + (size_t)wei_dt)
                * MAX_DT_NUM
                + (size_t)dst_dt;
    }
};

// clang-format off
const std::map<ip_impl_key_t, std::vector<impl_list_item_t>> impl_list_map {
    {{forward, f32, f32, f32}, {
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core))
#ifdef ENABLE_UNUSED_PRIM
        CPU_INSTANCE_AARCH64_ACL(acl_inner_product_fwd_t)
#endif
        REG_IP_P_FWD(CPU_INSTANCE(gemm_inner_product_fwd_t, f32))
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_fwd_t))
        nullptr,
    }},
    {{forward, bf16, bf16, f32}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_bf16))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16))
        REG_IP_P_FWD(CPU_INSTANCE_X64(gemm_bf16_inner_product_fwd_t, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_fwd_t))
#endif
        nullptr,
    }},
    {{forward, bf16, bf16, bf16}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_bf16))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16))
        REG_IP_P_FWD(CPU_INSTANCE_X64(gemm_bf16_inner_product_fwd_t, bf16))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_fwd_t))
#endif
        nullptr,
    }},
#ifdef ENABLE_UNUSED_PRIM
    {{backward_data, f32, f32, f32}, {
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_data_t, avx512_core))
        REG_IP_P_BWD(CPU_INSTANCE(gemm_inner_product_bwd_data_t, f32))
        REG_IP_P_BWD(CPU_INSTANCE(ref_inner_product_bwd_data_t))
        nullptr,
    }},
    {{backward_data, f32, bf16, bf16}, {
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_data_t, avx512_core_bf16_amx_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_data_t, avx512_core_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_data_t, f32))
        REG_IP_P_BWD(CPU_INSTANCE(ref_inner_product_bwd_data_t))
        nullptr,
    }},
    {{backward_data, bf16, bf16, bf16}, {
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_data_t, avx512_core_bf16_amx_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_data_t, avx512_core_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_data_t, bf16))
        REG_IP_P_BWD(CPU_INSTANCE(ref_inner_product_bwd_data_t))
        nullptr,
    }},
    {{backward_weights, f32, f32, f32}, {
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_weights_t, avx512_core))
        REG_IP_P_BWD(CPU_INSTANCE(gemm_inner_product_bwd_weights_t, f32))
        REG_IP_P_BWD(CPU_INSTANCE(ref_inner_product_bwd_weights_t))
        nullptr,
    }},
    {{backward_weights, bf16, f32, bf16}, {
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_weights_t, avx512_core_bf16_amx_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_weights_t, avx512_core_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_weights_t, f32))
        REG_IP_P_BWD(CPU_INSTANCE(ref_inner_product_bwd_weights_t))
        nullptr,
    }},
    {{backward_weights, bf16, bf16, bf16}, {
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_weights_t, avx512_core_bf16_amx_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(brgemm_inner_product_bwd_weights_t, avx512_core_bf16))
        REG_IP_P_BWD(CPU_INSTANCE_X64(gemm_bf16_inner_product_bwd_weights_t, bf16))
        REG_IP_P_BWD(CPU_INSTANCE(ref_inner_product_bwd_weights_t))
        nullptr,
    }},
#endif
    {{forward, s8, s8, f32}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, s8, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
#endif
        nullptr,
    }},
#ifdef ENABLE_UNUSED_PRIM
    {{forward, s8, s8, s32}, {
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, s8, s32))
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
        nullptr,
    }},
#endif
    {{forward, s8, s8, s8}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, s8, s8))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
#endif
        nullptr,
    }},
    {{forward, s8, s8, u8}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, s8, u8))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
#endif
        nullptr,
    }},
    {{forward, u8, s8, f32}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, u8, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
#endif
        nullptr,
    }},
#ifdef ENABLE_UNUSED_PRIM
    {{forward, u8, s8, s32}, {
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, u8, s32))
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
        nullptr,
    }},
#endif
    {{forward, u8, s8, s8}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, u8, s8))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
#endif
        nullptr,
    }},
    {{forward, u8, s8, u8}, {
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
#endif
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_vnni))
        REG_IP_P_FWD(CPU_INSTANCE(gemm_x8s8s32x_inner_product_fwd_t, u8, u8))
#ifdef ENABLE_UNUSED_PRIM
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
#endif
        nullptr,
    }},
#ifdef ENABLE_UNUSED_PRIM
    {{forward, s8, s8, bf16}, {
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16))
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
        nullptr,
    }},
    {{forward, u8, s8, bf16}, {
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16_amx_int8))
        REG_IP_P_FWD(CPU_INSTANCE_X64(brgemm_inner_product_fwd_t, avx512_core_bf16))
        REG_IP_P_FWD(CPU_INSTANCE(ref_inner_product_int8_fwd_t))
        nullptr,
    }},
#endif
};
// clang-format on
} // namespace

const impl_list_item_t *get_inner_product_impl_list(
        const inner_product_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : desc->prop_kind;

    const memory_desc_t *src_md = desc->prop_kind == backward_data
            ? &desc->diff_src_desc
            : &desc->src_desc;
    const memory_desc_t *wei_md = desc->prop_kind == backward_weights
            ? &desc->diff_weights_desc
            : &desc->weights_desc;
    const memory_desc_t *dst_md
            = is_fwd ? &desc->dst_desc : &desc->diff_dst_desc;
    ip_impl_key_t key {
            prop_kind,
            src_md->data_type,
            wei_md->data_type,
            dst_md->data_type,
    };

    const auto impl_list_it = impl_list_map.find(key);
    return (impl_list_it != impl_list_map.cend()) ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
