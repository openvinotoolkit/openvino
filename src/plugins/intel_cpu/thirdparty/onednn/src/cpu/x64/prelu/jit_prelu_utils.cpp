/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <algorithm>
#include <cassert>
#include <set>

#include "common/memory_desc_wrapper.hpp"
#include "cpu/x64/prelu/jit_prelu_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace prelu {

cpu_isa_t get_supported_isa() {
    if (mayiuse(avx512_core_bf16))
        return avx512_core_bf16;
    else if (mayiuse(avx512_core))
        return avx512_core;
    else if (mayiuse(avx512_common))
        return avx512_common;
    else if (mayiuse(avx2))
        return avx2;
    else if (mayiuse(avx))
        return avx;
    else if (mayiuse(sse41))
        return sse41;

    return isa_any;
}

static int get_vlen(const cpu_isa_t &isa) noexcept {
    if (isa == avx512_core_bf16)
        return cpu_isa_traits<avx512_core_bf16>::vlen;
    else if (isa == avx512_core)
        return cpu_isa_traits<avx512_core>::vlen;
    else if (isa == avx512_common)
        return cpu_isa_traits<avx512_common>::vlen;
    else if (isa == avx2)
        return cpu_isa_traits<avx2>::vlen;
    else if (isa == avx)
        return cpu_isa_traits<avx>::vlen;
    return cpu_isa_traits<sse41>::vlen;
}

int get_n_vregs(const cpu_isa_t &isa) noexcept {
    if (isa == avx512_core_bf16)
        return cpu_isa_traits<avx512_core_bf16>::n_vregs;
    else if (isa == avx512_core)
        return cpu_isa_traits<avx512_core>::n_vregs;
    else if (isa == avx512_common)
        return cpu_isa_traits<avx512_common>::n_vregs;
    else if (isa == avx2)
        return cpu_isa_traits<avx2>::n_vregs;
    else if (isa == avx)
        return cpu_isa_traits<avx>::n_vregs;
    return cpu_isa_traits<sse41>::n_vregs;
}

bool is_s8u8(const std::set<data_type_t> &tensor_data_types) noexcept {
    return std::any_of(tensor_data_types.cbegin(), tensor_data_types.cend(),
            [](const data_type_t &dt) {
                return utils::one_of(dt, data_type::s8, data_type::u8);
            });
}

int get_simd_w(const std::set<data_type_t> &tensor_data_types) noexcept {
    const auto &isa = prelu::get_supported_isa();

    return (isa == avx && is_s8u8(tensor_data_types))
            ? vmm_traits_t<Xbyak::Xmm>::vlen / sizeof(float)
            : prelu::get_vlen(isa) / sizeof(float);
}

static bool dims_equal(
        const dims_t &lhs_dims, const dims_t &rhs_dims, const dim_t ndims) {

    for (dim_t i = 0; i < ndims; ++i) {
        if (lhs_dims[i] != rhs_dims[i]) return false;
    }

    return true;
}

static bool is_full_bcast(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {
    const auto lhs_ndims = lhs.ndims();
    const auto rhs_ndims = rhs.ndims();
    const dims_t &lhs_dims = lhs.dims();
    const dims_t &rhs_dims = rhs.dims();

    if (lhs_ndims == rhs_ndims && dims_equal(lhs_dims, rhs_dims, lhs_ndims)
            && lhs.format_kind() == rhs.format_kind()) {

        if (lhs.is_blocking_desc()) {
            const auto &lhs_bd = lhs.blocking_desc();
            const auto &rhs_bd = rhs.blocking_desc();
            const dims_t &lhs_strides = lhs_bd.strides;
            const dims_t &rhs_strides = rhs_bd.strides;
            const dims_t &lhs_inner_blks = lhs_bd.inner_blks;
            const dims_t &rhs_inner_blks = rhs_bd.inner_blks;
            const dims_t &lhs_inner_idxs = lhs_bd.inner_idxs;
            const dims_t &rhs_inner_idxs = rhs_bd.inner_idxs;

            return lhs_bd.inner_nblks == rhs_bd.inner_nblks
                    && dims_equal(lhs_strides, rhs_strides, lhs_ndims)
                    && dims_equal(lhs_inner_blks, rhs_inner_blks, lhs_ndims)
                    && dims_equal(lhs_inner_idxs, rhs_inner_idxs, lhs_ndims);
        }

        return true;
    }

    return false;
}

static bool is_per_oc_bcast(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {

    const auto &rhs_dims = rhs.dims();
    const auto &lhs_dims = lhs.dims();
    const auto &rhs_ndims = rhs.ndims();

    bool bcast_per_oc_exists = rhs_dims[0] == 1 && rhs_dims[1] == lhs_dims[1];

    if (bcast_per_oc_exists) {
        for (int dim_id = 2; dim_id < rhs_ndims; ++dim_id) {
            bcast_per_oc_exists = bcast_per_oc_exists && rhs_dims[dim_id] == 1;
        }
    }

    return bcast_per_oc_exists;
}

bcast get_bcast_type(
        const memory_desc_wrapper &lhs, const memory_desc_wrapper &rhs) {

    if (is_full_bcast(lhs, rhs)) return bcast::full;
    const auto &lhs_ndims = lhs.ndims();
    const auto &rhs_ndims = rhs.ndims();

    if (lhs_ndims != rhs_ndims || lhs_ndims < 2) return bcast::unsupported;

    if (is_per_oc_bcast(lhs, rhs)) {
        const auto &strides = lhs.blocking_desc().strides;

        if (!lhs.is_plain())
            return bcast::per_oc_blocked;
        else if (strides[1] == 1)
            return bcast::per_oc_n_spatial_c;
        else if (strides[0] >= strides[1]
                && IMPLICATION(lhs_ndims >= 3, strides[1] >= strides[2]))
            return bcast::per_oc_n_c_spatial;
    }

    return bcast::unsupported;
}

bool dt_supported(const std::set<data_type_t> &tensor_data_types) noexcept {

    const bool tensor_dt_valid = std::all_of(tensor_data_types.cbegin(),
            tensor_data_types.cend(), [](const data_type_t &dt) {
                return utils::one_of(dt, data_type::bf16, data_type::f32,
                        data_type::s32, data_type::u8, data_type::s8);
            });

    if (tensor_dt_valid) {
        const bool any_tensor_bf16 = std::any_of(tensor_data_types.cbegin(),
                tensor_data_types.cend(),
                [](const data_type_t &dt) { return dt == data_type::bf16; });

        return IMPLICATION(any_tensor_bf16, mayiuse(avx512_core));
    }

    return false;
}

size_t c_blk_nelems(const memory_desc_t *mem, bool padding) noexcept {
    const memory_desc_wrapper mem_d {mem};
    return mem_d.nelems(padding) / mem_d.dims()[0];
}

size_t get_block_tail_size(const memory_desc_t *mem) noexcept {
    const memory_desc_wrapper mem_d {mem};
    return mem_d.padded_dims()[1] - mem_d.dims()[1];
}

void apply_zero_padding(jit_generator *host, const size_t tail_size,
        const data_type_t dt, const size_t block_tail_size,
        const Xbyak::Reg64 &reg_dst, const Xbyak::Reg64 *reg_offset) noexcept {
    using namespace Xbyak;
    using namespace Xbyak::util;

    const Reg32 &reg_zero = eax;
    const Reg64 &reg_ptr = rdi;
    const Reg64 &reg_counter = rcx;
    const auto dt_size = types::data_type_size(dt);
    const auto off_start = tail_size * dt_size;
    const auto off_end = off_start + block_tail_size * dt_size;

    host->xor_(reg_zero, reg_zero);
    if (reg_offset == nullptr)
        host->lea(reg_ptr, ptr[reg_dst + off_start]);
    else
        host->lea(reg_ptr, ptr[reg_dst + (*reg_offset * dt_size) + off_start]);
    host->mov(reg_counter, off_end - off_start);
    host->rep();
    host->stosb();
}

} // namespace prelu
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
