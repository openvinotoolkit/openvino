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

#ifndef CPU_X64_MATMUL_BRGEMM_MATMUL_UTILS_HPP
#define CPU_X64_MATMUL_BRGEMM_MATMUL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

constexpr int max_batch_ndims = DNNL_MAX_NDIMS - 2;

struct brgemm_matmul_bcast_desc_t {

    brgemm_matmul_bcast_desc_t()
        : bcast_mask(0)
        , first_bcast_dim(-1)
        , last_bcast_dim(-1)
        , first_bcast_dim_to_last_batch_dim_prod(1)
        , bcast_dims_prod(1)
        , batch_dims {0}
        , gb_off {0} {}

    void set_params(const dims_t &inp_dims, const dims_t &dst_d_dims,
            int batch_ndims, dim_t batch) {
        const int ndims = batch_ndims;
        const int mask = 1 << (ndims - 1);
        first_bcast_dim_to_last_batch_dim_prod = batch;
        for (int d = 0; d < ndims; ++d) {
            batch_dims[d] = dst_d_dims[d];
            gb_off[d] = (d == 0 ? batch : gb_off[d - 1]) / dst_d_dims[d];
            if (dst_d_dims[d] != 1 && inp_dims[d] == 1) { // broadcast
                bcast_mask |= (mask >> d);
                if (first_bcast_dim == -1) {
                    first_bcast_dim = d;
                    if (d == 0) // broadcast_dim == B0
                        first_bcast_dim_to_last_batch_dim_prod = batch;
                }
                last_bcast_dim = d;
                bcast_dims_prod *= dst_d_dims[d];
            }
            if (first_bcast_dim == -1) // broadcast_dim > B0
                first_bcast_dim_to_last_batch_dim_prod /= dst_d_dims[d];
        }
    }

    int bcast_mask; // sets bcast_dim = 1, non_bcast_dim = 0

    int first_bcast_dim;
    int last_bcast_dim;

    dim_t first_bcast_dim_to_last_batch_dim_prod;
    dim_t bcast_dims_prod;

    dim_t batch_dims[max_batch_ndims];
    dim_t gb_off[max_batch_ndims]; // generalized batch offset
};

struct brgemm_matmul_conf_t {
    int ndims, batch_ndims;
    dim_t M, N, K, batch, batch_without_first_dim;
    dim_t M_blk, N_blk, K_blk, M_tail, N_tail, K_tail;
    int M_chunk_size, N_chunk_size;
    dim_t LDA, LDB, LDC, LDD;
    int brgemm_batch_size;
    int wei_n_blk, wei_k_blk;
    brgemm_batch_kind_t brg_type;

    cpu_isa_t isa;

    format_tag_t src_tag, wei_tag, dst_tag, bia_tag;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;
    bool with_scales;
    bool s8s8_compensation_required;
    bool is_oscale_per_n;
    brgemm_broadcast_t src_zp_type;
    brgemm_broadcast_t wei_zp_type;
    brgemm_broadcast_t dst_zp_type;

    bool use_buffer_a;
    bool use_buffer_a_tail_only;
    bool use_buffer_b;
    bool use_buffer_c;

    brgemm_matmul_bcast_desc_t bcast_A_desc;
    brgemm_matmul_bcast_desc_t bcast_B_desc;

    data_type_t src_dt;
    data_type_t dst_dt;
    data_type_t wei_dt;
    data_type_t acc_dt;
    data_type_t bia_dt;
    int nthr;
    int nthr_k;

    // Auxiliary values for init_config() and execute()
    dim_t a_dt_sz, b_dt_sz, c_dt_sz, acc_dt_sz, bias_dt_sz;

    int M_chunks;
    int N_chunks;
    int K_chunks;
    int num_M_blocks;
    int num_N_blocks;
    dim_t M_chunk_elems;
    dim_t N_chunk_elems;
    dim_t K_chunk_elems;

    // Pre-calculated memory strides for each tensor
    dim_t A_strides[3];
    dim_t B_strides[3];
    dim_t C_strides[3];
    dim_t buffer_c_chunk_sz;
    dim_t buffer_c_per_thread_sz;

    dim_t buffer_a_chunk_sz;
    dim_t buffer_a_chunk_shift_along_m;
    dim_t buffer_a_per_thread_sz;

    dim_t buffer_b_chunk_sz;
    dim_t buffer_b_per_thread_sz;
    dim_t s8s8_comp_ithr_str;
    dim_t s8s8_comp_b_str;
    dim_t s8s8_comp_n_str;
    bool has_zero_point_a, has_zero_point_b, has_zero_point_c;
    bool post_ops_applicable;
    bool transposed_A;
    bool blocked_B;

    dim_t zp_a_comp_shift_n;
    dim_t zp_a_comp_elems_per_thr;

    dim_t zp_b_comp_result_shift_m;
    dim_t zp_b_comp_buffer_start;
    dim_t zp_b_comp_buffer_shift_m;
    dim_t zp_b_comp_elems_per_thr;

    int wsp_tile_per_thr_bytes;
    int brgemm_batch_element_per_thr_sz;
    bool is_amx;

    int required_k_granularity;
};

struct brgemm_matmul_conf_utils_t {

    brgemm_matmul_conf_utils_t(brgemm_matmul_conf_t &bgmmc, bool A_any_layout,
            bool B_any_layout, bool C_any_layout, bool bias_any_layout);

    void init_layout_params() {
        using namespace format_tag;

        // uses 'batch_ndims'
        plain_tensor_layout_tag = utils::pick(bgmmc.ndims - 2, ab, abc, abcd,
                abcde, abcdef, abcdefg, abcdefgh, abcdefghi, abcdefghij,
                abcdefghijk, abcdefghijkl);
        transposed_tensor_layout_tag = utils::pick(bgmmc.ndims - 2, ba, acb,
                abdc, abced, abcdfe, abcdegf, abcdefhg, abcdefgih, abcdefghji,
                abcdefghikj, abcdefghijlk);

        blocked_64n_B_layout_tag = pick_blocked_B_layout(64);
        blocked_48n_B_layout_tag = pick_blocked_B_layout(48);
        blocked_32n_B_layout_tag = pick_blocked_B_layout(32);
        blocked_16n_B_layout_tag = pick_blocked_B_layout(16);

        blocked_B_layouts_allowed = !utils::one_of(format_tag::undef,
                blocked_64n_B_layout_tag, blocked_48n_B_layout_tag,
                blocked_32n_B_layout_tag, blocked_16n_B_layout_tag);

        n_blk_fixed = (!B_any_layout) && blocked_B_layouts_allowed
                && check_b_layout_blocked_by_n(bgmmc.wei_tag);
    }

    inline bool check_b_layout_blocked_by_n(format_tag_t matrix_b_tag) const {
        return blocked_B_layouts_allowed
                && utils::one_of(matrix_b_tag, blocked_64n_B_layout_tag,
                        blocked_48n_B_layout_tag, blocked_32n_B_layout_tag,
                        blocked_16n_B_layout_tag);
    }

    inline bool get_blocked_B() const {
        return blocked_B_layouts_allowed
                && check_b_layout_blocked_by_n(bgmmc.wei_tag);
    }

    inline bool use_buffer_b(bool use_heuristic = true) const {
        if (bgmmc.is_amx) return !bgmmc.blocked_B;

        // Values based on measured performance difference
        // between plain and copy-to-blocked routine.
        size_t big_LDB = bgmmc.N > 256;
        bool is_pow2 = math::is_pow2(bgmmc.N);
        bool use_copy_buffer = IMPLICATION(
                this->is_f32(), use_heuristic && (big_LDB && is_pow2));
        return (use_copy_buffer && this->check_is_plain(bgmmc.wei_tag))
                || this->check_is_transposed(bgmmc.wei_tag);
    }

    inline dim_t get_actual_LDB() const {
        bool use_blocked_LDB = bgmmc.is_amx || bgmmc.use_buffer_b
                || bgmmc.wei_tag != plain_tensor_layout_tag;
        return use_blocked_LDB ? bgmmc.wei_n_blk : bgmmc.N;
    }

    inline bool check_n_blk_fixed() const { return n_blk_fixed; }

    inline bool check_is_transposed(format_tag_t tag) const {
        return tag == transposed_tensor_layout_tag;
    }

    inline bool check_is_plain(format_tag_t tag) const {
        return tag == plain_tensor_layout_tag;
    }

    inline bool is_f32() const { return f32_dt; }

    inline bool is_bf16() const { return bf16_dt; }

    inline bool is_int8() const { return int8_dt; }

    inline bool is_int8_with_bf16_dst() const {
        return this->is_int8() && bgmmc.dst_dt == data_type::bf16;
    }

    status_t set_or_check_B_tag(memory_desc_t &B_md) const;
    status_t update_and_check_B_tag(memory_desc_t &B_md, int n_blk_size) const;
    status_t set_or_check_tags(memory_desc_t &A_md, memory_desc_t &C_md,
            memory_desc_t &bias_md) const;
    status_t set_B_flags(memory_desc_t &B_md) const;
    format_tag_t pick_blocked_B_layout(int n_blk) const;

private:
    brgemm_matmul_conf_t &bgmmc;

    const bool f32_dt, bf16_dt, int8_dt;
    const bool A_any_layout;
    const bool B_any_layout;
    const bool C_any_layout;
    const bool bias_any_layout;
    bool n_blk_fixed;

    format_tag_t plain_tensor_layout_tag;
    format_tag_t transposed_tensor_layout_tag;
    format_tag_t blocked_64n_B_layout_tag, blocked_48n_B_layout_tag,
            blocked_32n_B_layout_tag, blocked_16n_B_layout_tag;
    bool blocked_B_layouts_allowed;
};

void init_aux_values(brgemm_matmul_conf_t &bgmmc,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
        const memory_desc_wrapper &dst_d);

status_t init_brgemm_matmul_conf(cpu_isa_t isa, brgemm_matmul_conf_t &bgmmc,
        const matmul_desc_t &mmd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc);

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
