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

#ifndef CPU_X64_BRGEMM_BRGEMM_TYPES_HPP
#define CPU_X64_BRGEMM_BRGEMM_TYPES_HPP

#include "common/primitive_attr.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// The type defines organization of batch of matrices
typedef enum {
    // A and B arrays of pointers
    brgemm_addr = 1,
    // Base address and array of offsets from base address.
    brgemm_offs = 2,
    // Base address and fixed stride between matrices.
    brgemm_strd = 3,
} brgemm_batch_kind_t;

// The type defines the storage format of matrix
typedef enum {
    brgemm_col_major = 1,
    brgemm_row_major = 2,
} brgemm_layout_t;

typedef enum {
    none = 0,
    per_tensor = 1,
    per_m = 2,
    per_n = 3,
    per_k = 4,
} brgemm_broadcast_t;

struct brgemm_strides_t {
    // Stride between A matrices
    dim_t stride_a;
    // Stride between B matrices
    dim_t stride_b;
};

typedef enum {
    brgemm_lo_default = 0,
    brgemm_lo_bl_1load,
    brgemm_lo_bl_1bcst,
} brgemm_kernel_loop_order_t;

typedef enum {
    brgemm_prf_default = 1,
} brgemm_kernel_prefetching_t;

typedef enum {
    brgemm_bd_loop_innermost = 0,
    brgemm_ld_loop_innermost,
} brgemm_kernel_innermost_loop_t;

struct DNNL_API brgemm_attr_t {
    brgemm_attr_t();
    // if unrollaed kernel is used (use_uker == true)
    // then "max_bs" is the the only batch size that can be used on kernel call
    // else "max_bs" is the maximum batch size that can be used
    int max_bs;
    int max_top_vpad, max_bottom_vpad;
    dim_t hint_expected_A_size, hint_expected_B_size, hint_expected_C_size;
    brgemm_kernel_innermost_loop_t hint_innermost_loop;
    brgemm_kernel_loop_order_t hint_loop_order;
    brgemm_kernel_prefetching_t hint_prefetching;
    bool wary_tail_read;
    bool generate_skip_accumulation;
    // bd_mask is char array in which each element is a boolean value that
    // determines whether to write this row to the result mastrix or skip
    char *bd_mask;
    // Value of bd_mask_level specifies how bd_mask is used in brgemm kernel
    // 0 – bd_mask is not used
    // 1 – bd_mask is used on storing stage only
    // 2 – bd_mask used both on reading and storing stages
    int bd_mask_level;
    // use_uker is a boolean value that determines whether to use the unrolled
    // kernel or not
    bool use_uker;
    // use_interleave_stores is a value that determines whether to use the
    // interleave stores or not
    bool use_interleave_stores;
};

struct brgemm_batch_element_t {
    brgemm_batch_element_t() {
        ptr.A = ptr.B = nullptr;
        vvpad.top = vvpad.bottom = 0;
    }
    union {
        struct {
            const void *A;
            const void *B;
        } ptr;
        struct {
            dim_t A;
            dim_t B;
        } offset;
    };
    union {
        struct {
            dim_t top;
            dim_t bottom;
        } vvpad;
        struct {
            dim_t left;
            dim_t right;
        } hvpad;
    };
};

struct brgemm_t {
    int bcast_dim = 0; // M;
    int load_dim = 0; // N;
    int reduce_dim = 0; // K;
    int LDA = 0;
    int LDB = 0;
    int LDC = 0;
    int LDD = 0;

    float alpha = 0.0f;
    float beta = 0.0f;

    int bdb = 0, bd_block = 0, bdb_tail = 0;
    int bdb2 = 0, bd_block2 = 0, bdb2_tail = 0;
    int ldb = 0, ld_block = 0, ldb_tail = 0;
    int ldb2 = 0, ld_block2 = 0, ldb2_tail = 0;
    int rdb = 0, rd_block = 0, rdb_tail = 0;
    int rd_step = 0, ld_step = 0;

    impl::data_type_t dt_a = data_type::undef;
    impl::data_type_t dt_c = data_type::undef;
    impl::data_type_t dt_b = data_type::undef;
    impl::data_type_t dt_d = data_type::undef;
    impl::data_type_t dt_bias = data_type::undef;

    int typesize_A = 0;
    int typesize_B = 0;
    int typesize_C = 0;
    int typesize_D = 0;
    int typesize_bias = 0;

    bool is_int8 = false, is_int8_amx = false;
    bool is_bf16 = false, is_bf16_amx = false;
    bool is_f32 = false;
    bool is_amx = false;

    dim_t stride_a = 0; // Offset in bytes
    dim_t stride_b = 0;

    brgemm_layout_t layout;
    brgemm_batch_kind_t type;

    bool embd_bcst = false;
    bool is_dgmm = false; // set to true in brdgmm_desc_init
    bool with_bias = false;
    bool with_sum = false;
    float sum_scale = 0.0f;
    int32_t sum_zp = 0;
    impl::data_type_t sum_dt;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_scales = false;
    bool req_s8s8_compensation = false;
    brgemm_broadcast_t zp_type_a = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_b = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_c = brgemm_broadcast_t::none;

    int is_oc_scale = 0;

    const primitive_attr_t *attr = nullptr;
    const memory_desc_t *dst_md = nullptr;

    brgemm_attr_t brgattr;
    static constexpr int MAX_VPAD = 100;

    int is_M_tail;
    // Tile register decomposition
    int get_ld_block2() const noexcept {
        return (ldb_tail) ? ld_block2 + 1 : ld_block2;
    }
    int get_num_C_tiles() const noexcept { return bd_block2 * get_ld_block2(); }
    int get_num_A_tiles() const noexcept { return bd_block2; }
    int get_num_B_tiles() const noexcept { return get_ld_block2(); }

    int get_C_tensor(int m, int n) const noexcept {
        return (m * get_ld_block2() + n);
    }
    int get_A_tensor(int m) const noexcept { return (get_num_C_tiles() + m); }
    int get_B_tensor(int n) const noexcept {
        return (get_num_C_tiles() + get_num_A_tiles() + n);
    }
};

struct brgemm_kernel_params_t {
    const void *ptr_A;
    const void *ptr_B;
    const brgemm_batch_element_t *batch;
    void *ptr_C;

    const void *ptr_bias;
    void *ptr_D;

    const void *ptr_scales;
    void *ptr_buf;

    size_t do_post_ops;
    size_t BS;

    /*
     * ptr to table of void * elements that are pointers to post_op binary
     * src1 tensors
     */
    const void *post_ops_binary_rhs_arg_vec;
    size_t oc_logical_off;
    size_t first_mb_matrix_addr_off;
    size_t dst_row_logical_off;

    char *data_C_ptr_;

    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *c_zp_values = nullptr;
    size_t skip_accm = 0;
};

struct jit_brgemm_kernel_t;
struct jit_brgemm_amx_uker_base_t;
struct jit_brdgmm_kernel_base_t;

struct brgemm_kernel_t {
    brgemm_kernel_t(const brgemm_t abrd) {};
    virtual ~brgemm_kernel_t() {};
    virtual status_t create_kernel() = 0;
    virtual void operator()(brgemm_kernel_params_t *) const = 0;
};

struct brgemm_kernel_common_t : public brgemm_kernel_t {
    brgemm_kernel_common_t(const brgemm_t abrd);
    ~brgemm_kernel_common_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;

private:
    jit_brgemm_kernel_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_kernel_common_t);
};

struct brgemm_amx_uker_t : public brgemm_kernel_t {
    brgemm_amx_uker_t(const brgemm_t abrd);
    ~brgemm_amx_uker_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;

private:
    jit_brgemm_amx_uker_base_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_amx_uker_t);
};

struct brdgmm_kernel_t : public brgemm_kernel_t {
    brdgmm_kernel_t(const brgemm_t abrd);
    ~brdgmm_kernel_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;

private:
    jit_brdgmm_kernel_base_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brdgmm_kernel_t);
};

/// @param bias Vector of bias (vector length is N)
/// @param scales Vector of scales (vector length is N)
/// @param binary_post_ops_rhs - Ptr to table of pointers to tensors used as rhs
///     in binary post-operation { void* binary_op_tensor1, ...,
///      void* binary_op_tensor_n}
/// @param oc_logical_off - Used in binary postops in per_oc bcast strategy.
///     Offset to start oc processed by given thread in elements.
/// @param dst_row_logical_off - Used in binary postops in per_oc bcast
///     strategy. Offset to start oc processed by given thread in elements.
/// @param a_zp_compensations - Pre-computed compensations for A matrix zero
///     point values.
/// @param b_zp_compensations - Pre-computed compensations for B matrix zero
///     point values.
/// @param c_zp_values - C matrix zero point values.
/// @param skip_accumulation - specifies whether to skip accumulation when
///    computing post-ops.
///
struct brgemm_post_ops_data_t {
    brgemm_post_ops_data_t() = default;
    brgemm_post_ops_data_t(const void *bias, const float *scales,
            const void *binary_post_ops_rhs, size_t oc_logical_off,
            const size_t dst_row_logical_off = 0, char *data_C_ptr_ = nullptr,
            const size_t first_mb_matrix_addr_off = 0,
            const void *a_zp_compensations = nullptr,
            const void *b_zp_compensations = nullptr,
            const void *c_zp_values = nullptr, bool skip_accumulation = false)
        : bias(bias)
        , scales(scales)
        , binary_post_ops_rhs(binary_post_ops_rhs)
        , oc_logical_off(oc_logical_off)
        , dst_row_logical_off(dst_row_logical_off)
        , data_C_ptr_(data_C_ptr_)
        , first_mb_matrix_addr_off(first_mb_matrix_addr_off)
        , a_zp_compensations(a_zp_compensations)
        , b_zp_compensations(b_zp_compensations)
        , c_zp_values(c_zp_values)
        , skip_accumulation(skip_accumulation) {}

    const void *bias = nullptr;
    const float *scales = nullptr;
    const void *binary_post_ops_rhs = nullptr;
    size_t oc_logical_off = 0;
    size_t dst_row_logical_off = 0;
    char *data_C_ptr_ = nullptr;
    size_t first_mb_matrix_addr_off = 0;
    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *c_zp_values = nullptr;
    const bool skip_accumulation = false;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
