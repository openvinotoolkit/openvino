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

#ifndef GPU_OCL_OCL_POST_OPS_H
#define GPU_OCL_OCL_POST_OPS_H

#ifndef SUB_GROUP_SIZE
#define SUB_GROUP_SIZE get_sub_group_size()
#endif

#if WITH_POST_OP

#if !WITH_ELTWISE
#undef WITH_ELTWISE
#define WITH_ELTWISE 1
#endif

#include "gpu/ocl/ocl_eltwise.h"
#include "gpu/ocl/ocl_types.h"

float fwd_Xnary(unsigned kind, unsigned algorithm, float x, float y,
        float alpha, float beta, float scale) {
    if (kind == PO_BINARY) {
        switch (algorithm) {
            // binary
            case BINARY_ADD: return x + y; break;
            case BINARY_MUL: return x * y; break;
            case BINARY_MIN: return x < y ? x : y; break;
            case BINARY_MAX: return x > y ? x : y; break;
            case BINARY_DIV: return x / y; break;
            case BINARY_SUB: return x - y; break;
            case BINARY_GE: return x >= y; break;
            case BINARY_GT: return x > y; break;
            case BINARY_LE: return x <= y; break;
            case BINARY_LT: return x < y; break;
            case BINARY_EQ: return x == y; break;
            case BINARY_NE: return x != y; break;
            case RELU: // binary && relu = prelu
                return fwd_eltwise_common(RELU, x, y, beta, scale);
                break;
            default: return 0.f;
        }
    } else { // eltwise kind
        return fwd_eltwise_common(algorithm, x, alpha, beta, scale);
    }
}

#define CONV_BIN_ARG_TO_FLOAT(idx, bin_arg_val) \
    ({ \
        float ret_val; \
        if (CONCAT3(PO_, idx, _BIN_ARG_DT_IS_BF16)) \
            ret_val = cvt_bf16_to_f32(bin_arg_val); \
        else \
            ret_val = convert_float(bin_arg_val); \
\
        ret_val; \
    })

#define FWD_XNARY_GENERIC_DT(po_kind, algorithm, result, result_elem_dt, \
        arg0_ptr, arg0_len, arg1_ptr, arg1_len, alpha, beta, scale) \
    { \
        const unsigned out_len = max((unsigned)arg0_len, (unsigned)arg1_len); \
        result_elem_dt *res_ptr = (result_elem_dt *)(&result); \
        unroll_for(unsigned idx = 0; idx < out_len; ++idx) { \
            if (arg0_len == 1 && arg1_len == 1) { \
                *res_ptr = fwd_Xnary(po_kind, algorithm, \
                        convert_float(*arg0_ptr), convert_float(*arg1_ptr), \
                        alpha, beta, scale); \
            } else if (arg0_len == 1) { \
                res_ptr[idx] = fwd_Xnary(po_kind, algorithm, \
                        convert_float(*arg0_ptr), \
                        convert_float(arg1_ptr[idx]), alpha, beta, scale); \
            } else if (arg1_len == 1) { \
                res_ptr[idx] = fwd_Xnary(po_kind, algorithm, \
                        convert_float(arg0_ptr[idx]), \
                        convert_float(*arg1_ptr), alpha, beta, scale); \
            } else { \
                res_ptr[idx] = fwd_Xnary(po_kind, algorithm, \
                        convert_float(arg0_ptr[idx]), \
                        convert_float(arg1_ptr[idx]), alpha, beta, scale); \
            } \
        } \
    }

#define FMA_BLOCK( \
        block_size, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b) \
    unroll_for(; nof_elems >= block_size; acc_ptr += block_size, \
               a_ptr += block_size, nof_elems -= block_size) { \
        CONCAT2(acc_elem_dt, block_size) \
        a_conv = CONCAT3(convert_, acc_elem_dt, block_size)( \
                *((CONCAT2(a_elem_dt, block_size) *)a_ptr)); \
        *((CONCAT2(acc_elem_dt, block_size) *)acc_ptr) = fma( \
                a_conv, b, *((CONCAT2(acc_elem_dt, block_size) *)acc_ptr)); \
    }

#define FMA_MIXED(acc_nof_elems, a, a_elem_dt, b, acc, acc_elem_dt) \
    { \
        unsigned nof_elems = acc_nof_elems; \
        a_elem_dt *a_ptr = (a_elem_dt *)(&a); \
        acc_elem_dt *acc_ptr = (acc_elem_dt *)(&acc); \
        FMA_BLOCK(8, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b); \
        FMA_BLOCK(4, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b); \
        FMA_BLOCK(2, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b); \
        if (nof_elems == 1) { *acc_ptr += (*a_ptr) * b; } \
    }

#define FILL_BIN_ARG_SERIAL(idx, dest_ptr, x0, x0_s, x1, x1_s, x1_incr, x2, \
        x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    unroll_for(unsigned x0_idx = x0, bin_arg_offset = 0; x0_idx < x0 + x0_s; \
               ++x0_idx) { \
        unroll_for(unsigned x1_idx = x1; x1_idx < x1 + x1_s; \
                   x1_idx += x1_incr) { \
            unroll_for(unsigned x2_idx = x2; x2_idx < x2 + x2_s; ++x2_idx) { \
                unroll_for(unsigned x3_idx = x3; x3_idx < x3 + x3_s; \
                           ++x3_idx) { \
                    unroll_for(unsigned x4_idx = x4; x4_idx < x4 + x4_s; \
                               ++x4_idx) { \
                        unroll_for(unsigned x5_idx = x5; x5_idx < x5 + x5_s; \
                                   ++x5_idx, ++bin_arg_offset) { \
                            const unsigned bin_arg_glob_off = OFF_MD( \
                                    CONCAT3(PO_, idx, _BIN_ARG), \
                                    x0_idx % CONCAT3(PO_, idx, _BIN_ARG_D0), \
                                    x1_idx % CONCAT3(PO_, idx, _BIN_ARG_D1), \
                                    x2_idx % CONCAT3(PO_, idx, _BIN_ARG_D2), \
                                    x3_idx % CONCAT3(PO_, idx, _BIN_ARG_D3), \
                                    x4_idx % CONCAT3(PO_, idx, _BIN_ARG_D4), \
                                    x5_idx % CONCAT3(PO_, idx, _BIN_ARG_D5)); \
                            dest_ptr[bin_arg_offset] = CONV_BIN_ARG_TO_FLOAT( \
                                    idx, \
                                    CONCAT3(po_, idx, \
                                            _binary_arg)[bin_arg_glob_off]); \
                        } \
                    } \
                } \
            } \
        } \
    }

// explicitly define sub group reads and data type with post fix "1" which are
// required in FILL_WITH_BLOCK_READ macro. In that function number of elements
// is appended to it so it works for 2,4 but for 1 there is no such name.
#define intel_sub_group_block_read_uc1 intel_sub_group_block_read_uc
#define intel_sub_group_block_read_us1 intel_sub_group_block_read_us
#define intel_sub_group_block_read1 intel_sub_group_block_read
#define uchar1 uchar
#define ushort1 ushort
#define uint1 uint

#define FILL_WITH_BLOCK_READ(idx, src_ptr, dst_ptr, nelem, data_type) \
    { \
        data_type tmp_storage[nelem]; \
        if (sizeof(data_type) == 1) { \
            *((CONCAT2(uchar, nelem) *)(&tmp_storage)) \
                    = (CONCAT2(intel_sub_group_block_read_uc, nelem)( \
                            (__global uchar *)(src_ptr))); \
        } \
        if (sizeof(data_type) == 2) { \
            *((CONCAT2(ushort, nelem) *)(&tmp_storage)) \
                    = CONCAT2(intel_sub_group_block_read_us, nelem)( \
                            (__global ushort *)(src_ptr)); \
        } \
        if (sizeof(data_type) == 4) { \
            *((CONCAT2(uint, nelem) *)(&tmp_storage)) \
                    = CONCAT2(intel_sub_group_block_read, nelem)( \
                            (__global uint *)(src_ptr)); \
        } \
        unroll_for(unsigned s_index = 0; s_index < nelem; ++s_index) { \
            dst_ptr[s_index] \
                    = CONV_BIN_ARG_TO_FLOAT(idx, tmp_storage[s_index]); \
        } \
    }

#define X_NELEMS(x) ({ x / SUB_GROUP_SIZE; })

#define CONDITIONAL_FILL( \
        idx, blocked_coord, nelem, src_ptr, dst_ptr, data_type) \
    if (blocked_coord / SUB_GROUP_SIZE == nelem) \
        FILL_WITH_BLOCK_READ(idx, src_ptr, dst_ptr, nelem, data_type);

#define FILL_BIN_ARG_TRY_BLOCK(idx, dest_ptr, dest_size, x0, x0_s, x1, x1_s, \
        x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    { \
        unroll_for(unsigned x0_idx = x0, arg_off = 0; x0_idx < x0 + x0_s; \
                   ++x0_idx, arg_off += X_NELEMS(x1_s)) { \
            const unsigned bin_arg_glob_off \
                    = OFF_MD(CONCAT3(PO_, idx, _BIN_ARG), \
                            x0_idx % CONCAT3(PO_, idx, _BIN_ARG_D0), \
                            x1 % CONCAT3(PO_, idx, _BIN_ARG_D1), \
                            x2 % CONCAT3(PO_, idx, _BIN_ARG_D2), \
                            x3 % CONCAT3(PO_, idx, _BIN_ARG_D3), \
                            x4 % CONCAT3(PO_, idx, _BIN_ARG_D4), \
                            x5 % CONCAT3(PO_, idx, _BIN_ARG_D5)); \
\
            CONDITIONAL_FILL(idx, x1_s, 1, \
                    (CONCAT3(po_, idx, _binary_arg) + bin_arg_glob_off), \
                    (dest_ptr + arg_off), CONCAT3(PO_, idx, _BIN_ARG_DATA_T)); \
            CONDITIONAL_FILL(idx, x1_s, 2, \
                    (CONCAT3(po_, idx, _binary_arg) + bin_arg_glob_off), \
                    (dest_ptr + arg_off), CONCAT3(PO_, idx, _BIN_ARG_DATA_T)); \
            CONDITIONAL_FILL(idx, x1_s, 4, \
                    (CONCAT3(po_, idx, _binary_arg) + bin_arg_glob_off), \
                    (dest_ptr + arg_off), CONCAT3(PO_, idx, _BIN_ARG_DATA_T)); \
        } \
    }

#define REPLICATE_DATA( \
        dest_ptr, dest_size, x0_s, x1_s, x2_s, x3_s, x4_s, x5_s) \
    { \
        const unsigned copy_size \
                = x0_s * X_NELEMS(x1_s) * x2_s * x3_s * x4_s * x5_s; \
        unroll_for(unsigned fid = copy_size; fid < dest_size; ++fid) { \
            *(dest_ptr + fid) = *(dest_ptr + (fid % copy_size)); \
        } \
    }

#define IS_BURSTABLE(idx, x0, x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, \
        x5, x5_s, is_burst) \
    ({ \
        bool is_burstable = is_burst; \
        if (x0_s > CONCAT3(PO_, idx, _BIN_ARG_D0) && x0_s > 1) \
            is_burstable = false; \
        if (x1_s > CONCAT3(PO_, idx, _BIN_ARG_D1) && x1_s > 1) \
            is_burstable = false; \
        if (x2_s > CONCAT3(PO_, idx, _BIN_ARG_D2) && x2_s > 1) \
            is_burstable = false; \
        if (x3_s > CONCAT3(PO_, idx, _BIN_ARG_D3) && x3_s > 1) \
            is_burstable = false; \
        if (x4_s > CONCAT3(PO_, idx, _BIN_ARG_D4) && x4_s > 1) \
            is_burstable = false; \
        if (x5_s > CONCAT3(PO_, idx, _BIN_ARG_D5) && x5_s > 1) \
            is_burstable = false; \
        if (CONCAT3(PO_, idx, _BIN_ARG_D0) * CONCAT3(PO_, idx, _BIN_ARG_D1) \
                        * CONCAT3(PO_, idx, _BIN_ARG_D2) \
                        * CONCAT3(PO_, idx, _BIN_ARG_D2) \
                        * CONCAT3(PO_, idx, _BIN_ARG_D3) \
                        * CONCAT3(PO_, idx, _BIN_ARG_D4) \
                        * CONCAT3(PO_, idx, _BIN_ARG_D5) \
                == 1) \
            is_burstable = false; \
\
        is_burstable; \
    })

#define BINARY_ARG_IS_SCALAR(idx) ({ false; })

#define APPLY_PO_BINARY(idx, accumulator, acc_elem_dt, x0, x0_s, x1, x1_s, \
        x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, is_burst) \
    { \
        const unsigned bin_arg_size = BINARY_ARG_IS_SCALAR(idx) \
                ? 1 \
                : sizeof(accumulator) / sizeof(acc_elem_dt); \
        float bin_arg[bin_arg_size]; \
        float *bin_arg_ptr = &bin_arg[0]; \
        const bool use_burst_read = IS_BURSTABLE(idx, x0, x0_s, x1, x1_s, x2, \
                x2_s, x3, x3_s, x4, x4_s, x5, x5_s, is_burst); \
        const unsigned x1_jump = is_burst ? SUB_GROUP_SIZE : 1; \
        if (use_burst_read) { \
            FILL_BIN_ARG_TRY_BLOCK(idx, bin_arg_ptr, bin_arg_size, x0, x0_s, \
                    x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                    x5_s); \
        } else { \
            FILL_BIN_ARG_SERIAL(idx, bin_arg_ptr, x0, x0_s, (x1 + x1_incr), \
                    x1_s, x1_jump, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        } \
        REPLICATE_DATA(bin_arg_ptr, bin_arg_size, x0_s, x1_s, x2_s, x3_s, \
                x4_s, x5_s); \
        FWD_XNARY_GENERIC_DT(PO_BINARY, CONCAT3(PO_, idx, _ALG), accumulator, \
                acc_elem_dt, ((acc_elem_dt *)(&accumulator)), \
                (sizeof(accumulator) / sizeof(acc_elem_dt)), bin_arg_ptr, \
                bin_arg_size, 0.0f, 0.0f, 1.0f); \
    }

#define APPLY_PO_SUM(idx, accumulator, acc_elem_dt, sum_src, sum_elem_dt) \
    { \
        unsigned acc_size = sizeof(accumulator) / sizeof(acc_elem_dt); \
        FMA_MIXED(acc_size, sum_src, sum_elem_dt, \
                CONCAT3(PO_, idx, _SUM_SCALE), accumulator, acc_elem_dt); \
    }

#define APPLY_PO_ELTWISE(idx, accumulator, acc_elem_dt) \
    { \
        FWD_XNARY_GENERIC_DT(PO_ELTWISE, CONCAT3(PO_, idx, _ALG), accumulator, \
                acc_elem_dt, ((acc_elem_dt *)(&accumulator)), \
                (sizeof(accumulator) / sizeof(acc_elem_dt)), \
                ((acc_elem_dt *)(&accumulator)), \
                (sizeof(accumulator) / sizeof(acc_elem_dt)), \
                CONCAT3(PO_, idx, _ELTWISE_ALPHA), \
                CONCAT3(PO_, idx, _ELTWISE_BETA), \
                CONCAT3(PO_, idx, _ELTWISE_SCALE)); \
    }

#define APPLY_PO_STAGE(idx, accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
        x0, x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    switch (CONCAT3(PO_, idx, _KIND)) { \
        case PO_BINARY: \
            APPLY_PO_BINARY(idx, accumulator, acc_elem_dt, x0, x0_s, x1, x1_s, \
                    x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
                    is_burst); \
            break; \
        case PO_ELTWISE: APPLY_PO_ELTWISE(idx, accumulator, acc_elem_dt); \
                break; \
        case PO_SUM: \
            APPLY_PO_SUM(idx, accumulator, acc_elem_dt, sum_src, sum_elem_dt); \
            break; \
    }

#if POST_OP_CHAIN_LENGTH == 0
#define EMPTY_POST_OPS(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    {}
#endif

#if POST_OP_CHAIN_LENGTH > 0
#define APPLY_1_PO_STAGE(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_PO_STAGE(0, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 1
#define APPLY_2_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_1_PO_STAGE(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(1, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 2
#define APPLY_3_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_2_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(2, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 3
#define APPLY_4_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_3_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(3, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 4
#define APPLY_5_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_4_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(4, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 5
#define APPLY_6_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_5_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(5, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 6
#define APPLY_7_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_6_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(6, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 7
#define APPLY_8_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_7_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(7, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 8
#define APPLY_9_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_8_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(8, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 9
#define APPLY_10_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_9_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(9, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 10
#define APPLY_11_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_10_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(10, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 11
#define APPLY_12_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_11_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(11, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 12
#define APPLY_13_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_12_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(12, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 13
#define APPLY_14_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_13_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(13, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 14
#define APPLY_15_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_14_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(14, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 15
#define APPLY_16_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_15_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(15, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 16
#define APPLY_17_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_16_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(16, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 17
#define APPLY_18_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_17_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(17, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 18
#define APPLY_19_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_18_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(18, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 19
#define APPLY_20_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_19_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(19, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 20
#define APPLY_21_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_20_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(20, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 21
#define APPLY_22_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_21_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(21, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 22
#define APPLY_23_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_22_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(22, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 23
#define APPLY_24_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_23_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(23, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 24
#define APPLY_25_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_24_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(24, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 25
#define APPLY_26_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_25_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(25, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 26
#define APPLY_27_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_26_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(26, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 27
#define APPLY_28_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_27_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(27, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 28
#define APPLY_29_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_28_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(28, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 29
#define APPLY_30_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_29_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(29, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 30
#define APPLY_31_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_30_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(30, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH > 31
#define APPLY_32_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_31_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(31, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
    }
#endif

#if POST_OP_CHAIN_LENGTH == 0
#define APPLY_ALL_PO_STAGES EMPTY_POST_OPS
#elif POST_OP_CHAIN_LENGTH == 1
#define APPLY_ALL_PO_STAGES APPLY_1_PO_STAGE
#elif POST_OP_CHAIN_LENGTH == 2
#define APPLY_ALL_PO_STAGES APPLY_2_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 3
#define APPLY_ALL_PO_STAGES APPLY_3_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 4
#define APPLY_ALL_PO_STAGES APPLY_4_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 5
#define APPLY_ALL_PO_STAGES APPLY_5_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 6
#define APPLY_ALL_PO_STAGES APPLY_6_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 7
#define APPLY_ALL_PO_STAGES APPLY_7_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 8
#define APPLY_ALL_PO_STAGES APPLY_8_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 9
#define APPLY_ALL_PO_STAGES APPLY_9_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 10
#define APPLY_ALL_PO_STAGES APPLY_10_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 11
#define APPLY_ALL_PO_STAGES APPLY_11_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 12
#define APPLY_ALL_PO_STAGES APPLY_12_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 13
#define APPLY_ALL_PO_STAGES APPLY_13_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 14
#define APPLY_ALL_PO_STAGES APPLY_14_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 15
#define APPLY_ALL_PO_STAGES APPLY_15_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 16
#define APPLY_ALL_PO_STAGES APPLY_16_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 17
#define APPLY_ALL_PO_STAGES APPLY_17_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 18
#define APPLY_ALL_PO_STAGES APPLY_18_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 19
#define APPLY_ALL_PO_STAGES APPLY_19_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 20
#define APPLY_ALL_PO_STAGES APPLY_20_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 21
#define APPLY_ALL_PO_STAGES APPLY_21_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 22
#define APPLY_ALL_PO_STAGES APPLY_22_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 23
#define APPLY_ALL_PO_STAGES APPLY_23_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 24
#define APPLY_ALL_PO_STAGES APPLY_24_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 25
#define APPLY_ALL_PO_STAGES APPLY_25_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 26
#define APPLY_ALL_PO_STAGES APPLY_26_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 27
#define APPLY_ALL_PO_STAGES APPLY_27_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 28
#define APPLY_ALL_PO_STAGES APPLY_28_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 29
#define APPLY_ALL_PO_STAGES APPLY_29_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 30
#define APPLY_ALL_PO_STAGES APPLY_30_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 31
#define APPLY_ALL_PO_STAGES APPLY_31_PO_STAGES
#elif POST_OP_CHAIN_LENGTH == 32
#define APPLY_ALL_PO_STAGES APPLY_32_PO_STAGES
#endif

#define APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s, \
        is_burst) \
    { \
        APPLY_ALL_PO_STAGES(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
                x0, x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        /*APPLY_PO_STAGE(0, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
         APPLY_PO_STAGE(1, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(2, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(3, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(4, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(5, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(6, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(7, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(8, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(9, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(10, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(11, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(12, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(13, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(14, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(15, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(16, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(17, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(18, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(19, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(20, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(21, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(22, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(23, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(24, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(25, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(26, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(27, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(28, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(29, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(30, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); \
        APPLY_PO_STAGE(31, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x1_incr, x2, x2_s, x3, x3_s, x4, x4_s, x5, \
                x5_s, is_burst); */ \
    }

#define APPLY_POST_OPS_TRY_BURST(accumulator, acc_elem_dt, sum_src, \
        sum_elem_dt, mb_start, mb_size, oc_start, oc_size, oc_serial_incr) \
    APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
            mb_start, mb_size, oc_start, oc_size, oc_serial_incr, 0, 1, 0, 1, \
            0, 1, 0, 1, true)

#define APPLY_POST_OPS_SERIAL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
        mb_start, mb_size, oc_start, oc_size, d2_start, d2_size, d3_start, \
        d3_size, d4_start, d4_size, d5_start, d5_size) \
    APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
            mb_start, mb_size, oc_start, oc_size, 0, d2_start, d2_size, \
            d3_start, d3_size, d4_start, d4_size, d5_start, d5_size, false)

#define APPLY_POST_OPS_SERIAL_BINARY_2D(accumulator, acc_elem_dt, sum_src, \
        sum_elem_dt, mb_start, mb_size, oc_start, oc_size) \
    APPLY_POST_OPS_BL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
            mb_start, mb_size, oc_start, oc_size, 0, 0, 1, 0, 1, 0, 1, 0, 1, \
            false)
#else

#define APPLY_POST_OPS_SERIAL(accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
        mb_start, mb_size, oc_start, oc_size, d2_start, d2_size, d3_start, \
        d3_size, d4_start, d4_size, d5_start, d5_size) \
    {}

#define APPLY_POST_OPS_SERIAL_BINARY_2D(accumulator, acc_elem_dt, sum_src, \
        sum_elem_dt, mb_start, mb_size, oc_start, oc_size) \
    {}

#define APPLY_POST_OPS_TRY_BURST(accumulator, acc_elem_dt, sum_src, \
        sum_elem_dt, mb_start, mb_size, oc_start, oc_size, oc_serial_incr) \
    {}

#endif // WITH_POST_OP

#endif
