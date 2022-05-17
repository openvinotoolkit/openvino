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

#ifndef COMMON_GEMM_TYPES_HPP
#define COMMON_GEMM_TYPES_HPP

#include <assert.h>

#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {
namespace impl {

enum transpose_t { dnnl_notrans, dnnl_trans };

namespace transpose {
const transpose_t notrans = dnnl_notrans;
const transpose_t trans = dnnl_trans;
} // namespace transpose

enum offsetc_t { dnnl_fixed, dnnl_column, dnnl_row };

namespace offsetc {
const offsetc_t fixed = dnnl_fixed;
const offsetc_t column = dnnl_column;
const offsetc_t row = dnnl_row;
} // namespace offsetc

/** A descriptor for a matrix multiplication (gemm) operation */
struct dnnl_gemm_desc_t {
    /* To make the interface consistent, the descriptor represent the
     * GEMM operation in row major */

    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #dnnl_gemm. */
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t a_desc;
    dnnl_memory_desc_t b_desc;
    dnnl_memory_desc_t c_desc;
    dnnl_memory_desc_t bias_desc;
    /** Type for accumulating A*B. */
    dnnl_data_type_t acc_type;

    // These accessors are to be used by the GEMM implementation
    // Because the GEMM implementation currently assumes column major
    // These accessors return data in column major fashion

    inline bool is_batched() const { return c_desc.ndims >= 3; }

    // Simplified accessors that comply to GEMM API
    transpose_t get_trans(dnnl_memory_desc_t md) const {
        return md.format_desc.blocking.strides[md.ndims - 1] != 1
                ? transpose::trans
                : transpose::notrans;
    }
    transpose_t transa() const { return get_trans(b_desc); };
    transpose_t transb() const { return get_trans(a_desc); };
    dnnl_dim_t batch() const {
        // if ndims < 3, it should return 1
        int64_t batch = 1;
        for (int i = 0; i < c_desc.ndims - 2; ++i) {
            if (c_desc.dims[i] == DNNL_RUNTIME_DIM_VAL)
                return DNNL_RUNTIME_DIM_VAL;
            batch *= c_desc.dims[i];
        }
        return batch;
    }

    /** Number of rows of C. */
    dnnl_dim_t m() const { return c_desc.dims[c_desc.ndims - 1]; }
    /** Number of columns of C. */
    dnnl_dim_t n() const { return c_desc.dims[c_desc.ndims - 2]; }
    /** Size of inner dimension shared between A and B. */
    dnnl_dim_t k() const { return a_desc.dims[a_desc.ndims - 1]; }

    /** Stride between 2 matrices A in a batch. */
    dnnl_dim_t stride_a(int dim = 0) const {
        return (dim >= b_desc.ndims - 2 || b_desc.dims[dim] == 1)
                ? 0
                : b_desc.format_desc.blocking.strides[dim];
    };
    /** Stride between 2 matrices B in a batch. */
    dnnl_dim_t stride_b(int dim = 0) const {
        return (dim >= a_desc.ndims - 2 || a_desc.dims[dim] == 1)
                ? 0
                : a_desc.format_desc.blocking.strides[dim];
    };
    /** Stride between 2 matrices C in a batch. */
    dnnl_dim_t stride_c(int dim = 0) const {
        return (dim >= c_desc.ndims - 2)
                ? 0
                : c_desc.format_desc.blocking.strides[dim];
    };

    // This assumes that one of the dimensions has strides 1
    dnnl_dim_t get_ld(dnnl_memory_desc_t md) const {
        auto strides = md.format_desc.blocking.strides;
        assert(strides[md.ndims - 1] == 1 || strides[md.ndims - 2] == 1);
        return strides[md.ndims - 1] != 1 ? strides[md.ndims - 1]
                                          : strides[md.ndims - 2];
    }
    /** Leading dimension of A. */
    dnnl_dim_t lda() const { return get_ld(b_desc); }
    /** Leading dimension of B. */
    dnnl_dim_t ldb() const { return get_ld(a_desc); }
    /** Leading dimension of C. */
    dnnl_dim_t ldc() const { return get_ld(c_desc); }

    /** Type of matrix A. */
    dnnl_data_type_t a_type() const { return b_desc.data_type; }
    /** Type of matrix B. */
    dnnl_data_type_t b_type() const { return a_desc.data_type; }
    /** Type of matrix C. */
    dnnl_data_type_t c_type() const { return c_desc.data_type; }
    /** Type of bias. */
    dnnl_data_type_t bias_type() const { return bias_desc.data_type; }
    /** Type of bias. */
    int bias_mask() const {
        assert(bias_desc.ndims <= 3);
        int mask = 0;
        // TODO: update the mask for batched dimension if we start
        // supporting more batch dimensions
        if (is_batched()) mask |= (bias_desc.dims[0] > 1) ? 1 << 0 : 0;

        // because the bias mask is in row major, we have to convert
        // to col major here by swapping two last dimensions
        int m_idx = is_batched();
        mask |= (bias_desc.dims[m_idx] > 1) ? 1 << (bias_desc.ndims - m_idx)
                                            : 0;
        mask |= (bias_desc.dims[m_idx + 1] > 1)
                ? 1 << (bias_desc.ndims - (m_idx + 1))
                : 0;
        return mask;
    }
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_GEMM_TYPES_HPP
