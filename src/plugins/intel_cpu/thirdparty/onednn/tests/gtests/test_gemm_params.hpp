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

#ifndef TEST_GEMM_DATA_H
#define TEST_GEMM_DATA_H

#include <cstdint>
#include <utility>
#include <type_traits>

#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {

enum class layout_t { ROW_MAJOR, COL_MAJOR };

struct test_igemm_params {
    char offsetc;
    bool nonzero_oa;
    bool nonzero_ob;
    bool nonzero_oc;

    int8_t oa() const { return (int8_t)(nonzero_oa ? 4 : 0); }
    int8_t ob() const { return (int8_t)(nonzero_ob ? 3 : 0); }
};

struct test_pack_params {
    bool pack_a;
    bool pack_b;
};

struct gemm_offset {
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t co;
};

struct test_params {
    char transA;
    char transB;
    int64_t M;
    int64_t N;
    int64_t K;
    float alpha;
    float beta;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;

    test_igemm_params igemm_params;
    test_pack_params pack_params;
    bool expect_to_fail;
    dnnl_status_t expected_status;

    gemm_offset off;

    bool tr_a() const { return transA == 'T' || transA == 't'; }
    bool tr_b() const { return transB == 'T' || transB == 't'; }
    int64_t sizeC() const { return M * ldc; }

    bool oc_is_R() const {
        auto c = igemm_params.offsetc;
        return c == 'R' || c == 'r';
    }
    bool oc_is_C() const {
        auto c = igemm_params.offsetc;
        return c == 'C' || c == 'c';
    }
    int64_t size_oc() const { return oc_is_R() ? N : oc_is_C() ? M : 1; }
};

template <typename... TArgs>
inline test_params make_test_params_with_offset(
        const gemm_offset &off, TArgs &&... args) {
    test_params params {std::forward<TArgs>(args)...};
    params.off = off;
    return params;
}

template <typename... TArgs>
inline test_params make_test_params_pack(
        const test_pack_params &pack_params, TArgs &&... args) {
    test_params params {std::forward<TArgs>(args)...};
    params.pack_params = pack_params;
    return params;
}

} // namespace dnnl

#endif
