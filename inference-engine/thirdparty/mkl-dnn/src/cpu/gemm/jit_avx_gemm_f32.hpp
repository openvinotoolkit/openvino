/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef JIT_AVX_GEMM_F32_HPP
#define JIT_AVX_GEMM_F32_HPP

#include "c_types_map.hpp"
#include "../jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

class jit_avx_gemm_f32 {
public:
    void sgemm(const char *transa, const char *transb, const int *M,
            const int *N, const int *K, const float *alpha, const float *A,
            const int *lda, const float *B, const int *ldb, const float *beta,
            float *C, const int *ldc, const float *bias = NULL);

    jit_avx_gemm_f32(
            char transa, char transb, float beta, bool hasBias = false);
    ~jit_avx_gemm_f32();

private:
    typedef void (*ker)(long long int, long long int, long long int, float *,
            float *, long long int, float *, long long int, float *, float *,
            long long int, float *);
    void sgemm_nocopy_driver(const char *transa, const char *transb, int m,
            int n, int k, const float *alpha, const float *a, int lda,
            const float *b, int ldb, const float *beta, float *c, int ldc,
            const float *bias, float *ws);

    char transa_, transb_;
    float beta_;
    bool hasBias_;
    struct xbyak_gemm;
    xbyak_gemm *ker_bn_, *ker_b1_, *ker_b0_;
    int nthrs_;
};
}
}
}

#endif
