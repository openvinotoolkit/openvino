// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mul_mat.hpp"

#include <string>
#include <vector>

#include "ggml/ggml.h"

namespace ov {
namespace intel_cpu {

void ggml_mul_mat(const int64_t M,
                  const int64_t N,
                  const int64_t K,
                  const float* A_ptr,
                  const float* B_ptr,
                  float* dst_ptr,
                  const float* bias_ptr) {
    struct ggml_init_params params = {
        /*.mem_size   =*/16 * 1024 * 1024,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };

    struct ggml_context* ctx = ggml_init(params);

    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return;
    }

    struct ggml_tensor* ggml_A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* ggml_B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    memcpy(ggml_A->data, A_ptr, ggml_nbytes(ggml_A));
    memcpy(ggml_B->data, B_ptr, ggml_nbytes(ggml_B));

    struct ggml_tensor* ggml_dst = ggml_mul_mat(ctx, ggml_A, ggml_B);
    struct ggml_cgraph ggml_graph = ggml_build_forward(ggml_dst);
    ggml_graph_compute_with_ctx(ctx, &ggml_graph, /*n_threads = */1);

    float* dst = reinterpret_cast<float*>(ggml_dst->data);
    memcpy(dst_ptr, dst, M * N);
}

}  // namespace intel_cpu
}  // namespace ov