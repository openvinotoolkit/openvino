// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "mul_mat.hpp"

#include <string>
#include <vector>
#include <iostream>

#include "ggml/ggml.h"

namespace ov {
namespace intel_cpu {

static void print_elements(const char* label, const struct ggml_tensor * t) {
    if (!t) {
        printf("%s: %s = null\n", __func__, label);
        return;
    }
    const int nelements = ggml_nelements(t);
    printf("%s: %s = [", __func__, label);
    for (int k = 0; k < nelements; ++k) {
        if (k > 0) { printf(", "); }
        printf("%.5f", ggml_get_f32_1d(t, k));
    }
    printf("] shape: [");
    for (int k = 0; k < t->n_dims; ++k) {
        if (k > 0) { printf(", "); }
        printf("%d", static_cast<int>(t->ne[k]));
    }
    printf("]\n");
}

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
        printf("%s: ggml_init() failed\n", __func__);
        return;
    }

    struct ggml_tensor* ggml_A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor* ggml_B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    memcpy(ggml_A->data, A_ptr, ggml_nbytes(ggml_A));
    memcpy(ggml_B->data, B_ptr, ggml_nbytes(ggml_B));

    struct ggml_tensor* ggml_dst = ggml_mul_mat(ctx, ggml_A, ggml_B);
    struct ggml_cgraph ggml_graph = ggml_build_forward(ggml_dst);
    ggml_graph_compute_with_ctx(ctx, &ggml_graph, /*n_threads = */1);

    //print_elements("ggml_A", ggml_A);
    //print_elements("ggml_B", ggml_B);
    //print_elements("ggml_dst", ggml_dst);
    float* dst = reinterpret_cast<float*>(ggml_dst->data);
    memcpy(dst_ptr, dst, ggml_nbytes(ggml_dst));
    ggml_free(ctx);
}

}  // namespace intel_cpu
}  // namespace ov