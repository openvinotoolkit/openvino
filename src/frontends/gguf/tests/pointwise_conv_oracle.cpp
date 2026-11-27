// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Independent ggml CPU reference for the opt-in F16 im2col convolution tests.

#include <cstdio>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

int main() {
    const float xv[] = {1.0003f, -2.0006f, 0.3334f, 2.0006f, 0.7002f, -1.0003f};
    const float wv[] = {1.0003f, 0.3334f, -0.7002f, 1.0003f};
    for (const auto weight_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        ggml_init_params params{1024 * 1024, nullptr, true};
        auto* ctx = ggml_init(params);
        auto* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, 2, 1);
        auto* w = ggml_new_tensor_4d(ctx, weight_type, 1, 1, 2, 2);
        auto* y = ggml_conv_2d(ctx, w, x, 1, 1, 0, 0, 1, 1);
        auto* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        auto backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 1);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(x, xv, 0, sizeof(xv));
        if (weight_type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(w, wv, 0, sizeof(wv));
        } else {
            ggml_fp16_t half[4];
            ggml_fp32_to_fp16_row(wv, half, 4);
            ggml_backend_tensor_set(w, half, 0, sizeof(half));
        }
        const auto status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS)
            return 1;
        std::vector<float> actual(ggml_nelements(y));
        ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
        std::printf("%s", ggml_type_name(weight_type));
        for (float value : actual)
            std::printf(" %.9g", value);
        std::printf("\n");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    return 0;
}
