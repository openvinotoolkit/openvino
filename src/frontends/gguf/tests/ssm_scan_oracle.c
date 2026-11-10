// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Generate SSM_SCAN output with real ggml CPU. Build with ggml/include and link
// ggml, ggml-base and ggml-cpu. Usage: ssm_scan_oracle <tokens> <output.f32>.
// The deterministic inputs are also used by GGUFOps.SSMScanMamba2.
#include <stdio.h>
#include <stdlib.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    const int nt = atoi(argv[1]);
    struct ggml_init_params ip = {16 * 1024 * 1024, NULL, true};
    struct ggml_context* ctx = ggml_init(ip);
    struct ggml_tensor* s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 3, 4, 3);
    struct ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 4, nt, 2);
    struct ggml_tensor* dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, nt, 2);
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 4);
    struct ggml_tensor* b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, nt, 2);
    struct ggml_tensor* c = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, nt, 2);
    struct ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    struct ggml_tensor* out = ggml_ssm_scan(ctx, s, x, dt, a, b, c, ids);
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    struct ggml_tensor* inputs[] = {s, x, dt, a, b, c};
    for (int j = 0; j < 6; ++j) {
        const int64_t count = ggml_nelements(inputs[j]);
        float* values = malloc(count * sizeof(float));
        for (int64_t i = 0; i < count; ++i)
            values[i] = j == 3 ? -.1f * (i + 1) : ((i * (j + 3) + j) % 29 - 14) * .07f;
        if (j == 2) {
            values[0] = -25.f;
            values[1] = 25.f;
        }
        ggml_backend_tensor_set(inputs[j], values, 0, count * sizeof(float));
        free(values);
    }
    const int32_t indices[] = {2, 0};
    ggml_backend_tensor_set(ids, indices, 0, sizeof(indices));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        return 3;
    const size_t bytes = ggml_nbytes(out);
    void* output = malloc(bytes);
    ggml_backend_tensor_get(out, output, 0, bytes);
    FILE* file = fopen(argv[2], "wb");
    if (!file || fwrite(output, 1, bytes, file) != bytes)
        return 4;
    fclose(file);
    free(output);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
