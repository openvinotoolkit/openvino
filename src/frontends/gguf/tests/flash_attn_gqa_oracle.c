// Standalone ggml-CPU reference for FlashAttnExtFlatKvGqaWithoutMask.
// Compile against llama.cpp's ggml CPU libraries as described in how_to_add_op.md.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    struct ggml_init_params ip = {64 * 1024 * 1024, NULL, true};
    struct ggml_context * ctx = ggml_init(ip);
    // ggml dimensions are [D, T, H, B]; flat data is [H, T, D] for q/k/v.
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 4, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 2, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 2, 1);
    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f, 0.0f, 0.0f);
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const float q_data[] = {1, 0, 0, 1, 0.5f, 1, 1, -0.5f, -1, 1, 1, 0, 0, -1, 0.5f, 0.5f};
    const float k_data[] = {1, 0, 0, 1, 0.5f, 0.5f, -0.5f, 1};
    const float v_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    ggml_backend_tensor_set(q, q_data, 0, sizeof(q_data));
    ggml_backend_tensor_set(k, k_data, 0, sizeof(k_data));
    ggml_backend_tensor_set(v, v_data, 0, sizeof(v_data));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml CPU attention failed\n");
        return 1;
    }

    float result[16];
    ggml_backend_tensor_get(out, result, 0, sizeof(result));
    for (size_t i = 0; i < 16; ++i) {
        printf("%.6f%s", result[i], (i + 1) % 8 ? " " : "\n");
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
