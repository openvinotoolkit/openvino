// Standalone ggml-CPU oracle for GGML_OP_ROPE in IMROPE (interleaved M-RoPE) mode, at qwen3.5's
// exact config (n_dims=64, sections={11,11,10,0}, freq_base=1e7). Text decode uses the same
// position id for all 4 mrope sections, so pos is [p,p,p,p] per token. Prints the flat rotated
// output so the OV frontend op-test can assert against ggml's own IMROPE kernel.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_ROPE_TYPE_IMROPE 40

int main(void) {
    const int64_t head_dim = 64, n_head = 1, n_tokens = 2, batch = 1;
    const int n_dims = 64;
    int sections[4] = {11, 11, 10, 0};

    struct ggml_init_params ip = { 64*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    // a: [head_dim, n_head, n_tokens, batch]
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tokens, batch);
    // pos: mrope needs 4 * n_tokens int32 (4 sections). Layout: [t-sect | h-sect | w-sect | e-sect].
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens * 4);
    ggml_set_name(a, "a"); ggml_set_name(pos, "pos");

    struct ggml_tensor * out = ggml_rope_multi(
        ctx, a, pos, NULL, n_dims, sections, GGML_ROPE_TYPE_IMROPE,
        /*n_ctx_orig*/262144, /*freq_base*/1e7f, /*freq_scale*/1.0f,
        /*ext_factor*/0.0f, /*attn_factor*/1.0f, /*beta_fast*/32.0f, /*beta_slow*/1.0f);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;

    // a: deterministic ramp per token: element e of token t = (t+1) + 0.01*e
    float ad[64*2];
    for (int t = 0; t < n_tokens; ++t)
        for (int e = 0; e < head_dim; ++e)
            ad[t*head_dim + e] = (float)(t+1) + 0.01f*e;
    // positions: token 0 -> pos 0, token 1 -> pos 1, identical across all 4 sections
    int32_t pd[8] = {0,1, 0,1, 0,1, 0,1};
    ggml_backend_tensor_set(a, ad, 0, sizeof(ad));
    ggml_backend_tensor_set(pos, pd, 0, sizeof(pd));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    printf("out ne = [%lld, %lld, %lld, %lld]\n",
           (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3]);
    int64_t n = ggml_nelements(out);
    float * od = malloc(n * sizeof(float));
    ggml_backend_tensor_get(out, od, 0, n * sizeof(float));
    printf("flat out (%lld elems), per token row of head_dim:\n", (long long)n);
    for (int64_t i = 0; i < n; i++) printf("%.6f%s", od[i], (i+1)%(out->ne[0]) ? " " : "\n");
    printf("\n");
    free(od);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
