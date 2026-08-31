// Standalone ggml-CPU oracle for GGML_OP_GATED_DELTA_NET.
// Builds a single GDN node with the SAME tiny multi-head inputs as the OV frontend op-test,
// runs it on the ggml CPU backend, and prints the flat output. This is the authoritative
// reference: ggml's own kernel, not a hand-derived recurrence that might share the frontend's
// layout assumption.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    const int64_t B = 1, T = 2, H = 2, D = 2;  // n_seqs, n_tokens, heads, head-dim (S_k=S_v=D)

    struct ggml_init_params ip = { 64*1024*1024, NULL, true };  // no_alloc: backend allocates
    struct ggml_context * ctx = ggml_init(ip);

    // ggml layout (ne[0] innermost):
    //   q,k : [S_k, H_k, n_tokens, n_seqs]   here H_k=H
    //   v   : [S_v, H_v, n_tokens, n_seqs]
    //   g   : [1,   H_v, n_tokens, n_seqs]   scalar gate
    //   beta: [1,   H_v, n_tokens, n_seqs]
    //   state:[S_v, S_v, H_v, n_seqs]
    struct ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, B);
    struct ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, B);
    struct ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, B);
    struct ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    struct ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H, B);

    ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v");
    ggml_set_name(g, "g"); ggml_set_name(beta, "beta"); ggml_set_name(state, "state");

    struct ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, /*K=*/1);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Allocate on CPU backend and fill inputs (same values/order as the OV op-test, [T,H,D]).
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;

    float qd[]   = {1,0, 0,1,  0,1, 1,0};   // [T,H,D]
    float kd[]   = {1,0, 0,1,  1,1, 0,1};
    float vd[]   = {1,2, 3,4,  5,6, 7,8};
    float gd[]   = {0,0, 0,0};              // [T,H]
    float bd[]   = {1,1, 1,1};
    float sd[8]; memset(sd, 0, sizeof(sd));  // [H,D,D] zero initial state

    ggml_backend_tensor_set(q, qd, 0, sizeof(qd));
    ggml_backend_tensor_set(k, kd, 0, sizeof(kd));
    ggml_backend_tensor_set(v, vd, 0, sizeof(vd));
    ggml_backend_tensor_set(g, gd, 0, sizeof(gd));
    ggml_backend_tensor_set(beta, bd, 0, sizeof(bd));
    ggml_backend_tensor_set(state, sd, 0, sizeof(sd));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    printf("out ne = [%lld, %lld, %lld, %lld]\n",
           (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3]);
    int64_t n = ggml_nelements(out);
    float * od = malloc(n * sizeof(float));
    ggml_backend_tensor_get(out, od, 0, n * sizeof(float));
    printf("flat out (%lld elems):\n", (long long)n);
    for (int64_t i = 0; i < n; i++) printf("%.6f%s", od[i], (i+1)%(out->ne[0]) ? " " : "\n");
    printf("\n");
    free(od);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
