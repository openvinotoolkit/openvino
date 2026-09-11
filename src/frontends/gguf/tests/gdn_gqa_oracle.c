// Standalone ggml-CPU oracle for GGML_OP_GATED_DELTA_NET with GQA (H_v != H_k).
// Real qwen3.5 has num_v_heads=32, num_k_heads=16 (ratio 2): every V head pairs with Q/K head
// (v_head % H_k). None of the existing GDN op-tests exercise this (they all use H_k==H_v), so a
// wrong Q/K->V head mapping in the frontend would be invisible to them yet corrupt every real
// token. This oracle uses H_v=4, H_k=2 (ratio 2) with DISTINCT per-head inputs so a mis-pairing
// (e.g. grouped vs interleaved repeat) produces a different output. Authoritative: ggml's own kernel.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    const int64_t B = 1, T = 2, H_v = 4, H_k = 2, D = 2;  // GQA: 4 v-heads, 2 k-heads, ratio 2

    struct ggml_init_params ip = { 64*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    // ggml layout (ne[0] innermost):
    //   q,k : [S_k, H_k, n_tokens, n_seqs]
    //   v   : [S_v, H_v, n_tokens, n_seqs]
    //   g   : [1,   H_v, n_tokens, n_seqs]   scalar gate
    //   beta: [1,   H_v, n_tokens, n_seqs]
    //   state:[S_v, S_v, H_v, n_seqs]
    struct ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H_k, T, B);
    struct ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H_k, T, B);
    struct ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H_v, T, B);
    struct ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, T, B);
    struct ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, T, B);
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D, H_v, B);

    ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v");
    ggml_set_name(g, "g"); ggml_set_name(beta, "beta"); ggml_set_name(state, "state");

    struct ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, /*K=*/1);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;

    // DISTINCT per-head q/k so head0 and head1 differ (else GQA pairing is unobservable).
    // Layout [T, H_k, D]:  t0{h0,h1}, t1{h0,h1}
    float qd[]   = {1,0, 0,1,   0,1, 1,0};   // [T, H_k=2, D]
    float kd[]   = {1,0, 0,1,   1,1, 0,1};   // [T, H_k=2, D]
    // V has 4 heads; heads 0,2 -> k-head0, heads 1,3 -> k-head1 (ggml pairing v_head % H_k).
    // Layout [T, H_v=4, D]
    float vd[]   = {1,2, 3,4, 5,6, 7,8,   9,10, 11,12, 13,14, 15,16};   // [T, H_v=4, D]
    float gd[]   = {0,0,0,0,  0,0,0,0};      // [T, H_v] scalar gate exp(g)=1
    float bd[]   = {1,1,1,1,  1,1,1,1};      // full update
    float sd[H_v*D*D]; memset(sd, 0, sizeof(sd));  // zero initial state

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
