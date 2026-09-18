// Standalone ggml-CPU oracle for GGML_OP_SSM_CONV at the qwen3.5/qwen3-next conv dims
// (d_conv=4, n_t>1, multiple channels). Builds one ggml_ssm_conv node, runs it on the ggml
// CPU backend, and prints the flat output so the OV frontend op-test can assert against ggml's
// own kernel instead of a hand-derived depthwise-conv reference.
//
// ggml layout:
//   sx (conv state + input) : 3D [ncs, d_inner, n_s], ncs = n_t + d_conv - 1
//   c  (conv1d weight)      : 2D [d_conv, d_inner]
//   out                     : 3D [d_inner, n_t, n_s]
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const int64_t d_conv = 4, d_inner = 3, n_t = 5, n_s = 1;
    const int64_t ncs = n_t + d_conv - 1;  // 8

    struct ggml_init_params ip = { 64*1024*1024, NULL, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, d_inner, n_s);
    struct ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
    ggml_set_name(sx, "sx"); ggml_set_name(c, "c");

    struct ggml_tensor * out = ggml_ssm_conv(ctx, sx, c);
    ggml_set_name(out, "out");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;

    // sx: channel-major [d_inner][ncs]; ch0 = 1..8, ch1 = 11..18, ch2 = 21..28
    float sxd[3*8];
    for (int ch = 0; ch < d_inner; ++ch)
        for (int j = 0; j < ncs; ++j)
            sxd[ch*ncs + j] = (float)(ch*10 + j + 1);
    // c: per-channel kernel [d_inner][d_conv]; distinct so channel mixing would show up
    float cd[3*4] = {
        0.5f, 0.25f, 0.125f, 0.0625f,   // ch0
        1.0f, 0.0f,  0.0f,   0.0f,       // ch1 (picks earliest window element)
        0.0f, 0.0f,  0.0f,   1.0f,       // ch2 (picks latest window element)
    };
    ggml_backend_tensor_set(sx, sxd, 0, sizeof(sxd));
    ggml_backend_tensor_set(c,  cd,  0, sizeof(cd));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }

    printf("out ne = [%lld, %lld, %lld, %lld]\n",
           (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3]);
    int64_t n = ggml_nelements(out);
    float * od = malloc(n * sizeof(float));
    ggml_backend_tensor_get(out, od, 0, n * sizeof(float));
    printf("flat out (%lld elems), layout [n_s][n_t][d_inner]:\n", (long long)n);
    for (int64_t i = 0; i < n; i++) printf("%.6f%s", od[i], (i+1)%(out->ne[0]) ? " " : "\n");
    printf("\n");
    free(od);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
