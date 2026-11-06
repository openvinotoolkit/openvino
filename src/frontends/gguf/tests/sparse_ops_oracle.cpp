// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Real ggml CPU references for the sparse-selection and hyper-connection operations.
// Build against a CPU-only llama.cpp checkout (ggml/include, link ggml, ggml-base, ggml-cpu);
// see gen_sparse_ops_accuracy.py. Every tensor is written as <case>.<name>.bin, raw in ggml memory
// order, which is the row-major OpenVINO order of the reversed shape.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace {

std::string g_dir;

void save(const std::string& name, const ggml_tensor* t) {
    FILE* f = std::fopen((g_dir + "/" + name + ".bin").c_str(), "wb");
    std::fwrite(t->data, 1, ggml_nbytes(t), f);
    std::fclose(f);
    FILE* s = std::fopen((g_dir + "/" + name + ".shape").c_str(), "w");
    std::fprintf(s,
                 "%s %lld %lld %lld %lld\n",
                 ggml_type_name(t->type),
                 (long long)t->ne[3],
                 (long long)t->ne[2],
                 (long long)t->ne[1],
                 (long long)t->ne[0]);
    std::fclose(s);
}

ggml_tensor* filled(ggml_context* ctx, ggml_type type, std::vector<int64_t> ne, float offset) {
    auto* t = ggml_new_tensor(ctx, type, static_cast<int>(ne.size()), ne.data());
    const auto n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        const float v = std::sin(0.37f * static_cast<float>(i) + offset) * 1.3f;
        if (type == GGML_TYPE_F32)
            static_cast<float*>(t->data)[i] = v;
        else
            static_cast<ggml_fp16_t*>(t->data)[i] = ggml_fp32_to_fp16(v);
    }
    return t;
}

ggml_tensor* ints(ggml_context* ctx, std::vector<int64_t> ne, std::vector<int32_t> values) {
    auto* t = ggml_new_tensor(ctx, GGML_TYPE_I32, static_cast<int>(ne.size()), ne.data());
    std::memcpy(t->data, values.data(), values.size() * sizeof(int32_t));
    return t;
}

void compute(ggml_context* ctx, ggml_tensor* out) {
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    g_dir = argv[1];
    ggml_init_params params{256 * 1024 * 1024, nullptr, false};
    auto* ctx = ggml_init(params);

    {  // DSA lightning indexer: two streams, mask shared by both.
        auto* q = filled(ctx, GGML_TYPE_F32, {8, 3, 4, 2}, 0.1f);
        auto* k = filled(ctx, GGML_TYPE_F32, {8, 1, 6, 2}, 0.7f);
        auto* w = filled(ctx, GGML_TYPE_F32, {3, 4, 1, 2}, 1.3f);
        auto* m = filled(ctx, GGML_TYPE_F16, {6, 4, 1, 1}, 2.1f);
        for (int t = 0; t < 4; ++t)
            for (int c = t + 3; c < 6; ++c)
                static_cast<ggml_fp16_t*>(m->data)[t * 6 + c] = ggml_fp32_to_fp16(-INFINITY);
        auto* out = ggml_lightning_indexer(ctx, q, k, w, m);
        compute(ctx, out);
        save("indexer.q", q), save("indexer.k", k), save("indexer.w", w), save("indexer.mask", m);
        save("indexer.out", out);
    }
    {  // Top-k mask with live scores, block members, extra cells and out-of-range entries.
        auto* mask = filled(ctx, GGML_TYPE_F32, {10, 3, 1, 1}, 0.3f);
        auto* top_k = ints(ctx, {2, 3, 2, 1}, {0, 2, 1, 3, 3, 0, 2, 1, 1, 0, 3, 2});
        auto* score = filled(ctx, GGML_TYPE_F32, {4, 3, 2, 1}, 0.9f);
        static_cast<float*>(score->data)[1] = -INFINITY;
        static_cast<float*>(score->data)[4 * 3 + 2] = -INFINITY;
        auto* blocks = ints(ctx, {2, 4}, {0, 1, 2, 3, 4, 10, 8, 9});
        auto* extra = ints(ctx, {1, 3, 2, 1}, {5, 10, 7, -1, 6, 4});
        auto* out = ggml_top_k_mask(ctx, mask, top_k, score, blocks, extra);
        compute(ctx, out);
        save("topk.mask", mask), save("topk.top_k", top_k), save("topk.score", score);
        save("topk.blocks", blocks), save("topk.extra", extra), save("topk.out", out);
        auto* plain = ggml_top_k_mask(ctx, mask, top_k, nullptr, nullptr, nullptr);
        compute(ctx, plain);
        save("topk.plain", plain);
    }
    {  // Hyper-connection read: weighted and gated.
        auto* x = filled(ctx, GGML_TYPE_F32, {5, 4, 3}, 0.2f);
        auto* w = filled(ctx, GGML_TYPE_F32, {4, 3}, 1.1f);
        auto* gate = filled(ctx, GGML_TYPE_F32, {5, 4, 3}, 2.3f);
        auto* pre = ggml_dsv4_hc_pre(ctx, x, w);
        auto* gated = ggml_dsv4_hc_pre_gated(ctx, x, gate, 0.25f);
        compute(ctx, pre);
        compute(ctx, gated);
        save("hc_pre.x", x), save("hc_pre.w", w), save("hc_pre.gate", gate);
        save("hc_pre.out", pre), save("hc_pre.gated", gated);
    }
    {  // Hyper-connection write: with and without stream mixing.
        auto* x = filled(ctx, GGML_TYPE_F32, {5, 3}, 0.4f);
        auto* residual = filled(ctx, GGML_TYPE_F32, {5, 4, 3}, 1.7f);
        auto* post = filled(ctx, GGML_TYPE_F32, {4, 3}, 2.9f);
        auto* comb = filled(ctx, GGML_TYPE_F32, {4, 4, 3}, 3.3f);
        auto* mixed = ggml_dsv4_hc_post(ctx, x, residual, post, comb);
        auto* identity = ggml_dsv4_hc_post(ctx, x, residual, post, nullptr);
        compute(ctx, mixed);
        compute(ctx, identity);
        save("hc_post.x", x), save("hc_post.residual", residual), save("hc_post.post", post);
        save("hc_post.comb", comb), save("hc_post.out", mixed), save("hc_post.identity", identity);
    }
    ggml_free(ctx);
    return 0;
}
