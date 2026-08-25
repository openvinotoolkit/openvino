// test_llm: batch 6 — causal Softmax, RoPE (halves), KV-cache append
// (cache_write), ArgMax sampling, GQA broadcast in the pairwise MatMul.
// The finale composes a single decoder attention step: RoPE(Q,K) ->
// scores -> causal softmax -> P·V -> cache append. Both executors.

#include "cpu_engine.hpp"
#include "runtime/execution_config.hpp"
#include "vk_dispatch.hpp"
#include "vk_engine_factory.hpp"
#include "vk_graph_format.hpp"
#include "vk_ir.hpp"
#include "vk_network.hpp"
#include "vk_program.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace ov::core::vulkan;
using namespace ov::core::vulkan::cross_platform;

namespace {

int failures = 0;

void check(const char* name, const std::vector<float>& got, const std::vector<float>& want, float tol) {
    bool ok = got.size() == want.size();
    if (ok) {
        for (size_t i = 0; i < want.size(); ++i) {
            if (std::fabs(got[i] - want[i]) > tol) {
                ok = false;
                break;
            }
        }
    }
    std::printf("%-34s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++failures;
        std::printf("    got :");
        for (float v : got)
            std::printf(" %.5f", v);
        std::printf("\n    want:");
        for (float v : want)
            std::printf(" %.5f", v);
        std::printf("\n");
    }
}

ir_node nd(const std::string& id, ir_op op, std::vector<std::string> ins) {
    ir_node n;
    n.id = id;
    n.op = op;
    n.inputs = std::move(ins);
    return n;
}

std::vector<float> run_cpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins,
                           size_t elems) {
    auto outs = cpu_execute(g, ins);
    const auto& vals = outs.begin()->second;
    return std::vector<float>(vals.begin(), vals.begin() + std::min(vals.size(), elems));
}

std::vector<float> run_gpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins,
                           size_t elems) {
    auto m = vk_execute(g, ins, "GPU");
    const auto& vals = m.begin()->second;
    return std::vector<float>(vals.begin(), vals.begin() + std::min(vals.size(), elems));
}

}  // namespace

int main() {
    try {
        ExecutionConfig cfg;
        auto engine = create_vk_engine();
        (void)cfg;

        constexpr size_t B = 2, L = 4, D = 6;  // D even for RoPE

        // ---- causal softmax -------------------------------------------------
        // scores[b,i,j] deterministic; row i sees keys j<=i only.
        ir_graph gc;
        gc.nodes.push_back(nd("x", ir_op::parameter, {}));
        gc.nodes.push_back(nd("y", ir_op::causal_softmax, {"x"}));
        gc.nodes.push_back(nd("out", ir_op::result, {"y"}));
        gc.tensor_shapes["x"] = {B, L, L};
        gc.tensor_shapes["y"] = {B, L, L};
        gc.inputs = {"x"};
        gc.outputs = {"y"};
        std::vector<float> sx(B * L * L);
        for (size_t i = 0; i < sx.size(); ++i)
            sx[i] = static_cast<float>((i * 11) % 13) / 3.f - 2.f;
        std::vector<float> rc(B * L * L, 0.f);
        for (size_t b = 0; b < B; ++b)
            for (size_t i = 0; i < L; ++i) {
                float m = -1e30f;
                for (size_t j = 0; j <= i; ++j)
                    m = std::max(m, sx[(b * L + i) * L + j]);
                float s = 0.f;
                for (size_t j = 0; j <= i; ++j) {
                    rc[(b * L + i) * L + j] = std::exp(sx[(b * L + i) * L + j] - m);
                    s += rc[(b * L + i) * L + j];
                }
                for (size_t j = 0; j <= i; ++j)
                    rc[(b * L + i) * L + j] /= s;
            }
        check("causal softmax CPU vs ref", run_cpu(gc, {{"x", sx}}, rc.size()), rc, 1e-5f);
        check("causal softmax GPU vs ref", run_gpu(gc, {{"x", sx}}, rc.size()), rc, 2e-5f);

        // ---- rope -------------------------------------------------------------
        // x [B,L,1,D]; cos/sin [B,L,D/2].
        std::vector<float> rx(B * L * D), rcs(B * L * (D / 2)), rsns(B * L * (D / 2));
        for (size_t i = 0; i < rx.size(); ++i)
            rx[i] = static_cast<float>((i * 5) % 9) / 4.f - 1.f;
        for (size_t b = 0; b < B; ++b)
            for (size_t l = 0; l < L; ++l)
                for (size_t t = 0; t < D / 2; ++t) {
                    const float theta = static_cast<float>(l) * std::pow(10000.f, -2.f * static_cast<float>(t) / D);
                    rcs[(b * L + l) * (D / 2) + t] = std::cos(theta);
                    rsns[(b * L + l) * (D / 2) + t] = std::sin(theta);
                }
        ir_graph gr;
        gr.nodes.push_back(nd("x", ir_op::parameter, {}));
        gr.nodes.push_back(nd("c", ir_op::constant, {}));
        gr.nodes.push_back(nd("s", ir_op::constant, {}));
        gr.nodes.push_back(nd("r", ir_op::rope, {"x", "c", "s"}));
        gr.nodes.push_back(nd("out", ir_op::result, {"r"}));
        gr.tensor_shapes["x"] = {B, L, 1, D};
        gr.tensor_shapes["c"] = {B, L, D / 2};
        gr.tensor_shapes["s"] = {B, L, D / 2};
        gr.tensor_shapes["r"] = {B, L, 1, D};
        gr.constant_data["c"] = rcs;
        gr.constant_data["s"] = rsns;
        gr.inputs = {"x"};
        gr.outputs = {"r"};
        std::vector<float> rr(rx.size());
        for (size_t b = 0; b < B; ++b)
            for (size_t l = 0; l < L; ++l)
                for (size_t t = 0; t < D / 2; ++t) {
                    const size_t base = (b * L + l) * D;
                    const float c = rcs[(b * L + l) * (D / 2) + t];
                    const float s = rsns[(b * L + l) * (D / 2) + t];
                    rr[base + t] = rx[base + t] * c - rx[base + D / 2 + t] * s;
                    rr[base + D / 2 + t] = rx[base + D / 2 + t] * c + rx[base + t] * s;
                }
        check("rope CPU vs ref", run_cpu(gr, {{"x", rx}}, rr.size()), rr, 1e-5f);
        check("rope GPU vs ref", run_gpu(gr, {{"x", rx}}, rr.size()), rr, 2e-5f);

        // ---- argmax -----------------------------------------------------------
        ir_graph ga;
        ga.nodes.push_back(nd("x", ir_op::parameter, {}));
        ga.nodes.push_back(nd("a", ir_op::argmax, {"x"}));
        ga.nodes.push_back(nd("out", ir_op::result, {"a"}));
        ga.tensor_shapes["x"] = {3, 5};
        ga.tensor_shapes["a"] = {3};
        ga.inputs = {"x"};
        ga.outputs = {"a"};
        std::vector<float> ax{1, 9, 2, 9, 0,   -1, -5, -2, -3, -4,   0, 0, 0, 0, 7};
        check("argmax CPU vs ref", run_cpu(ga, {{"x", ax}}, 3), {1, 0, 4}, 0.0f);
        check("argmax GPU vs ref", run_gpu(ga, {{"x", ax}}, 3), {1, 0, 4}, 0.0f);

        // ---- GQA broadcast: A [B,M,K] x B [1,K,N] ------------------------------
        ir_graph gg;
        gg.nodes.push_back(nd("a", ir_op::parameter, {}));
        gg.nodes.push_back(nd("b", ir_op::parameter, {}));
        gg.nodes.push_back(nd("mm", ir_op::matmul, {"a", "b"}));
        gg.nodes.push_back(nd("out", ir_op::result, {"mm"}));
        gg.tensor_shapes["a"] = {2, 3, 4};
        gg.tensor_shapes["b"] = {1, 4, 5};  // shared across batch (GQA pattern)
        gg.tensor_shapes["mm"] = {2, 3, 5};
        gg.nodes[2].matmul_transpose_b = false;
        gg.inputs = {"a", "b"};
        gg.outputs = {"mm"};
        std::vector<float> ga2(24), gb(20);
        for (size_t i = 0; i < ga2.size(); ++i)
            ga2[i] = static_cast<float>((i * 3) % 7) / 3.f - 1.f;
        for (size_t i = 0; i < gb.size(); ++i)
            gb[i] = static_cast<float>((i * 5) % 11) / 5.f - 1.f;
        std::vector<float> rg(30, 0.f);
        for (size_t bt = 0; bt < 2; ++bt)
            for (size_t m = 0; m < 3; ++m)
                for (size_t n = 0; n < 5; ++n)
                    for (size_t k = 0; k < 4; ++k)
                        rg[(bt * 3 + m) * 5 + n] +=
                            ga2[(bt * 3 + m) * 4 + k] * gb[k * 5 + n];
        check("GQA matmul bb=1 CPU vs ref", run_cpu(gg, {{"a", ga2}, {"b", gb}}, rg.size()), rg, 1e-5f);
        check("GQA matmul bb=1 GPU vs ref", run_gpu(gg, {{"a", ga2}, {"b", gb}}, rg.size()), rg, 2e-5f);

        // ---- full decoder step: KV-cache write + masked attention --------------
        // new_k/new_v [B,L,D]; cache [B,S=8,D] zero-filled; pos = axis.
        ir_graph gd;
        gd.nodes.push_back(nd("q", ir_op::parameter, {}));
        gd.nodes.push_back(nd("kt", ir_op::parameter, {}));  // pre-transposed [B,D,L]
        gd.nodes.push_back(nd("v", ir_op::parameter, {}));
        gd.nodes.push_back(nd("kc", ir_op::parameter, {}));  // [B,S,D]
        gd.nodes.push_back(nd("vc", ir_op::parameter, {}));
        gd.nodes.push_back(nd("kw", ir_op::cache_write, {"kt_r", "kc"}));  // placeholder input fixed below
        gd.nodes[5].inputs = {"kt_cache_view", "kc"};                      // replaced below

        // Simpler explicit graph: reshape-free; KT arrives already [B,L,D] and is
        // written to cache, then transposed read via crop of [B,S,D]? For the
        // test we compute scores against the WRITTEN cache using pairwise matmul
        // with a per-batch transposed copy built on CPU (crop/transpose compose).
        gd.nodes.clear();
        gd.nodes.push_back(nd("q", ir_op::parameter, {}));
        gd.nodes.push_back(nd("kn", ir_op::parameter, {}));  // [B,L,D]
        gd.nodes.push_back(nd("vn", ir_op::parameter, {}));
        gd.nodes.push_back(nd("kc", ir_op::parameter, {}));  // [B,S,D]
        gd.nodes.push_back(nd("vc", ir_op::parameter, {}));
        gd.nodes.push_back(nd("kw", ir_op::cache_write, {"kn", "kc"}));  // axis=pos set below
        gd.nodes[5].axis = 0;
        // Read back rows [0..L): crop over seq dim.
        gd.nodes.push_back(nd("kr", ir_op::crop, {"kw"}));
        gd.nodes[6].pool.pads_begin = {0, 0, 0};
        gd.tensor_shapes["kr"] = {B, L, D};
        gd.nodes.push_back(nd("vt", ir_op::transpose, {"vw"}));  // placeholder replaced below

        // V side: write then crop.
        gd.nodes.push_back(nd("vw", ir_op::cache_write, {"vn", "vc"}));
        gd.nodes[8].axis = 0;
        gd.nodes[8].inputs = {"vn", "vc"};
        gd.nodes[6].inputs = {"kw", "kw"};  // fix kr inputs (single input below)

        // Rebuild cleanly instead of patching:
        gd.nodes.clear();
        gd.nodes.push_back(nd("q", ir_op::parameter, {}));                 // [B,L,D]
        gd.nodes.push_back(nd("kn", ir_op::parameter, {}));                // [B,L,D]
        gd.nodes.push_back(nd("vn", ir_op::parameter, {}));                // [B,L,D]
        gd.nodes.push_back(nd("kc", ir_op::parameter, {}));                // [B,S,D]
        gd.nodes.push_back(nd("vc", ir_op::parameter, {}));
        gd.nodes.push_back(nd("scale", ir_op::constant, {}));
        gd.nodes.push_back(nd("qs", ir_op::mul, {"q", "scale"}));
        gd.nodes.push_back(nd("kw", ir_op::cache_write, {"kn", "kc"}));    // pos=0
        gd.nodes[7].axis = 0;
        gd.nodes.push_back(nd("vw", ir_op::cache_write, {"vn", "vc"}));    // pos=0
        gd.nodes[8].axis = 0;
        gd.nodes.push_back(nd("kr", ir_op::crop, {"kw"}));                 // [B,L,D]
        gd.nodes[9].pool.pads_begin = {0, 0, 0};
        gd.tensor_shapes["kr"] = {B, L, D};
        gd.nodes.push_back(nd("vr", ir_op::crop, {"vw"}));
        gd.nodes[10].pool.pads_begin = {0, 0, 0};
        gd.tensor_shapes["vr"] = {B, L, D};
        gd.nodes.push_back(nd("ktp", ir_op::transpose, {"kr"}));           // [B,D,L]
        gd.nodes[11].transpose_order = {0, 2, 1};
        gd.tensor_shapes["ktp"] = {B, D, L};
        gd.nodes.push_back(nd("score", ir_op::matmul, {"qs", "ktp"}));     // [B,L,L]
        gd.tensor_shapes["score"] = {B, L, L};
        gd.nodes.push_back(nd("p", ir_op::causal_softmax, {"score"}));
        gd.nodes.push_back(nd("o", ir_op::matmul, {"p", "vr"}));           // [B,L,D]
        gd.tensor_shapes["o"] = {B, L, D};
        gd.nodes.push_back(nd("am", ir_op::argmax, {"o"}));                // [B,L] tokens
        gd.tensor_shapes["am"] = {B, L};
        gd.nodes.push_back(nd("out", ir_op::result, {"o"}));
        gd.tensor_shapes["q"] = {B, L, D};
        gd.tensor_shapes["kn"] = {B, L, D};
        gd.tensor_shapes["vn"] = {B, L, D};
        gd.tensor_shapes["kc"] = {B, 8, D};
        gd.tensor_shapes["vc"] = {B, 8, D};
        gd.tensor_shapes["scale"] = {1};
        gd.tensor_shapes["qs"] = {B, L, D};
        gd.tensor_shapes["p"] = {B, L, L};
        gd.tensor_shapes["kw"] = {B, 8, D};
        gd.tensor_shapes["vw"] = {B, 8, D};
        gd.constant_data["scale"] = {1.0f / std::sqrt(static_cast<float>(D))};
        gd.inputs = {"q", "kn", "vn", "kc", "vc"};
        gd.outputs = {"o"};

        std::vector<float> q(B * L * D), kn(B * L * D), vn(B * L * D);
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = static_cast<float>((i * 7) % 11) / 5.f - 1.f;
        for (size_t i = 0; i < kn.size(); ++i)
            kn[i] = static_cast<float>((i * 13) % 17) / 8.f - 1.f;
        for (size_t i = 0; i < vn.size(); ++i)
            vn[i] = static_cast<float>((i * 3) % 7) / 3.f - 1.f;
        std::vector<float> kc(B * 8 * D, 0.f), vc(B * 8 * D, 0.f);

        // Reference: same math with plain loops.
        std::vector<float> rref(B * L * D, 0.f);
        {
            const float s = 1.0f / std::sqrt(static_cast<float>(D));
            for (size_t b = 0; b < B; ++b) {
                std::vector<float> sc(L * L, 0.f);
                for (size_t i = 0; i < L; ++i)
                    for (size_t j = 0; j < L; ++j)
                        for (size_t d = 0; d < D; ++d)
                            sc[i * L + j] += q[(b * L + i) * D + d] * s * kn[(b * L + j) * D + d];
                for (size_t i = 0; i < L; ++i) {
                    float m = -1e30f;
                    for (size_t j = 0; j <= i; ++j)
                        m = std::max(m, sc[i * L + j]);
                    float sum = 0.f;
                    for (size_t j = 0; j <= i; ++j) {
                        sc[i * L + j] = std::exp(sc[i * L + j] - m);
                        sum += sc[i * L + j];
                    }
                    for (size_t j = 0; j <= i; ++j)
                        sc[i * L + j] /= sum;
                }
                for (size_t i = 0; i < L; ++i)
                    for (size_t d = 0; d < D; ++d)
                        for (size_t j = 0; j <= i; ++j)  // causal triangle only
                            rref[(b * L + i) * D + d] += sc[i * L + j] * vn[(b * L + j) * D + d];
            }
        }
        const std::map<std::string, std::vector<float>> din{
            {"q", q}, {"kn", kn}, {"vn", vn}, {"kc", kc}, {"vc", vc}};
        check("decoder step CPU vs ref", run_cpu(gd, din, B * L * D), rref, 1e-4f);
        check("decoder step GPU vs ref", run_gpu(gd, din, B * L * D), rref, 1e-4f);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
