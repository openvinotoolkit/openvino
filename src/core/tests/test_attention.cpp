// test_attention: batch 5 — scaled dot-product attention composed purely from
// core ops (mul-scale, pairwise-batched MatMul, Softmax), plus Crop.
//   qs    = Q * (1/sqrt(D))                    [B,L,D] * scalar const
//   score = qs . KT                            pairwise bb: [B,L,D]x[B,D,L]
//   p     = softmax(score, axis = last)        [B,L,L]
//   out   = p . V                              pairwise bb: [B,L,L]x[B,L,D]
// Also crops a window out of a 4D tensor. Both executors.

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

std::vector<float> run_cpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins) {
    auto outs = cpu_execute(g, ins);
    return outs.begin()->second;
}

std::vector<float> run_gpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins) {
    auto outs_map = vk_execute(g, ins, "GPU");
    return outs_map.begin()->second;
}

}  // namespace

int main() {
    try {
        constexpr size_t B = 2, L = 3, D = 4;

        // Deterministic pseudo-random tensors.
        std::vector<float> Q(B * L * D), KT(B * D * L), V(B * L * D);
        for (size_t i = 0; i < Q.size(); ++i)
            Q[i] = static_cast<float>((i * 13) % 17) / 8.f - 1.f;
        for (size_t i = 0; i < KT.size(); ++i)
            KT[i] = static_cast<float>((i * 7 + 3) % 11) / 5.f - 1.f;
        for (size_t i = 0; i < V.size(); ++i)
            V[i] = static_cast<float>((i * 5 + 1) % 9) / 4.f - 1.f;

        ir_graph g;
        g.nodes.push_back(nd("q", ir_op::parameter, {}));                 // [B,L,D]
        g.nodes.push_back(nd("kt", ir_op::parameter, {}));                // [B,D,L]
        g.nodes.push_back(nd("v", ir_op::parameter, {}));                 // [B,L,D]
        g.nodes.push_back(nd("scale", ir_op::constant, {}));
        g.nodes.push_back(nd("qs", ir_op::mul, {"q", "scale"}));
        g.nodes.push_back(nd("score", ir_op::matmul, {"qs", "kt"}));      // [B,L,L]
        g.nodes.push_back(nd("p", ir_op::softmax, {"score"}));
        g.nodes[6].axis = 2;                                              // last axis of [B,L,L]
        g.nodes.push_back(nd("o", ir_op::matmul, {"p", "v"}));            // [B,L,D]
        g.nodes.push_back(nd("out", ir_op::result, {"o"}));

        g.tensor_shapes["q"] = {B, L, D};
        g.tensor_shapes["kt"] = {B, D, L};
        g.tensor_shapes["v"] = {B, L, D};
        g.tensor_shapes["scale"] = {1};       // broadcast-expanded by the core
        g.tensor_shapes["qs"] = {B, L, D};
        g.tensor_shapes["score"] = {B, L, L};
        g.tensor_shapes["p"] = {B, L, L};
        g.tensor_shapes["o"] = {B, L, D};
        g.constant_data["scale"] = {1.0f / std::sqrt(static_cast<float>(D))};
        g.inputs = {"q", "kt", "v"};
        g.outputs = {"o"};

        // Reference.
        const float s = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> ref(B * L * D, 0.0f);
        for (size_t b = 0; b < B; ++b) {
            // scores [L,L]
            std::vector<float> sc(L * L);
            for (size_t i = 0; i < L; ++i)
                for (size_t j = 0; j < L; ++j)
                    for (size_t d = 0; d < D; ++d)
                        sc[i * L + j] += Q[(b * L + i) * D + d] * s * KT[(b * D + d) * L + j];
            for (size_t i = 0; i < L; ++i) {  // softmax rows
                float m = -1e30f;
                for (size_t j = 0; j < L; ++j)
                    m = std::max(m, sc[i * L + j]);
                float sum = 0.f;
                for (size_t j = 0; j < L; ++j) {
                    sc[i * L + j] = std::exp(sc[i * L + j] - m);
                    sum += sc[i * L + j];
                }
                for (size_t j = 0; j < L; ++j)
                    sc[i * L + j] /= sum;
            }
            for (size_t i = 0; i < L; ++i)
                for (size_t d = 0; d < D; ++d)
                    for (size_t j = 0; j < L; ++j)
                        ref[(b * L + i) * D + d] += sc[i * L + j] * V[(b * L + j) * D + d];
        }

        const std::map<std::string, std::vector<float>> ins{
            {"q", Q}, {"kt", KT}, {"v", V}};
        check("attention composition CPU vs ref", run_cpu(g, ins), ref, 1e-4f);
        check("attention composition GPU vs ref", run_gpu(g, ins), ref, 1e-4f);

        // ---- crop ----------------------------------------------------------------
        ir_graph gc;
        gc.nodes.push_back(nd("x", ir_op::parameter, {}));
        gc.nodes.push_back(nd("c", ir_op::crop, {"x"}));
        gc.nodes.push_back(nd("out", ir_op::result, {"c"}));
        gc.tensor_shapes["x"] = {1, 1, 4, 6};
        gc.tensor_shapes["c"] = {1, 1, 2, 3};
        gc.nodes[1].pool.pads_begin = {0, 0, 1, 2};  // begin offsets
        gc.inputs = {"x"};
        gc.outputs = {"c"};
        std::vector<float> xdata(24);
        for (size_t i = 0; i < 24; ++i)
            xdata[i] = static_cast<float>(i);
        std::vector<float> rc;
        for (size_t h = 1; h < 3; ++h)
            for (size_t w = 2; w < 5; ++w)
                rc.push_back(xdata[h * 6 + w]);
        check("crop CPU vs ref", run_cpu(gc, {{"x", xdata}}), rc, 0.0f);
        check("crop GPU vs ref", run_gpu(gc, {{"x", xdata}}), rc, 0.0f);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
