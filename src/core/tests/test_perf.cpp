// test_perf: batch 7 вЂ” validates the tiled MatMul kernels against the naive
// ones and measures GPU throughput (wall clock around dispatch + readback).
// Sizes: correctness at the 16/17 boundary, perf at 256..512.

#include "cpu_engine.hpp"
#include "runtime/execution_config.hpp"
#include "vk_dispatch.hpp"
#include "vk_engine_factory.hpp"
#include "vk_graph_format.hpp"
#include "vk_ir.hpp"
#include "vk_network.hpp"
#include "vk_program.hpp"

#include <chrono>
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

ir_node nd(const std::string& id, ir_op op, std::vector<std::string> ins) {
    ir_node n;
    n.id = id;
    n.op = op;
    n.inputs = std::move(ins);
    return n;
}

ir_graph make_mm(size_t M, size_t K, size_t N) {
    ir_graph g;
    g.nodes.push_back(nd("a", ir_op::parameter, {}));
    g.nodes.push_back(nd("b", ir_op::constant, {}));
    g.nodes.push_back(nd("mm", ir_op::matmul, {"a", "b"}));
    g.nodes.push_back(nd("out", ir_op::result, {"mm"}));
    g.tensor_shapes["a"] = {M, K};
    g.tensor_shapes["b"] = {K, N};
    g.tensor_shapes["mm"] = {M, N};
    g.constant_data["b"] = {};  // filled by the caller before execution
    g.inputs = {"a"};
    g.outputs = {"mm"};
    return g;
}

void fill(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>((i * 37) % 19) / 9.f - 1.f;
}

double now_ms() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1e6;
}

}  // namespace

int main() {
    try {
        ExecutionConfig cfg;
        auto engine = create_vk_engine();
        const size_t K = 512;

        // ---- correctness: boundary sizes around the 16-tile threshold -------
        for (auto [m, n] : {std::pair<size_t, size_t>{15, 15}, {16, 16}, {17, 17}, {33, 48}}) {
            auto g = make_mm(m, 17, n);  // K odd -> partial tiles exercised
            std::vector<float> a(m * 17), b(17 * n);
            fill(a);
            fill(b);
            g.constant_data["b"] = b;
            // CPU reference.
            auto ref_out = cpu_execute(g, {{"a", a}});
            const auto& ref = ref_out.begin()->second;
            // GPU (tiled path picks itself: M,N,K>=16 except first case).
            auto got_map = vk_execute(g, {{"a", a}}, "GPU");
            const auto& got = got_map.begin()->second;
            bool ok = got.size() == ref.size();
            if (ok)
                for (size_t i = 0; i < ref.size(); ++i)
                    if (std::fabs(got[i] - ref[i]) > 2e-3f) {
                        ok = false;
                        break;
                    }
            std::printf("tiled matmul %zux%zux%zu parity vs CPU   %s\n", m, n != 0 ? 17 : 17, n,
                        ok ? "PASS" : "FAIL");
            if (!ok)
                ++failures;
        }

        // ---- throughput -------------------------------------------------------
        for (size_t mn : {256, 512}) {
            auto g = make_mm(mn, K, mn);
            std::vector<float> a(mn * K), b(K * mn);
            fill(a);
            fill(b);
            g.constant_data["b"] = b;
            // Build once; time pure execution.
            vk_program_builder builder(*engine, cfg);
            auto prog = builder.build(g);
            vk_network net(*engine, cfg, prog);
            const layout lay(g.tensor_shapes["a"], 4);
            auto amem = engine->allocate_memory(lay, allocation_type::usm_host, true);
            std::memcpy(amem->lock(), a.data(), a.size() * sizeof(float));
            amem->unlock();
            net.set_input_data("a", amem);
            const std::string out_id = prog->output_port_to_id.begin()->second;
            auto out_mem =
                engine->allocate_memory(layout(g.tensor_shapes["mm"], 4), allocation_type::usm_host, true);
            net.set_output_memory(out_id, out_mem, false);

            net.execute({});  // warmup
            const int iters = 20;
            const double t0 = now_ms();
            for (int it = 0; it < iters; ++it)
                net.execute({});
            const double ms = (now_ms() - t0) / iters;

            // Sanity: result matches CPU reference on a small crop.
            const float* p = static_cast<const float*>(out_mem->lock());
            double chk = 0.0;
            for (size_t i = 0; i < 64; ++i)
                chk += p[i];
            out_mem->unlock();

            auto ref_out = cpu_execute(g, {{"a", a}});
            double ref_chk = 0.0;
            const auto& refv = ref_out.begin()->second;
            for (size_t i = 0; i < 64 && i < refv.size(); ++i)
                ref_chk += refv[i];

            const double gflop = 2.0 * mn * K * mn / 1e9;
            std::printf("GPU matmul %zux%zux%zu tiled: %.3f ms  %.1f GFLOP/s  (chk %.3f vs %.3f %s)\n",
                        mn, K, mn, ms, gflop / (ms / 1e3), chk, ref_chk,
                        std::fabs(chk - ref_chk) < 1.0 ? "ok" : "BAD");
            if (!(std::fabs(chk - ref_chk) < 1.0))
                ++failures;
        }

        // ---- benchmark #1: decoder step (attention + KV-cache) -----------------
        {
            constexpr size_t B = 1, L = 128, D = 256;
            ir_graph g;
            g.nodes.push_back(nd("q", ir_op::parameter, {}));
            g.nodes.push_back(nd("kn", ir_op::parameter, {}));
            g.nodes.push_back(nd("vn", ir_op::parameter, {}));
            g.nodes.push_back(nd("kc", ir_op::parameter, {}));
            g.nodes.push_back(nd("vc", ir_op::parameter, {}));
            g.nodes.push_back(nd("scale", ir_op::constant, {}));
            g.nodes.push_back(nd("qs", ir_op::mul, {"q", "scale"}));
            g.nodes.push_back(nd("kw", ir_op::cache_write, {"kn", "kc"}));
            g.nodes[7].axis = L;  // append at the tail of a full cache
            g.nodes.push_back(nd("vw", ir_op::cache_write, {"vn", "vc"}));
            g.nodes[8].axis = L;
            g.nodes.push_back(nd("kr", ir_op::crop, {"kw"}));
            g.nodes[9].pool.pads_begin = {0, 0, 0};
            g.tensor_shapes["kr"] = {B, L, D};
            g.nodes.push_back(nd("ktp", ir_op::transpose, {"kr"}));
            g.nodes[10].transpose_order = {0, 2, 1};
            g.tensor_shapes["ktp"] = {B, D, L};
            g.nodes.push_back(nd("score", ir_op::matmul, {"qs", "ktp"}));
            g.tensor_shapes["score"] = {B, L, L};
            g.nodes.push_back(nd("p", ir_op::causal_softmax, {"score"}));
            g.nodes.push_back(nd("vr", ir_op::crop, {"vw"}));
            g.nodes[13].pool.pads_begin = {0, 0, 0};
            g.tensor_shapes["vr"] = {B, L, D};
            g.nodes.push_back(nd("o", ir_op::matmul, {"p", "vr"}));
            g.tensor_shapes["o"] = {B, L, D};
            g.nodes.push_back(nd("am", ir_op::argmax, {"o"}));
            g.tensor_shapes["am"] = {B, L};
            g.nodes.push_back(nd("out", ir_op::result, {"o"}));
            for (const char* t : {"q", "kn", "vn"}) g.tensor_shapes[t] = {B, L, D};
            g.tensor_shapes["kc"] = {B, 2 * L, D};
            g.tensor_shapes["vc"] = {B, 2 * L, D};
            g.tensor_shapes["kw"] = {B, 2 * L, D};
            g.tensor_shapes["vw"] = {B, 2 * L, D};
            g.tensor_shapes["qs"] = {B, L, D};
            g.tensor_shapes["scale"] = {1};
            g.tensor_shapes["p"] = {B, L, L};
            g.constant_data["scale"] = {1.0f / std::sqrt(static_cast<float>(D))};
            g.inputs = {"q", "kn", "vn", "kc", "vc"};
            g.outputs = {"o"};

            std::vector<float> q(B * L * D), kn(B * L * D), vn(B * L * D);
            fill(q); fill(kn); fill(vn);
            std::vector<float> kc(B * 2 * L * D, 0.f), vc(B * 2 * L * D, 0.f);

            vk_program_builder builder(*engine, cfg);
            auto prog = builder.build(g);
            vk_network net(*engine, cfg, prog);
            const auto bind = [&](const char* id, const std::vector<float>& v) {
                auto mem = engine->allocate_memory(layout(g.tensor_shapes.at(id), 4),
                                                   allocation_type::usm_host, true);
                std::memcpy(mem->lock(), v.data(), v.size() * sizeof(float));
                mem->unlock();
                net.set_input_data(id, mem);
            };
            bind("q", q); bind("kn", kn); bind("vn", vn); bind("kc", kc); bind("vc", vc);
            const std::string out_id = prog->output_port_to_id.begin()->second;
            auto out_mem = engine->allocate_memory(layout({B, L, D}, 4), allocation_type::usm_host, true);
            net.set_output_memory(out_id, out_mem, false);

            net.execute({});  // warmup
            const int iters = 20;
            const double t0 = now_ms();
            for (int it = 0; it < iters; ++it)
                net.execute({});
            const double ms = (now_ms() - t0) / iters;
            std::printf("decoder step B=%zu L=%zu D=%zu: %.3f ms/step  (~%.0f tokens/s)\n",
                        B, L, D, ms, L / (ms / 1e3));
        }

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}


