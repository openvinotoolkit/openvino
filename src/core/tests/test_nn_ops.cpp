// test_nn_ops: batch 3 — batched MatMul (A [B,M,K] x shared B [K,N|N,K]),
// GELU (tanh approximation) and SwiGLU, on both executors; plus the
// convolution contract (exactly 3 inputs: data, weights, bias).

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

void check_true(const char* name, bool ok) {
    std::printf("%-34s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

ir_node nd(const std::string& id, ir_op op, std::vector<std::string> ins) {
    ir_node n;
    n.id = id;
    n.op = op;
    n.inputs = std::move(ins);
    return n;
}

const std::vector<float> xa_data{1,     2,     -1,    0.5f, 0,      -2,    3,     1.5f,
                                 2.5f,  1,     -0.5f, -3,   0.25f,  -0.75f, 1.25f, 2.f,
                                 -1.5f, 0.5f,  -2.5f, 3.5f, 1.f,    -1.f,   0.f,   2.f};
const std::vector<float> wb_data{0.5f, -1, 2, 0, 1, -0.5f, 0.25f, -2, 1.5f, 0.75f,
                                 -1.25f, 3, -0.25f, 1, -3, 0.5f, 2.5f, -1.5f, 0, 1};

// A [2,3,4] x shared W -> [2,3,5]; transpose_b stores W as [5,4].
ir_graph make_batched_graph(bool transpose_b) {
    ir_graph g;
    g.nodes.push_back(nd("x", ir_op::parameter, {}));
    g.nodes.push_back(nd("w", ir_op::constant, {}));
    g.nodes.push_back(nd("mm", ir_op::matmul, {"x", "w"}));
    g.nodes.push_back(nd("out", ir_op::result, {"mm"}));
    g.tensor_shapes["x"] = {2, 3, 4};
    g.tensor_shapes["w"] = transpose_b ? std::vector<size_t>{5, 4} : std::vector<size_t>{4, 5};
    g.tensor_shapes["mm"] = {2, 3, 5};
    g.constant_data["w"] = wb_data;
    g.nodes[2].matmul_transpose_b = transpose_b;
    g.inputs = {"x"};
    g.outputs = {"mm"};
    return g;
}

std::vector<float> ref_batched(bool transpose_b) {
    const size_t B = 2, M = 3, K = 4, N = 5;
    std::vector<float> out(B * M * N, 0.0f);
    for (size_t bt = 0; bt < B; ++bt)
        for (size_t m = 0; m < M; ++m)
            for (size_t n = 0; n < N; ++n)
                for (size_t k = 0; k < K; ++k) {
                    const float av = xa_data[(bt * M + m) * K + k];
                    const float bv = transpose_b ? wb_data[n * K + k] : wb_data[k * N + n];
                    out[(bt * M + m) * N + n] += av * bv;
                }
    return out;
}

std::vector<float> run_cpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins,
                           size_t expect_elems) {
    auto outs = cpu_execute(g, ins);
    const auto& vals = outs.begin()->second;
    return std::vector<float>(vals.begin(), vals.begin() + std::min(vals.size(), expect_elems));
}

std::vector<float> run_gpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins,
                           size_t expect_elems) {
    auto outs_map = vk_execute(g, ins, "GPU");
    const auto& vals = outs_map.begin()->second;
    return std::vector<float>(vals.begin(), vals.begin() + std::min(vals.size(), expect_elems));
}

}  // namespace

int main() {
    try {
        // ---- batched matmul -------------------------------------------------
        auto gb_plain = make_batched_graph(false);
        const auto rb = ref_batched(false);
        const std::map<std::string, std::vector<float>> xin{{"x", xa_data}, {"w", wb_data}};
        check("batched matmul CPU vs ref", run_cpu(gb_plain, xin, rb.size()), rb, 1e-5f);

        auto gb_tb = make_batched_graph(true);
        const auto rbt = ref_batched(true);
        check("batched matmul+tb CPU vs ref", run_cpu(gb_tb, xin, rbt.size()), rbt, 1e-5f);

        ExecutionConfig cfg;
        auto engine = create_vk_engine();
        check("batched matmul GPU vs ref", run_gpu(gb_plain, xin, rb.size()), rb, 2e-5f);
        check("batched matmul+tb GPU vs ref", run_gpu(gb_tb, xin, rbt.size()), rbt, 2e-5f);

        // ---- gelu -----------------------------------------------------------
        const std::vector<float> gx{-3.f, -1.f, -0.25f, 0.f, 0.25f, 1.f, 2.f, 5.f};
        ir_graph gg;
        gg.nodes.push_back(nd("x", ir_op::parameter, {}));
        gg.nodes.push_back(nd("g", ir_op::gelu, {"x"}));
        gg.nodes.push_back(nd("out", ir_op::result, {"g"}));
        gg.tensor_shapes["x"] = {8};
        gg.tensor_shapes["g"] = {8};
        gg.inputs = {"x"};
        gg.outputs = {"g"};
        std::vector<float> rg(gx.size());
        for (size_t i = 0; i < gx.size(); ++i) {
            const float xi = gx[i];
            rg[i] = 0.5f * xi * (1.0f + std::tanh(0.7978845608028654f * (xi + 0.044715f * xi * xi * xi)));
        }
        check("gelu CPU vs ref", run_cpu(gg, {{"x", gx}}, rg.size()), rg, 1e-5f);
        check("gelu GPU vs ref", run_gpu(gg, {{"x", gx}}, rg.size()), rg, 2e-5f);

        // ---- swiglu ----------------------------------------------------------
        ir_graph gs2;
        gs2.nodes.push_back(nd("a", ir_op::parameter, {}));
        gs2.nodes.push_back(nd("b", ir_op::parameter, {}));
        gs2.nodes.push_back(nd("sw", ir_op::swiglu, {"a", "b"}));
        gs2.nodes.push_back(nd("out", ir_op::result, {"sw"}));
        for (const char* id : {"a", "b", "sw"})
            gs2.tensor_shapes[id] = {2, 4};
        gs2.inputs = {"a", "b"};
        gs2.outputs = {"sw"};
        const std::vector<float> ga{1, -1, 2, -2, 0.5f, -0.5f, 3, -3};
        const std::vector<float> gbv{2, 2, -1, -1, 1, 1, 0.5f, 0.5f};
        std::vector<float> rs(ga.size());
        for (size_t i = 0; i < ga.size(); ++i)
            rs[i] = ga[i] / (1.0f + std::exp(-ga[i])) * gbv[i];
        check("swiglu CPU vs ref", run_cpu(gs2, {{"a", ga}, {"b", gbv}}, rs.size()), rs, 1e-5f);
        check("swiglu GPU vs ref", run_gpu(gs2, {{"a", ga}, {"b", gbv}}, rs.size()), rs, 2e-5f);

        // ---- conv contract: bias is mandatory ---------------------------------
        ir_graph gcb;
        gcb.nodes.push_back(nd("x", ir_op::parameter, {}));
        gcb.nodes.push_back(nd("w", ir_op::constant, {}));
        gcb.nodes.push_back(nd("cv", ir_op::convolution, {"x", "w"}));  // no bias!
        gcb.nodes.push_back(nd("out", ir_op::result, {"cv"}));
        gcb.tensor_shapes["x"] = {1, 1, 4, 4};
        gcb.tensor_shapes["w"] = {1, 1, 2, 2};
        gcb.tensor_shapes["cv"] = {1, 1, 3, 3};
        gcb.constant_data["w"] = {1, 0, 0, 1};
        gcb.inputs = {"x"};
        gcb.outputs = {"cv"};
        bool cpu_rejected = false;
        try {
            (void)run_cpu(gcb, {{"x", std::vector<float>(16, 1.f)}}, 9);
        } catch (const std::exception&) {
            cpu_rejected = true;
        }
        check_true("conv w/o bias CPU rejected", cpu_rejected);

        bool gpu_rejected = false;
        try {
            (void)run_gpu(gcb, {{"x", std::vector<float>(16, 1.f)}}, 9);
        } catch (const std::exception&) {
            gpu_rejected = true;
        }
        check_true("conv w/o bias GPU rejected", gpu_rejected);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
