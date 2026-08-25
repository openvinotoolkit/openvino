// test_shape_ops: batch 4 — QuickGELU, RMSNorm (last axis), Pad (constant
// fill), quantized batched MatMul (Q4_0, blocks along N of the shared
// matrix); plus the last-axis guard for Softmax. Both executors.

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
        ExecutionConfig cfg;
        auto engine = create_vk_engine();
        (void)cfg;
        (void)engine;

        // ---- quick_gelu -------------------------------------------------------
        const std::vector<float> qx{-2.f, -0.5f, 0.f, 0.5f, 1.f, 3.f};
        ir_graph gqg;
        gqg.nodes.push_back(nd("x", ir_op::parameter, {}));
        gqg.nodes.push_back(nd("y", ir_op::quick_gelu, {"x"}));
        gqg.nodes.push_back(nd("out", ir_op::result, {"y"}));
        gqg.tensor_shapes["x"] = {6};
        gqg.tensor_shapes["y"] = {6};
        gqg.inputs = {"x"};
        gqg.outputs = {"y"};
        std::vector<float> rq(qx.size());
        for (size_t i = 0; i < qx.size(); ++i)
            rq[i] = qx[i] / (1.0f + std::exp(-1.702f * qx[i]));
        check("quick_gelu CPU vs ref", run_cpu(gqg, {{"x", qx}}, rq.size()), rq, 1e-5f);
        check("quick_gelu GPU vs ref", run_gpu(gqg, {{"x", qx}}, rq.size()), rq, 2e-5f);

        // ---- rms_norm ---------------------------------------------------------
        const std::vector<float> rx{1.f, -2.f, 3.f, -4.f, 0.5f, -0.25f, -1.5f, 2.f};
        const std::vector<float> rw{2.f, 1.f, 0.5f, 0.25f};
        ir_graph grn;
        grn.nodes.push_back(nd("x", ir_op::parameter, {}));
        grn.nodes.push_back(nd("w", ir_op::constant, {}));
        grn.nodes.push_back(nd("n", ir_op::rms_norm, {"x", "w"}));
        grn.nodes.push_back(nd("out", ir_op::result, {"n"}));
        grn.tensor_shapes["x"] = {2, 4};
        grn.tensor_shapes["w"] = {4};
        grn.tensor_shapes["n"] = {2, 4};
        grn.constant_data["w"] = rw;
        grn.nodes[2].axis = 1;
        grn.nodes[2].alpha = 1e-5f;  // eps
        grn.inputs = {"x"};
        grn.outputs = {"n"};
        std::vector<float> rrn(8);
        for (size_t l = 0; l < 2; ++l) {
            float ss = 0;
            for (size_t i = 0; i < 4; ++i)
                ss += rx[l * 4 + i] * rx[l * 4 + i];
            const float inv = 1.0f / std::sqrt(ss / 4.0f + 1e-5f);
            for (size_t i = 0; i < 4; ++i)
                rrn[l * 4 + i] = rx[l * 4 + i] * inv * rw[i];
        }
        check("rms_norm CPU vs ref", run_cpu(grn, {{"x", rx}}, rrn.size()), rrn, 1e-5f);
        check("rms_norm GPU vs ref", run_gpu(grn, {{"x", rx}}, rrn.size()), rrn, 2e-5f);

        // ---- pad ----------------------------------------------------------------
        ir_graph gp;
        gp.nodes.push_back(nd("x", ir_op::parameter, {}));
        gp.nodes.push_back(nd("p", ir_op::pad, {"x"}));
        gp.nodes.push_back(nd("out", ir_op::result, {"p"}));
        gp.tensor_shapes["x"] = {1, 1, 2, 2};
        gp.tensor_shapes["p"] = {1, 1, 4, 4};
        gp.nodes[1].pool.pads_begin = {0, 0, 1, 1};
        gp.nodes[1].pool.pads_end = {0, 0, 1, 1};
        gp.nodes[1].alpha = 9.0f;  // fill value
        gp.inputs = {"x"};
        gp.outputs = {"p"};
        const std::vector<float> px{1, 2, 3, 4};
        std::vector<float> rp(16, 9.0f);
        rp[5] = 1;
        rp[6] = 2;
        rp[9] = 3;
        rp[10] = 4;
        check("pad CPU vs ref", run_cpu(gp, {{"x", px}}, rp.size()), rp, 0.0f);
        check("pad GPU vs ref", run_gpu(gp, {{"x", px}}, rp.size()), rp, 0.0f);

        // ---- quantized batched matmul -------------------------------------------
        ir_graph gqm;
        gqm.nodes.push_back(nd("x", ir_op::parameter, {}));
        gqm.nodes.push_back(nd("w", ir_op::constant, {}));
        gqm.nodes.push_back(nd("mm", ir_op::matmul, {"x", "w"}));
        gqm.nodes.push_back(nd("out", ir_op::result, {"mm"}));
        gqm.tensor_shapes["x"] = {2, 3, 4};   // B=2, M=3, K=4
        gqm.tensor_shapes["w"] = {4, 32};     // K=4 rows, N=32 -> one Q4_0 block per row
        gqm.tensor_shapes["mm"] = {2, 3, 32};
        gqm.constant_data["w"] = {};
        gqm.quant_constants["w"] = [] {
            ir_quant_const qc;
            qc.type = 2;  // Q4_0
            const uint16_t scales[4] = {0x3C00, 0x3800, 0x4000, 0x3400};  // 1.0/0.5/2.0/0.25
            for (int k = 0; k < 4; ++k) {
                qc.bytes.push_back(static_cast<uint8_t>(scales[k] & 0xFF));
                qc.bytes.push_back(static_cast<uint8_t>(scales[k] >> 8));
                for (int j = 0; j < 16; ++j) {
                    const int lo = (2 * j) % 16;
                    const int hi = (2 * j + 1) % 16;
                    qc.bytes.push_back(static_cast<uint8_t>(lo | (hi << 4)));
                }
            }
            return qc;
        }();
        gqm.inputs = {"x"};
        gqm.outputs = {"mm"};
        const float ds[4] = {1.0f, 0.5f, 2.0f, 0.25f};
        const std::vector<float> qxa{1, 2, -1, 0.5f, 0, -2, 3, 1.5f, 2.5f, 1, -0.5f, -3,
                                     -1, 0.25f, 2, -0.5f, 1.5f, -2, 0, 3, -3, 0.75f, 1.25f, -1.25f};
        std::vector<float> rqm(2 * 3 * 32, 0.0f);
        for (size_t bt = 0; bt < 2; ++bt)
            for (size_t m = 0; m < 3; ++m)
                for (size_t n = 0; n < 32; ++n)
                    for (size_t k = 0; k < 4; ++k)
                        rqm[(bt * 3 + m) * 32 + n] +=
                            qxa[(bt * 3 + m) * 4 + k] * ds[k] * static_cast<float>(static_cast<int>(n % 16) - 8);
        check("quant batched matmul CPU vs ref", run_cpu(gqm, {{"x", qxa}}, rqm.size()), rqm, 1e-3f);
        check("quant batched matmul GPU vs ref", run_gpu(gqm, {{"x", qxa}}, rqm.size()), rqm, 1e-3f);

        // ---- softmax middle axis works since batch 5 ------------------------------
        ir_graph gsm;
        gsm.nodes.push_back(nd("x", ir_op::parameter, {}));
        gsm.nodes.push_back(nd("s", ir_op::softmax, {"x"}));
        gsm.nodes.push_back(nd("out", ir_op::result, {"s"}));
        gsm.tensor_shapes["x"] = {2, 3, 4};
        gsm.tensor_shapes["s"] = {2, 3, 4};
        gsm.nodes[1].axis = 1;  // middle axis
        gsm.inputs = {"x"};
        gsm.outputs = {"s"};
        std::vector<float> sx(24);
        for (size_t i = 0; i < 24; ++i)
            sx[i] = static_cast<float>((i * 7) % 11) - 5.f;
        std::vector<float> rsm_mid(24);
        for (size_t a = 0; a < 2; ++a)          // outer = 2
            for (size_t c = 0; c < 4; ++c) {    // inner = 4
                float m = -1e30f, s = 0.f;
                for (size_t b = 0; b < 3; ++b)
                    m = std::max(m, sx[(a * 3 + b) * 4 + c]);
                for (size_t b = 0; b < 3; ++b) {
                    rsm_mid[(a * 3 + b) * 4 + c] = std::exp(sx[(a * 3 + b) * 4 + c] - m);
                    s += rsm_mid[(a * 3 + b) * 4 + c];
                }
                for (size_t b = 0; b < 3; ++b)
                    rsm_mid[(a * 3 + b) * 4 + c] /= s;
            }
        check("softmax middle-axis CPU vs ref", run_cpu(gsm, {{"x", sx}}, 24), rsm_mid, 1e-5f);
        check("softmax middle-axis GPU vs ref", run_gpu(gsm, {{"x", sx}}, 24), rsm_mid, 2e-5f);

        // ---- FB v5 round-trip keeps pads_end and new ops --------------------------
        const std::vector<ir_graph> graphs{gp};
        auto pb = serialize_pb(graphs);
        auto back = deserialize_pb(pb);
        check_true("round-trip keeps pads_end/fill",
                   back[0].nodes[1].pool.pads_end == std::vector<size_t>{0, 0, 1, 1} &&
                       back[0].nodes[1].alpha == 9.0f && back[0].nodes[1].op == ir_op::pad);
        check("round-trip pad CPU vs ref", run_cpu(back[0], {{"x", px}}, rp.size()), rp, 0.0f);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
