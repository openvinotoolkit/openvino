// test_ops_smoke: every ir_op against analytic references, on both executors
// (cpu_engine and the Vulkan runtime), plus FB/PB round-trips (alpha/axis/
// transpose_order/quant_constants survive) and device-name dispatch checks
// (CPU / GPU / GPU.N / NPU clean error). 37 checks, ALL PASS expected.

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

ir_node nd(const std::string& id, ir_op op, std::vector<std::string> ins, float alpha = 0.0f) {
    ir_node n;
    n.id = id;
    n.op = op;
    n.inputs = std::move(ins);
    n.alpha = alpha;
    return n;
}

ir_graph make_unary_graph() {
    // x -> leaky_relu(0.1) -> sigmoid -> tanh -> out
    ir_graph g;
    g.nodes.push_back(nd("x", ir_op::parameter, {}));
    g.nodes.push_back(nd("lr", ir_op::leaky_relu, {"x"}, 0.1f));
    g.nodes.push_back(nd("sg", ir_op::sigmoid, {"lr"}));
    g.nodes.push_back(nd("th", ir_op::tanh, {"sg"}));
    g.nodes.push_back(nd("out", ir_op::result, {"th"}));
    for (const char* id : {"x", "lr", "sg", "th"})
        g.tensor_shapes[id] = {1, 1, 2, 3};
    g.inputs = {"x"};
    g.outputs = {"th"};
    return g;
}

ir_graph make_binary_graph() {
    // m = a*b; s = m-b; d = s/a; out = d
    ir_graph g;
    g.nodes.push_back(nd("a", ir_op::parameter, {}));
    g.nodes.push_back(nd("b", ir_op::parameter, {}));
    g.nodes.push_back(nd("m", ir_op::mul, {"a", "b"}));
    g.nodes.push_back(nd("s", ir_op::sub, {"m", "b"}));
    g.nodes.push_back(nd("d", ir_op::div, {"s", "a"}));
    g.nodes.push_back(nd("out", ir_op::result, {"d"}));
    for (const char* id : {"a", "b", "m", "s", "d"})
        g.tensor_shapes[id] = {1, 1, 2, 2};
    g.inputs = {"a", "b"};
    g.outputs = {"d"};
    return g;
}

ir_graph make_legacy_graph() {
    // Regression guard for pre-existing ops: y = relu(x) @ w + bias
    ir_graph g;
    g.nodes.push_back(nd("x", ir_op::parameter, {}));
    g.nodes.push_back(nd("r", ir_op::relu, {"x"}));
    g.nodes.push_back(nd("w", ir_op::constant, {}));
    g.nodes.push_back(nd("mm", ir_op::matmul, {"r", "w"}));
    g.nodes.push_back(nd("bias", ir_op::constant, {}));
    g.nodes.push_back(nd("y", ir_op::add, {"mm", "bias"}));
    g.nodes.push_back(nd("out", ir_op::result, {"y"}));
    g.tensor_shapes["x"] = {2, 3};
    g.tensor_shapes["r"] = {2, 3};
    g.tensor_shapes["w"] = {3, 2};
    g.tensor_shapes["mm"] = {2, 2};
    g.tensor_shapes["bias"] = {2, 2};  // broadcast inputs are materialized upstream (core contract)
    g.tensor_shapes["y"] = {2, 2};
    g.constant_data["w"] = {1.0f, -1.0f, 0.5f, 2.0f, 0.0f, 1.5f};
    g.constant_data["bias"] = {10.0f, -10.0f, 10.0f, -10.0f};
    g.inputs = {"x"};
    g.outputs = {"y"};
    return g;
}

ir_graph make_legacy_bcast_graph() {
    // Same as legacy, but the bias keeps its native [1,2] broadcast shape:
    // the core must materialize the constant expansion itself now.
    ir_graph g = make_legacy_graph();
    g.tensor_shapes["bias"] = {1, 2};
    g.constant_data["bias"] = {10.0f, -10.0f};
    return g;
}

// Q4_0 matmul: x [3,4] @ w[4,32], one 18-byte Q4_0 block per K row
// (blocks run along N; N=32 -> exactly one block per row).
ir_graph make_quant_graph() {
    ir_graph g;
    g.nodes.push_back(nd("x", ir_op::parameter, {}));
    g.nodes.push_back(nd("w", ir_op::constant, {}));
    g.nodes.push_back(nd("mm", ir_op::matmul, {"x", "w"}));
    g.nodes.push_back(nd("out", ir_op::result, {"mm"}));
    g.tensor_shapes["x"] = {3, 4};
    g.tensor_shapes["w"] = {4, 32};
    g.tensor_shapes["mm"] = {3, 32};
    ir_quant_const qc;
    qc.type = 2;  // Q4_0: f16 scale + 16 nibble bytes per 32 weights
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
    g.quant_constants["w"] = std::move(qc);
    g.inputs = {"x"};
    g.outputs = {"mm"};
    return g;
}

// ---- batch 2: shape ops ------------------------------------------------------

ir_node ndt(const std::string& id, ir_op op, std::vector<std::string> ins,
            std::vector<size_t> order = {}, uint32_t axis = 0) {
    ir_node n = nd(id, op, std::move(ins));
    n.transpose_order = std::move(order);
    n.axis = axis;
    return n;
}

// NCHW -> NHWC: x[1,2,2,3] --perm{0,2,3,1}--> t[1,2,3,2]
ir_graph make_transpose_graph() {
    ir_graph g;
    g.nodes.push_back(ndt("x", ir_op::parameter, {}));
    g.nodes.push_back(ndt("t", ir_op::transpose, {"x"}, {0, 2, 3, 1}));
    g.nodes.push_back(ndt("out", ir_op::result, {"t"}));
    g.tensor_shapes["x"] = {1, 2, 2, 3};
    g.tensor_shapes["t"] = {1, 2, 3, 2};
    g.inputs = {"x"};
    g.outputs = {"t"};
    return g;
}

// relu -> reshape [2,3] -> [6]
ir_graph make_reshape_graph() {
    ir_graph g;
    g.nodes.push_back(ndt("x", ir_op::parameter, {}));
    g.nodes.push_back(nd("r", ir_op::relu, {"x"}));
    g.nodes.push_back(ndt("s", ir_op::reshape, {"r"}));
    g.nodes.push_back(ndt("out", ir_op::result, {"s"}));
    g.tensor_shapes["x"] = {2, 3};
    g.tensor_shapes["r"] = {2, 3};
    g.tensor_shapes["s"] = {6};
    g.inputs = {"x"};
    g.outputs = {"s"};
    return g;
}

// concat(a[2,3], b[2,2], axis=1) -> [2,5]
ir_graph make_concat_graph() {
    ir_graph g;
    g.nodes.push_back(ndt("a", ir_op::parameter, {}));
    g.nodes.push_back(ndt("b", ir_op::parameter, {}));
    g.nodes.push_back(ndt("c", ir_op::concat, {"a", "b"}, {}, 1));
    g.nodes.push_back(ndt("out", ir_op::result, {"c"}));
    g.tensor_shapes["a"] = {2, 3};
    g.tensor_shapes["b"] = {2, 2};
    g.tensor_shapes["c"] = {2, 5};
    g.inputs = {"a", "b"};
    g.outputs = {"c"};
    return g;
}

// softmax(axis=1) on [2,4]
ir_graph make_softmax_graph() {
    ir_graph g;
    g.nodes.push_back(ndt("x", ir_op::parameter, {}));
    g.nodes.push_back(ndt("sm", ir_op::softmax, {"x"}, {}, 1));
    g.nodes.push_back(ndt("out", ir_op::result, {"sm"}));
    g.tensor_shapes["x"] = {2, 4};
    g.tensor_shapes["sm"] = {2, 4};
    g.inputs = {"x"};
    g.outputs = {"sm"};
    return g;
}

ir_graph make_reduce_graph(ir_op op) {
    ir_graph g;
    g.nodes.push_back(ndt("x", ir_op::parameter, {}));
    g.nodes.push_back(ndt("rd", op, {"x"}, {}, 1));
    g.nodes.push_back(ndt("out", ir_op::result, {"rd"}));
    g.tensor_shapes["x"] = {2, 4};
    g.tensor_shapes["rd"] = {2};
    g.inputs = {"x"};
    g.outputs = {"rd"};
    return g;
}

const std::vector<float> x4_data{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
const std::vector<float> ca_data{1, 2, 3, 4, 5, 6};
const std::vector<float> cb_data{7, 8, 9, 10};
const std::vector<float> xs_data{1, 2, 3, 4, -1, 0, 1, 2};

std::vector<float> ref_transpose_nhwc() {
    // x[n,c,h,w] with dims {1,2,2,3}; out dims {1,2,3,2}
    const size_t C = 2, H = 2, W = 3;
    std::vector<float> out;
    for (size_t h = 0; h < H; ++h)
        for (size_t w = 0; w < W; ++w)
            for (size_t c = 0; c < C; ++c)
                out.push_back(x4_data[(c * H + h) * W + w]);
    return out;
}

std::vector<float> ref_concat() {
    std::vector<float> out;
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 3; ++c)
            out.push_back(ca_data[r * 3 + c]);
        for (size_t c = 0; c < 2; ++c)
            out.push_back(cb_data[r * 2 + c]);
    }
    return out;
}

std::vector<float> ref_softmax() {
    std::vector<float> out;
    for (size_t l = 0; l < 2; ++l) {
        float m = xs_data[l * 4];
        for (size_t i = 1; i < 4; ++i)
            m = std::max(m, xs_data[l * 4 + i]);
        float s = 0;
        std::vector<float> e(4);
        for (size_t i = 0; i < 4; ++i) {
            e[i] = std::exp(xs_data[l * 4 + i] - m);
            s += e[i];
        }
        for (size_t i = 0; i < 4; ++i)
            out.push_back(e[i] / s);
    }
    return out;
}

std::vector<float> ref_reduce(ir_op op) {
    std::vector<float> out;
    for (size_t r = 0; r < 2; ++r) {
        float acc = op == ir_op::reduce_max ? xs_data[r * 4] : 0.0f;
        for (size_t c = 0; c < 4; ++c) {
            const float v = xs_data[r * 4 + c];
            if (op == ir_op::reduce_sum)
                acc += v;
            else if (op == ir_op::reduce_mean)
                acc += v;
            else
                acc = std::max(acc, v);
        }
        if (op == ir_op::reduce_mean)
            acc /= 4.0f;
        out.push_back(acc);
    }
    return out;
}

const std::vector<float> x_data{-2.0f, -0.5f, 0.25f, 1.5f, 3.0f, -7.0f};
const std::vector<float> a_data{2.0f, 4.0f, 6.0f, 8.0f};
const std::vector<float> b_data{0.5f, 1.5f, 2.5f, 3.5f};
const std::vector<float> xq_data{1.0f, 2.0f, -1.0f, 0.5f, 0.0f, -2.0f, 3.0f, 1.5f, 2.5f, 1.0f, -0.5f, -3.0f};

std::vector<float> ref_unary() {
    std::vector<float> out;
    out.reserve(x_data.size());
    for (float xi : x_data) {
        const float lr = xi > 0.0f ? xi : 0.1f * xi;   // leaky_relu, alpha=0.1
        const float sg = 1.0f / (1.0f + std::exp(-lr));  // sigmoid
        out.push_back(std::tanh(sg));                    // tanh
    }
    return out;
}

std::vector<float> ref_binary() {
    std::vector<float> out;
    out.reserve(a_data.size());
    for (size_t i = 0; i < a_data.size(); ++i) {
        const float m = a_data[i] * b_data[i];
        const float s = m - b_data[i];
        out.push_back(s / a_data[i]);
    }
    return out;
}

// y = relu(x) @ w + bias, x [2,3] row-major, w [3,2], bias broadcast [1,2].
std::vector<float> ref_legacy() {
    const float w[3][2] = {{1.0f, -1.0f}, {0.5f, 2.0f}, {0.0f, 1.5f}};
    std::vector<float> out;
    for (size_t row = 0; row < 2; ++row) {
        for (size_t col = 0; col < 2; ++col) {
            float acc = col == 0 ? 10.0f : -10.0f;
            for (size_t k = 0; k < 3; ++k) {
                const float xi = std::max(0.0f, x_data[row * 3 + k]);
                acc += xi * w[k][col];
            }
            out.push_back(acc);
        }
    }
    return out;
}

// Independent Q4_0 dequant + [3,4]@[4,32] reference.
std::vector<float> ref_quant_matmul() {
    const float ds[4] = {1.0f, 0.5f, 2.0f, 0.25f};
    std::vector<float> out(3 * 32, 0.0f);
    for (size_t m = 0; m < 3; ++m) {
        for (size_t n = 0; n < 32; ++n) {
            for (size_t k = 0; k < 4; ++k) {
                const float wk = ds[k] * static_cast<float>(static_cast<int>(n % 16) - 8);
                out[m * 32 + n] += xq_data[m * 4 + k] * wk;
            }
        }
    }
    return out;
}

std::vector<float> run_cpu(const ir_graph& g, const std::map<std::string, std::vector<float>>& ins,
                           size_t expect_elems) {
    auto outs = cpu_execute(g, ins);
    std::vector<float> res;
    for (const auto& [id, vals] : outs) {
        (void)id;
        res.insert(res.end(), vals.begin(), vals.begin() + std::min(vals.size(), expect_elems));
        break;  // single-output graphs
    }
    return res;
}

std::vector<float> run_gpu(vk_engine& engine, const ExecutionConfig& cfg, const ir_graph& g,
                           const std::map<std::string, std::vector<float>>& ins, size_t expect_elems) {
    vk_program_builder builder(engine, cfg);
    auto prog = builder.build(g);
    vk_network net(engine, cfg, prog);
    for (const auto& [id, vals] : ins) {
        const layout lay(g.tensor_shapes.at(id), 4);
        auto mem = engine.allocate_memory(lay, allocation_type::usm_host, true);
        void* p = mem->lock();
        std::memcpy(p, vals.data(), vals.size() * sizeof(float));
        mem->unlock();
        net.set_input_data(id, mem);
    }
    // Output buffers are device-local; read back through a host-visible
    // override keyed by the program's output id.
    const std::string out_id = prog->output_port_to_id.begin()->second;
    const layout out_lay(g.tensor_shapes.at(g.outputs[0]), 4);
    auto out_mem = engine.allocate_memory(out_lay, allocation_type::usm_host, true);
    net.set_output_memory(out_id, out_mem, false);

    net.execute({});
    const float* p = static_cast<const float*>(out_mem->lock());
    std::vector<float> res(p, p + expect_elems);
    out_mem->unlock();
    return res;
}

}  // namespace

int main() {
    try {
        auto g1 = make_unary_graph();
        auto g2 = make_binary_graph();
        const auto r1 = ref_unary();
        const auto r2 = ref_binary();

        // CPU executors
        check("unary chain CPU vs ref", run_cpu(g1, {{"x", x_data}}, r1.size()), r1, 1e-5f);
        check("binary chain CPU vs ref", run_cpu(g2, {{"a", a_data}, {"b", b_data}}, r2.size()), r2, 1e-5f);

        // Legacy ops regression guard (relu/add/matmul path)
        auto g3 = make_legacy_graph();
        const auto r3 = ref_legacy();
        const std::map<std::string, std::vector<float>> legacy_in{{"x", x_data}};
        check("legacy relu+matmul+add CPU", run_cpu(g3, legacy_in, r3.size()), r3, 1e-5f);

        // Constant-broadcast inputs must be expanded by the core itself.
        auto g4 = make_legacy_bcast_graph();
        check("legacy bcast bias CPU", run_cpu(g4, legacy_in, r3.size()), r3, 1e-5f);

        // GPU executor
        ExecutionConfig cfg;
        auto engine = create_vk_engine();
        check("unary chain GPU vs ref",
              run_gpu(*engine, cfg, g1, {{"x", x_data}}, r1.size()), r1, 2e-5f);
        check("binary chain GPU vs ref",
              run_gpu(*engine, cfg, g2, {{"a", a_data}, {"b", b_data}}, r2.size()), r2, 2e-5f);
        check("legacy relu+matmul+add GPU", run_gpu(*engine, cfg, g3, legacy_in, r3.size()), r3, 2e-5f);
        check("legacy bcast bias GPU", run_gpu(*engine, cfg, g4, legacy_in, r3.size()), r3, 2e-5f);

        // FB/PB round-trip: the alpha attribute must survive serialization.
        const std::vector<ir_graph> graphs{g1, g2};
        auto pb = serialize_pb(graphs);
        auto back = deserialize_pb(pb);
        check("round-trip graph count", {static_cast<float>(back.size())}, {2.0f}, 0.0f);
        check("round-trip unary CPU vs ref", run_cpu(back[0], {{"x", x_data}}, r1.size()), r1, 1e-5f);
        check("round-trip binary CPU vs ref", run_cpu(back[1], {{"a", a_data}, {"b", b_data}}, r2.size()), r2, 1e-5f);
        check("round-trip unary GPU vs ref",
              run_gpu(*engine, cfg, back[0], {{"x", x_data}}, r1.size()), r1, 2e-5f);

        // Quantized weights must survive FB/PB round-trip (FB v3).
        auto gq = make_quant_graph();
        const auto rq = ref_quant_matmul();
        const std::map<std::string, std::vector<float>> qin{{"x", xq_data}};
        const std::vector<ir_graph> qgraphs{gq};
        auto qpb = serialize_pb(qgraphs);
        auto qback = deserialize_pb(qpb);
        check_true("round-trip keeps quant const",
                   qback[0].quant_constants.count("w") == 1 &&
                       qback[0].quant_constants.at("w").type == 2 &&
                       qback[0].quant_constants.at("w").bytes == gq.quant_constants.at("w").bytes);
        check("quant matmul CPU vs ref", run_cpu(gq, qin, rq.size()), rq, 1e-3f);
        check("quant round-trip CPU vs ref", run_cpu(qback[0], qin, rq.size()), rq, 1e-3f);
        check("quant matmul GPU vs ref", run_gpu(*engine, cfg, gq, qin, rq.size()), rq, 1e-3f);
        check("quant round-trip GPU vs ref", run_gpu(*engine, cfg, qback[0], qin, rq.size()), rq, 1e-3f);

        // Unified dispatcher: one entry point routes by device name.
        check("dispatch unary CPU vs ref", vk_execute(g1, {{"x", x_data}}, "CPU")["th"], r1, 1e-5f);
        check("dispatch unary GPU vs ref", vk_execute(g1, {{"x", x_data}}, "GPU")["th"], r1, 2e-5f);
        check("dispatch legacy GPU.0 vs ref", vk_execute(g3, legacy_in, "GPU.0")["y"], r3, 2e-5f);
        try {
            (void)vk_execute(g1, {{"x", x_data}}, "NPU");
            check_true("dispatch NPU clean error", false);
        } catch (const std::exception& e) {
            check_true("dispatch NPU clean error",
                       std::string(e.what()).find("No Vulkan device matching") != std::string::npos);
        }
        const auto devs = vk_available_devices();
        std::printf("%-34s", "available devices:");
        for (const auto& d : devs)
            std::printf(" %s", d.c_str());
        std::printf("\n");

        // ---- batch 2: transpose / reshape / concat / softmax / reduce ----
        auto gt = make_transpose_graph();
        const auto rt = ref_transpose_nhwc();
        const std::map<std::string, std::vector<float>> tin{{"x", x4_data}};
        check("transpose NHWC CPU vs ref", run_cpu(gt, tin, rt.size()), rt, 1e-5f);
        check("transpose NHWC GPU vs ref", run_gpu(*engine, cfg, gt, tin, rt.size()), rt, 2e-5f);

        auto gr = make_reshape_graph();
        const auto rr = ref_legacy();  // reuse relu(x) values (same x layout [2,3])
        std::vector<float> rr6;
        for (size_t row = 0; row < 2; ++row)
            for (size_t c = 0; c < 3; ++c)
                rr6.push_back(std::max(0.0f, x_data[row * 3 + c]));
        const std::map<std::string, std::vector<float>> rin{{"x", x_data}};
        check("reshape alias CPU vs ref", run_cpu(gr, rin, rr6.size()), rr6, 0.0f);
        check("reshape alias GPU vs ref", run_gpu(*engine, cfg, gr, rin, rr6.size()), rr6, 0.0f);

        auto gc = make_concat_graph();
        const auto rc = ref_concat();
        const std::map<std::string, std::vector<float>> cin{{"a", ca_data}, {"b", cb_data}};
        check("concat axis1 CPU vs ref", run_cpu(gc, cin, rc.size()), rc, 1e-5f);
        check("concat axis1 GPU vs ref", run_gpu(*engine, cfg, gc, cin, rc.size()), rc, 1e-5f);
        check("concat axis1 dispatch vs ref", vk_execute(gc, cin, "GPU")["c"], rc, 1e-5f);

        auto gs = make_softmax_graph();
        const auto rsm = ref_softmax();
        const std::map<std::string, std::vector<float>> sin{{"x", xs_data}};
        check("softmax CPU vs ref", run_cpu(gs, sin, rsm.size()), rsm, 1e-5f);
        check("softmax GPU vs ref", run_gpu(*engine, cfg, gs, sin, rsm.size()), rsm, 2e-5f);

        for (const ir_op op : {ir_op::reduce_sum, ir_op::reduce_mean, ir_op::reduce_max}) {
            auto gred = make_reduce_graph(op);
            const auto rrd = ref_reduce(op);
            const char* nm = op == ir_op::reduce_sum ? "sum" : (op == ir_op::reduce_mean ? "mean" : "max");
            check((std::string("reduce ") + nm + " CPU vs ref").c_str(), run_cpu(gred, sin, rrd.size()), rrd, 1e-5f);
            check((std::string("reduce ") + nm + " GPU vs ref").c_str(),
                  run_gpu(*engine, cfg, gred, sin, rrd.size()), rrd, 2e-5f);
        }

        // New node attributes must survive serialization.
        const std::vector<ir_graph> b2graphs{gt, gc, gs};
        auto b2pb = serialize_pb(b2graphs);
        auto b2back = deserialize_pb(b2pb);
        check_true("round-trip keeps perm/axis",
                   b2back[0].nodes[1].transpose_order == std::vector<size_t>{0, 2, 3, 1} &&
                       b2back[1].nodes[2].axis == 1 && b2back[2].nodes[1].axis == 1);
        check("round-trip concat CPU vs ref", run_cpu(b2back[1], cin, rc.size()), rc, 1e-5f);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}
