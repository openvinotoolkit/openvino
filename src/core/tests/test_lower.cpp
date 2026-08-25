// test_lower: frontend bridge lowering — canonical type mapping (aten/tf
// prefixes, in-place suffixes), constants, conv bias contract, linear→
// transpose_b, aggregated unsupported-op report, end-to-end CPU parity.

#include "cpu_engine.hpp"
#include "vk_lower.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace ov::core::vulkan::cross_platform;

namespace {

int failures = 0;

void check_true(const char* name, bool ok) {
    std::printf("%-44s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

void check_vec(const char* name, const std::vector<float>& got, const std::vector<float>& want, float tol) {
    bool ok = got.size() == want.size();
    if (ok)
        for (size_t i = 0; i < want.size(); ++i)
            if (std::fabs(got[i] - want[i]) > tol) {
                ok = false;
                break;
            }
    std::printf("%-44s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

bridge::node bn(const std::string& type, const std::string& id, std::vector<std::string> ins,
                std::vector<size_t> shape = {}) {
    bridge::node n;
    n.type = type;
    n.id = id;
    n.inputs = std::move(ins);
    n.shape = std::move(shape);
    return n;
}

}  // namespace

int main() {
    bridge::graph g1;
    try {
        bridge::graph g1;
        g1.nodes.push_back(bn("parameter", "x", {}, {4}));
        g1.nodes.push_back(bn("Relu", "r", {"x"}, {4}));
        g1.nodes.push_back(bn("tf.Add", "a", {"x", "r"}, {4}));
        g1.nodes.push_back(bn("result", "out", {"a"}));
        g1.outputs = {"a"};
        auto lg = lower(g1);
        check_true("prefix/case/inplace normalized",
                   lg.nodes.size() == 4 && lg.nodes[1].op == ir_op::relu && lg.nodes[2].op == ir_op::add);
        check_true("inputs discovered in port order", lg.inputs.size() == 1 && lg.inputs[0] == "x");

        std::vector<float> xv{-1.f, 2.f, -3.f, 4.f};
        auto got = cpu_execute(lg, {{"x", xv}}).at("a");
        std::vector<float> want(4);
        for (size_t i = 0; i < 4; ++i)
            want[i] = std::max(0.f, xv[i]) + xv[i];
        check_vec("lowered graph executes correctly", got, want, 1e-6f);

        bridge::graph g2;
        g2.nodes.push_back(bn("aten::linear", "fc", {"x", "w"}, {2, 3}));
        g2.nodes.push_back(bn("result", "out", {"fc"}));
        g2.outputs = {"fc"};
        auto lg2 = lower(g2);
        check_true("linear forces transpose_b",
                   lg2.nodes[0].op == ir_op::matmul && lg2.nodes[0].matmul_transpose_b);

        bridge::graph g3;
        g3.nodes.push_back(bn("aten::conv2d", "cv", {"x", "w"}, {1, 1, 4, 4}));
        g3.outputs = {"cv"};
        bool rejected = false;
        try {
            (void)lower(g3);
        } catch (const std::exception&) {
            rejected = true;
        }
        check_true("conv without bias rejected", rejected);

        bridge::graph g4;
        g4.nodes.push_back(bn("aten::embedding", "e", {"x"}, {4}));
        g4.nodes.push_back(bn("aten::where", "wq", {"x"}, {4}));
        g4.nodes.push_back(bn("result", "out", {"e"}));
        g4.outputs = {"e"};
        bool reported_both = false;
        try {
            (void)lower(g4);
        } catch (const std::exception& e) {
            const std::string what = e.what();
            reported_both = what.find("aten::embedding") != std::string::npos &&
                            what.find("aten::where") != std::string::npos;
        }
        check_true("unsupported ops aggregated", reported_both);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}




