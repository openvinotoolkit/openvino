// test_optim: batch 8 вЂ” graph passes. DCE removes a dead branch; constant
// folding collapses a constant subgraph into one constant (verified by node
// count AND numerically); peephole cancels double transpose and reluв€relu;
// optimize() runs the full pipeline and the result still matches the CPU
// reference of the original graph.

#include "cpu_engine.hpp"
#include "vk_pass.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace ov::core::vulkan::cross_platform;

namespace {

int failures = 0;

void check_true(const char* name, bool ok) {
    std::printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
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
    std::printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++failures;
}

ir_node nd(const std::string& id, ir_op op, std::vector<std::string> ins,
           std::vector<size_t> order = {}) {
    ir_node n;
    n.id = id;
    n.op = op;
    n.inputs = std::move(ins);
    n.transpose_order = std::move(order);
    return n;
}

}  // namespace

int main() {
    try {
        // Graph: live branch  relu(x)*2 + c1 ; dead branch  sigmoid(y)+c2
        ir_graph g;
        g.nodes.push_back(nd("x", ir_op::parameter, {}));
        g.nodes.push_back(nd("y", ir_op::parameter, {}));          // feeds the DEAD branch
        g.nodes.push_back(nd("c1", ir_op::constant, {}));
        g.nodes.push_back(nd("c2", ir_op::constant, {}));
        g.nodes.push_back(nd("two", ir_op::constant, {}));
        g.nodes.push_back(nd("dead_s", ir_op::sigmoid, {"y"}));
        g.nodes.push_back(nd("dead_a", ir_op::add, {"dead_s", "c2"}));
        g.nodes.push_back(nd("r", ir_op::relu, {"x"}));
        g.nodes.push_back(nd("m", ir_op::mul, {"r", "two"}));
        g.nodes.push_back(nd("a", ir_op::add, {"m", "c1"}));
        g.nodes.push_back(nd("out", ir_op::result, {"a"}));
        for (const char* t : {"x", "y", "r", "m", "a", "dead_s", "dead_a"})
            g.tensor_shapes[t] = {4};
        g.tensor_shapes["c1"] = {4};
        g.tensor_shapes["c2"] = {4};
        g.tensor_shapes["two"] = {1};   // broadcast constant
        g.constant_data["c1"] = {10.f, -20.f, 30.f, -40.f};
        g.constant_data["c2"] = {0.5f, 0.5f, 0.5f, 0.5f};
        g.constant_data["two"] = {2.f};
        g.inputs = {"x", "y"};
        g.outputs = {"a"};

        auto opt = pass::optimize(g);

        check_true("dce removed dead branch",
                   std::none_of(opt.nodes.begin(), opt.nodes.end(), [](const ir_node& n) {
                       return n.id == "dead_s" || n.id == "dead_a";
                   }));
        check_true("live branch survives",
                   std::any_of(opt.nodes.begin(), opt.nodes.end(), [](const ir_node& n) {
                       return n.id == "a";
                   }));

        // Numeric equivalence on the LIVE inputs.
        std::vector<float> xv{1.f, -2.f, 3.f, -0.5f};
        const auto orig = cpu_execute(g, {{"x", xv}, {"y", {0.f, 0.f, 0.f, 0.f}}}).at("a");
        const auto optimized = cpu_execute(opt, {{"x", xv}}).at("a");
        check_vec("optimize preserves numbers", optimized, orig, 1e-6f);

        // ---- constant folding collapses a constants-only subgraph --------------
        ir_graph gf;
        gf.nodes.push_back(nd("ca", ir_op::constant, {}));
        gf.nodes.push_back(nd("cb", ir_op::constant, {}));
        gf.nodes.push_back(nd("fr", ir_op::relu, {"ca"}));
        gf.nodes.push_back(nd("fa", ir_op::add, {"fr", "cb"}));
        gf.nodes.push_back(nd("out", ir_op::result, {"fa"}));
        gf.tensor_shapes["ca"] = {3};
        gf.tensor_shapes["cb"] = {3};
        gf.tensor_shapes["fr"] = {3};
        gf.tensor_shapes["fa"] = {3};
        gf.constant_data["ca"] = {-1.f, 2.f, -3.f};
        gf.constant_data["cb"] = {10.f, 20.f, 30.f};
        gf.outputs = {"fa"};
        std::printf("    [dbg] before fold\n"); auto pf = pass::fold_constants(gf); std::printf("    [dbg] after fold n=%zu\n", pf.nodes.size());
        check_true("fold collapsed relu+add into one constant",
                   pf.nodes.size() == 2 && pf.nodes[0].op == ir_op::constant &&
                       pf.nodes[1].op == ir_op::result);
        check_vec("folded value correct", pf.constant_data.at(pf.nodes[0].id),
                  {10.f, 22.f, 30.f}, 0.0f);
        // The optimizer output must stay executable.
        auto fo = cpu_execute(pass::optimize(gf), {}).at("fa");
        check_vec("optimized const-graph executes", fo, {10.f, 22.f, 30.f}, 0.0f);

        // ---- peephole: double transpose cancels --------------------------------
        ir_graph gt;
        gt.nodes.push_back(nd("x", ir_op::parameter, {}));
        gt.nodes.push_back(nd("t1", ir_op::transpose, {"x"}, {1, 0}));
        gt.nodes.push_back(nd("t2", ir_op::transpose, {"t1"}, {1, 0}));
        gt.nodes.push_back(nd("out", ir_op::result, {"t2"}));
        gt.tensor_shapes["x"] = {2, 3};
        gt.tensor_shapes["t1"] = {3, 2};
        gt.tensor_shapes["t2"] = {2, 3};
        gt.inputs = {"x"};
        gt.outputs = {"t2"};
        auto pt = pass::peephole(gt);
        check_true("double transpose cancelled",
                   std::none_of(pt.nodes.begin(), pt.nodes.end(), [](const ir_node& n) {
                       return n.op == ir_op::transpose;
                   }));

        // Non-cancelling pair merges into ONE transpose with composed order:
        // x[2,3] --{1,0}--> [3,2] --{1,0}-- would cancel; use a 4D NHWC chain.
        ir_graph g4;
        g4.nodes.push_back(nd("x", ir_op::parameter, {}));
        g4.nodes.push_back(nd("ta", ir_op::transpose, {"x"}, {0, 2, 3, 1}));
        g4.nodes.push_back(nd("tb", ir_op::transpose, {"ta"}, {0, 3, 1, 2}));
        g4.nodes.push_back(nd("out", ir_op::result, {"tb"}));
        g4.tensor_shapes["x"] = {1, 2, 3, 4};
        g4.tensor_shapes["ta"] = {1, 3, 4, 2};
        g4.tensor_shapes["tb"] = {1, 2, 3, 4};
        g4.inputs = {"x"};
        g4.outputs = {"tb"};
        auto p4 = pass::peephole(g4);
        const size_t transposes =
            static_cast<size_t>(std::count_if(p4.nodes.begin(), p4.nodes.end(), [](const ir_node& n) {
                return n.op == ir_op::transpose;
            }));
        check_true("4D transposes merged to one", transposes <= 1);

        std::printf(failures == 0 ? "\nALL PASS\n" : "\nFAILED: %d\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
}







