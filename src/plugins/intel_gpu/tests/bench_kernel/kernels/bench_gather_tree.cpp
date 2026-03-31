// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_tree.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_gather_tree : public kernel_base {
public:
    std::string name() const override { return "gather_tree"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: step_input:parent_input:max_seq_len:end_token
        // step_input & parent_input: [T, B, beam_width]
        // max_seq_len: [B]
        // end_token: scalar [1]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 4) throw std::runtime_error("gather_tree requires 4 shapes (step:parent:max_seq_len:end_token). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        auto make_layout = [&](const std::vector<int64_t>& s, cldnn::data_types t) {
            ov::PartialShape ps(std::vector<ov::Dimension>(s.begin(), s.end()));
            return cldnn::layout(ps, t, cldnn::format::bfyx);
        };

        auto lay0 = make_layout(shapes[0], dt);
        auto lay1 = make_layout(shapes[1], dt);
        auto lay2 = make_layout(shapes[2], dt);
        // end_token must be scalar (rank 0)
        cldnn::layout lay3(ov::PartialShape{}, dt, cldnn::format::bfyx);
        auto mem0 = engine.allocate_memory(lay0);
        auto mem1 = engine.allocate_memory(lay1);
        auto mem2 = engine.allocate_memory(lay2);
        auto mem3 = engine.allocate_memory(lay3);
        fill_memory_random(mem0, *stream, dt);
        fill_memory_random(mem1, *stream, dt);
        fill_memory_random(mem2, *stream, dt);
        fill_memory_random(mem3, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("step", lay0));
        topology.add(cldnn::input_layout("parent", lay1));
        topology.add(cldnn::input_layout("max_seq", lay2));
        topology.add(cldnn::input_layout("end_tok", lay3));
        topology.add(cldnn::gather_tree("prim",
            cldnn::input_info("step"), cldnn::input_info("parent"),
            cldnn::input_info("max_seq"), cldnn::input_info("end_tok")));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("step", mem0);
        network.set_input_data("parent", mem1);
        network.set_input_data("max_seq", mem2);
        network.set_input_data("end_tok", mem3);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        [[maybe_unused]] acc_result acc_res;
        if (config.is_acc()) {
            throw bench_unimplemented("CPU reference not implemented");
        }
        if (config.is_perf()) {
            run_perf(network, config, timer);
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_gather_tree)

}  // namespace bench_kernel
