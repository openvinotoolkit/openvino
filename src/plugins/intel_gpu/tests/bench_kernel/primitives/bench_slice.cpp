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
#include <intel_gpu/primitives/slice.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_slice : public kernel_base {
public:
    std::string name() const override { return "slice"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("slice requires at least 1 shape. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        // Use ss_begin/ss_end/ss_strides for slice parameters
        auto& in_shape = shapes[0];
        size_t rank = in_shape.size();

        std::vector<int64_t> start_vec, stop_vec, step_vec, axes_vec;
        if (!config.ss_begin.empty()) start_vec = parse_colon_vec(config.ss_begin);
        if (!config.ss_end.empty()) stop_vec = parse_colon_vec(config.ss_end);
        if (!config.ss_strides.empty()) step_vec = parse_colon_vec(config.ss_strides);

        if (start_vec.empty()) start_vec.resize(rank, 0);
        if (stop_vec.empty()) { stop_vec = std::vector<int64_t>(in_shape.begin(), in_shape.end()); }
        if (step_vec.empty()) step_vec.resize(rank, 1);
        for (size_t i = 0; i < rank; ++i) axes_vec.push_back(static_cast<int64_t>(i));

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        // Create const data for start/stop/step/axes
        auto make_i64_data = [&](const std::string& name, const std::vector<int64_t>& vals) {
            ov::PartialShape dps({static_cast<int64_t>(vals.size())});
            cldnn::layout dlay(dps, cldnn::data_types::i64, cldnn::format::bfyx);
            auto dmem = engine.allocate_memory(dlay);
            cldnn::mem_lock<int64_t> lock(dmem, *stream);
            for (size_t i = 0; i < vals.size(); ++i) lock[i] = vals[i];
            return dmem;
        };

        auto start_mem = make_i64_data("start", start_vec);
        auto stop_mem = make_i64_data("stop", stop_vec);
        auto step_mem = make_i64_data("step", step_vec);
        auto axes_mem = make_i64_data("axes", axes_vec);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::data("start", start_mem));
        topology.add(cldnn::data("stop", stop_mem));
        topology.add(cldnn::data("step", step_mem));
        topology.add(cldnn::data("axes", axes_mem));
        topology.add(cldnn::slice("prim", {
            cldnn::input_info("input"), cldnn::input_info("start"),
            cldnn::input_info("stop"), cldnn::input_info("step"),
            cldnn::input_info("axes")}));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto ref_out = ref::slice_ref(input_f32, in_shape, start_vec, stop_vec, step_vec, axes_vec);
            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
            reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
            if (!acc_res.pass) test_passed = false;
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

REGISTER_KERNEL(bench_slice)

}  // namespace bench_kernel
