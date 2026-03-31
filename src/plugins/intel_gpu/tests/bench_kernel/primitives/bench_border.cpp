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
#include <intel_gpu/primitives/border.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_border : public kernel_base {
public:
    std::string name() const override { return "border"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("border requires at least 1 shape. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        // Parse pad begin/end from config
        ov::CoordinateDiff pad_begin, pad_end;
        if (!config.pads_begin.empty()) {
            auto v = parse_x_svec(config.pads_begin);
            pad_begin.assign(v.begin(), v.end());
        }
        if (!config.pads_end.empty()) {
            auto v = parse_x_svec(config.pads_end);
            pad_end.assign(v.begin(), v.end());
        }
        if (pad_begin.empty()) pad_begin.resize(shapes[0].size(), 1);
        if (pad_end.empty()) pad_end.resize(shapes[0].size(), 1);

        ov::op::PadMode pad_mode = ov::op::PadMode::CONSTANT;
        if (config.border_mode == 1) pad_mode = ov::op::PadMode::EDGE;
        else if (config.border_mode == 2) pad_mode = ov::op::PadMode::REFLECT;
        else if (config.border_mode == 3) pad_mode = ov::op::PadMode::SYMMETRIC;
        float pad_value = config.border_value;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::border("prim", {cldnn::input_info("input")}, 0,
                     pad_begin, pad_end, pad_mode, pad_value));

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
            std::vector<int64_t> pb(pad_begin.begin(), pad_begin.end());
            std::vector<int64_t> pe(pad_end.begin(), pad_end.end());
            auto ref_out = ref::border_pad(input_f32, shapes[0], pb, pe, config.border_mode, pad_value);
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

REGISTER_KERNEL(bench_border)

}  // namespace bench_kernel
