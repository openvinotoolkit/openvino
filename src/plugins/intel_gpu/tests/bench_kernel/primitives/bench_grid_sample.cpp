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
#include <intel_gpu/primitives/grid_sample.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_grid_sample : public kernel_base {
public:
    std::string name() const override { return "grid_sample"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: data[N,C,H_in,W_in]:grid[N,H_out,W_out,2]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("grid_sample requires 2 shapes (data:grid). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        ov::op::v9::GridSample::Attributes attrs;
        attrs.align_corners = config.grid_align_corners != 0;
        if (config.grid_mode == 0) attrs.mode = ov::op::v9::GridSample::InterpolationMode::BILINEAR;
        else if (config.grid_mode == 1) attrs.mode = ov::op::v9::GridSample::InterpolationMode::BICUBIC;
        else attrs.mode = ov::op::v9::GridSample::InterpolationMode::NEAREST;
        if (config.grid_padding == 0) attrs.padding_mode = ov::op::v9::GridSample::PaddingMode::ZEROS;
        else if (config.grid_padding == 1) attrs.padding_mode = ov::op::v9::GridSample::PaddingMode::BORDER;
        else attrs.padding_mode = ov::op::v9::GridSample::PaddingMode::REFLECTION;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(lay0);
        auto grid_mem = engine.allocate_memory(lay1);
        fill_memory_random(data_mem, *stream, dt);
        fill_memory_random(grid_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", lay0));
        topology.add(cldnn::input_layout("grid", lay1));
        topology.add(cldnn::grid_sample("prim",
            {cldnn::input_info("data"), cldnn::input_info("grid")}, attrs));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);
        network.set_input_data("grid", grid_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto data_f32 = read_memory_to_f32(data_mem, *stream);
            auto grid_f32 = read_memory_to_f32(grid_mem, *stream);
            auto ref_out = ref::grid_sample(data_f32, grid_f32, shapes[0], shapes[1], config.grid_mode, config.grid_padding, config.grid_align_corners != 0);
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

REGISTER_KERNEL(bench_grid_sample)

}  // namespace bench_kernel
