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
#include <intel_gpu/primitives/fake_convert.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_fake_convert : public kernel_base {
public:
    std::string name() const override { return "fake_convert"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: input:scale
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("fake_convert requires 2 shapes (input:scale). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(lay0);
        auto scale_mem = engine.allocate_memory(lay1);
        fill_memory_random(input_mem, *stream, dt);
        fill_memory_random(scale_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", lay0));
        topology.add(cldnn::input_layout("scale", lay1));
        topology.add(cldnn::fake_convert("prim",
            cldnn::input_info("input"), cldnn::input_info("scale"),
            ov::element::f8e4m3));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);
        network.set_input_data("scale", scale_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto scale_f32 = read_memory_to_f32(scale_mem, *stream);
            auto ref_out = ref::fake_convert(input_f32, scale_f32, shapes[0], shapes[1]);
            float atol = 1.0f, rtol = 0.5f;  // f8e4m3 has very low precision
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

REGISTER_KERNEL(bench_fake_convert)

}  // namespace bench_kernel
