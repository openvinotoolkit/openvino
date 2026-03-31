// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/rms.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// RMS (Root Mean Square Normalization) kernel benchmark
//
// Usage:
//   --rms --dt=f16 shape
//
// RMS norm: output = input / sqrt(mean(input^2) + eps) * gamma
// gamma shape is the last dimension of input
//
// Example:
//   --rms --dt=f16 1x128x4096
//   --rms --dt=f16 1x1x4096
// ============================================================================

class bench_rms : public kernel_base {
public:
    std::string name() const override { return "rms"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("RMS requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        // Read epsilon from config (shared with MVN) or default to 1e-6
        float epsilon = 1e-6f;
        if (!config.epsilon.empty()) {
            try { epsilon = std::stof(config.epsilon); } catch (...) {}
        }

        auto exec_config = make_exec_config(config, "rms_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout input_layout_desc(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout_desc);
        fill_memory_random(input_mem, *stream, dt);

        // Gamma shape: last dimension of input (e.g., hidden_dim)
        int64_t hidden_dim = shape.back();
        ov::PartialShape gamma_ps({hidden_dim});
        cldnn::layout gamma_layout(gamma_ps, dt, cldnn::format::bfyx);
        auto gamma_mem = engine.allocate_memory(gamma_layout);
        fill_memory_random(gamma_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout_desc));
        topology.add(cldnn::data("gamma", gamma_mem));
        topology.add(cldnn::rms("rms_prim",
            cldnn::input_info("input"),
            cldnn::input_info("gamma"),
            epsilon));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        // Execute and collect results
        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        // Accuracy mode
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "rms_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto gamma_f32 = read_memory_to_f32(gamma_mem, *stream);
            auto ref_out = ref::rms_norm(input_f32, gamma_f32, shape, epsilon);

            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
            has_acc = true;
            reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
            if (!acc_res.pass) test_passed = false;
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            has_perf = true;
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "rms_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_rms)

}  // namespace bench_kernel
