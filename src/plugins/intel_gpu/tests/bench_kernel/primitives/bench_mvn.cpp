// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mvn.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// MVN (Mean Variance Normalization) kernel benchmark
//
// Usage:
//   --mvn --dt=f16 shape
//
// Example:
//   --mvn --dt=f16 1x768x196
// ============================================================================

class bench_mvn : public kernel_base {
public:
    std::string name() const override { return "mvn"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("MVN requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "mvn_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout layout(ps, dt, get_input_format(config, 0, shape.size()));
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, dt);

        // Parse reduction axes from config or use default [1, ..., rank-1]
        std::vector<int64_t> reduction_axes;
        if (!config.mvn_reduction_axes.empty()) {
            // Parse colon-separated axes
            std::istringstream axes_stream(config.mvn_reduction_axes);
            std::string axis_str;
            while (std::getline(axes_stream, axis_str, ':')) {
                if (!axis_str.empty()) {
                    reduction_axes.push_back(static_cast<int64_t>(std::stoll(axis_str)));
                }
            }
        } else {
            // Default: normalize across all dims except batch (axes [1, ..., rank-1])
            for (size_t i = 1; i < shape.size(); ++i) {
                reduction_axes.push_back(static_cast<int64_t>(i));
            }
        }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        bool norm_var = (config.normalize_variance != 0);
        float eps = config.epsilon.empty() ? 1e-6f : std::stof(config.epsilon);
        bool eps_sqrt = (config.eps_inside_sqrt != 0);
        topology.add(cldnn::mvn("mvn_prim",
            cldnn::input_info("input"),
            norm_var,
            eps,
            eps_sqrt,
            reduction_axes));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", mem);

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
            auto gpu_out = read_network_output_f32(outputs, "mvn_prim", *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            auto ref_out = ref::mvn(input_f32, shape, norm_var, eps, eps_sqrt);

            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            // MVN involves reduction (mean/var) which accumulates errors
            atol = std::max(atol, 5e-4f);
            rtol = std::max(rtol, 1e-3f);
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
        print_result(network, "mvn_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_mvn)

}  // namespace bench_kernel
