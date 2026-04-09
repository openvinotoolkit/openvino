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
#include <intel_gpu/primitives/reduce.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Reduce kernel benchmark
//
// Usage:
//   --reduce --dt=f16 --attr-post-ops=MODE:AXES shape
//   MODE: max, min, mean, sum, prod
//   AXES: comma-separated (e.g., 2x3 for axes [2,3])
//
// Example:
//   --reduce --dt=f16 --attr-post-ops=mean 1x64x56x56
//   (reduces along last axis by default)
// ============================================================================

inline cldnn::reduce_mode parse_reduce_mode(const std::string& s) {
    if (s == "max")  return cldnn::reduce_mode::max;
    if (s == "min")  return cldnn::reduce_mode::min;
    if (s == "mean") return cldnn::reduce_mode::mean;
    if (s == "sum")  return cldnn::reduce_mode::sum;
    if (s == "prod") return cldnn::reduce_mode::prod;
    if (s == "logical_and") return cldnn::reduce_mode::logical_and;
    if (s == "logical_or")  return cldnn::reduce_mode::logical_or;
    return cldnn::reduce_mode::mean;  // default
}

class bench_reduce : public kernel_base {
public:
    std::string name() const override { return "reduce"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Reduce requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        // Parse mode: from config (verbose), post-ops string, or default
        cldnn::reduce_mode mode = static_cast<cldnn::reduce_mode>(config.reduce_mode);
        std::vector<int64_t> axes;

        // config.reduce_axes from verbose log (e.g. "1:2:3")
        if (!config.reduce_axes.empty()) {
            axes = parse_colon_vec(config.reduce_axes);
        }

        // Override from post-ops string for backward compatibility
        if (!config.attr_post_ops_str.empty()) {
            auto parts = split(config.attr_post_ops_str, ':');
            if (!parts.empty()) {
                mode = parse_reduce_mode(parts[0]);
            }
            if (parts.size() >= 2) {
                auto axis_dims = parse_dims(parts[1]);
                axes = axis_dims;
            }
        }

        // Default: reduce along last axis
        auto& shape = shapes[0];
        if (axes.empty()) {
            axes.push_back(static_cast<int64_t>(shape.size()) - 1);
        }

        auto exec_config = make_exec_config(config, "reduce_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout layout(ps, dt, get_input_format(config, 0, shape.size()));
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        bool keep_dims = (config.keep_dims != 0);
        topology.add(cldnn::reduce("reduce_prim",
            cldnn::input_info("input"),
            mode, axes, keep_dims));

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
            auto gpu_out = read_network_output_f32(outputs, "reduce_prim", *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            auto ref_out = ref::reduce(input_f32, shape, mode, axes, keep_dims);

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
        print_result(network, "reduce_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_reduce)

}  // namespace bench_kernel
