// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/concatenation.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Permute kernel benchmark
//
// Usage:
//   --permute --dt=f16 shape  (default: reverse order)
//
// Example:
//   --permute --dt=f16 1x32x128x128    -> permute [0,3,2,1]
//   --permute --dt=f16 --attr-post-ops=0x2x1x3 1x32x128x128
// ============================================================================

class bench_permute : public kernel_base {
public:
    std::string name() const override { return "permute"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Permute requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto& shape = shapes[0];

        // Parse permute order from --permute_order or --attr-post-ops (legacy) or default
        std::vector<uint16_t> order;
        if (!config.permute_order.empty()) {
            auto dims = parse_colon_vec(config.permute_order);
            for (auto d : dims) order.push_back(static_cast<uint16_t>(d));
        } else if (!config.attr_post_ops_str.empty()) {
            // Legacy: order was passed via --attr-post-ops
            auto dims = parse_dims(config.attr_post_ops_str);
            for (auto d : dims) order.push_back(static_cast<uint16_t>(d));
        } else {
            // Default: swap last two dims (common transpose)
            order.resize(shape.size());
            std::iota(order.begin(), order.end(), 0);
            if (order.size() >= 2) {
                std::swap(order[order.size()-1], order[order.size()-2]);
            }
        }

        auto exec_config = make_exec_config(config, "permute_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        auto fmt = get_input_format(config, 0, shape.size());
        cldnn::layout layout(ps, dt, fmt);
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        topology.add(cldnn::permute("permute_prim", cldnn::input_info("input"), order));

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
            auto gpu_out = read_network_output_f32(outputs, "permute_prim", *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            auto ref_out = ref::permute(input_f32, shape, order);

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
        print_result(network, "permute_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_permute)

// ============================================================================
// Reorder kernel benchmark
//
// Usage:
//   --reorder --dt=f16:f32 shape  (reorder from first dt to second dt)
//
// Example:
//   --reorder --dt=f16:f32 1x64x56x56
//   --reorder --dt=f32:f16 1x4096
// ============================================================================

class bench_reorder : public kernel_base {
public:
    std::string name() const override { return "reorder"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Reorder requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        cldnn::data_types output_dt = dts.size() > 1 ? dts[1] : cldnn::data_types::f32;

        auto exec_config = make_exec_config(config, "reorder_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        auto fmt = get_input_format(config, 0, shape.size());
        cldnn::layout layout(ps, input_dt, fmt);
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, input_dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        // truncate flag: Convert op uses truncate=true, plain Reorder uses false.
        // When --truncate is specified, use GPU truncation mode (convert_long then narrow).
        topology.add(cldnn::reorder("reorder_prim",
            cldnn::input_info("input"),
            fmt, output_dt,
            std::vector<float>(),
            cldnn::reorder_mean_mode::subtract,
            cldnn::padding(),
            config.truncate));

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
            auto gpu_out = read_network_output_f32(outputs, "reorder_prim", *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            auto ref_out = ref::reorder(input_f32, input_dt, output_dt, config.truncate);

            float atol, rtol;
            get_default_tolerance(output_dt, atol, rtol);
            // For truncation mode with integer output, allow small mismatch for edge cases
            float threshold = 0.0f;
            if (config.truncate && !ref::is_fp_type(output_dt)) {
                threshold = 0.001f;  // allow 0.1% mismatch
            }
            acc_res = compare_f32(gpu_out, ref_out, atol, rtol, threshold);
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
        print_result(network, "reorder_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_reorder)

// ============================================================================
// Concatenation kernel benchmark
//
// Usage:
//   --concatenation --dt=f16 shape0:shape1[:shape2...]
//   Concatenates along last axis by default
//
// Example:
//   --concatenation --dt=f16 1x64x56x56:1x64x56x56
// ============================================================================

class bench_concatenation : public kernel_base {
public:
    std::string name() const override { return "concatenation"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Concatenation requires at least 2 shapes. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        // Concat axis: from config (verbose log) or post-ops or default to 1 (channel)
        int64_t axis = config.concat_axis;
        if (!config.attr_post_ops_str.empty()) {
            axis = std::stoll(config.attr_post_ops_str);
        }

        auto exec_config = make_exec_config(config, "concat_prim");
        auto stream = engine.create_stream(exec_config);

        cldnn::topology topology;
        std::vector<cldnn::input_info> inputs;
        std::vector<cldnn::memory::ptr> memories;

        for (size_t i = 0; i < shapes.size(); ++i) {
            std::string id = "input" + std::to_string(i);
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[i].begin(), shapes[i].end()));
            cldnn::layout layout(ps, dt, get_input_format(config, i, shapes[i].size()));
            auto mem = engine.allocate_memory(layout);
            fill_memory_random(mem, *stream, dt);
            topology.add(cldnn::input_layout(id, layout));
            inputs.push_back(cldnn::input_info(id));
            memories.push_back(mem);
        }

        topology.add(cldnn::concatenation("concat_prim", inputs, axis));

        cldnn::network network(engine, topology, exec_config);
        for (size_t i = 0; i < memories.size(); ++i) {
            network.set_input_data("input" + std::to_string(i), memories[i]);
        }

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
            auto gpu_out = read_network_output_f32(outputs, "concat_prim", *stream);

            std::vector<std::vector<float>> input_data;
            for (auto& m : memories) input_data.push_back(read_memory_to_f32(m, *stream));
            auto ref_out = ref::concat(input_data, shapes, axis);

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
        print_result(network, "concat_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_concatenation)

}  // namespace bench_kernel
