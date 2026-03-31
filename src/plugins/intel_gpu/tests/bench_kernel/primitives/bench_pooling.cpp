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
#include <intel_gpu/primitives/pooling.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Pooling kernel benchmark
//
// Usage:
//   --pooling --dt=f16 shape
//   Default: max pooling with 2x2 kernel, stride 2x2
//
// Example:
//   --pooling --dt=f16 1x64x112x112
// ============================================================================

class bench_pooling : public kernel_base {
public:
    std::string name() const override { return "pooling"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Pooling requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        // Pooling mode: from config or post-ops string
        cldnn::pooling_mode mode = static_cast<cldnn::pooling_mode>(config.pool_mode);
        if (!config.attr_post_ops_str.empty()) {
            if (config.attr_post_ops_str == "avg" || config.attr_post_ops_str == "average")
                mode = cldnn::pooling_mode::average;
        }

        size_t spatial_dims = shapes[0].size() - 2;
        ov::Shape kernel_size(spatial_dims, 2);
        ov::Strides stride(spatial_dims, 2);
        ov::Shape pads_begin(spatial_dims, 0);
        ov::Shape pads_end(spatial_dims, 0);

        // Override from config if provided
        if (!config.pool_kernel.empty()) {
            auto v = parse_x_vec(config.pool_kernel);
            if (v.size() == spatial_dims) kernel_size = ov::Shape(v.begin(), v.end());
        }
        if (!config.pool_strides.empty()) {
            auto v = parse_x_vec(config.pool_strides);
            if (v.size() == spatial_dims) stride = ov::Strides(v.begin(), v.end());
        }
        if (!config.pads_begin.empty()) {
            auto v = parse_x_vec(config.pads_begin);
            if (v.size() == spatial_dims) pads_begin = ov::Shape(v.begin(), v.end());
        }
        if (!config.pads_end.empty()) {
            auto v = parse_x_vec(config.pads_end);
            if (v.size() == spatial_dims) pads_end = ov::Shape(v.begin(), v.end());
        }

        auto exec_config = make_exec_config(config, "pool_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout layout(ps, dt, get_input_format(config, 0, shape.size()));
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, dt);

        // Rounding type from config
        ov::op::RoundingType rounding_type = static_cast<ov::op::RoundingType>(config.rounding_type);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        topology.add(cldnn::pooling("pool_prim",
            cldnn::input_info("input"),
            mode, kernel_size, stride, pads_begin, pads_end,
            ov::op::PadType::EXPLICIT,
            rounding_type));

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
            auto gpu_out = read_network_output_f32(outputs, "pool_prim", *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            int pool_mode_int = static_cast<int>(mode);

            std::vector<int64_t> k_vec(spatial_dims), s_vec(spatial_dims);
            std::vector<int64_t> pb_vec(spatial_dims, 0), pe_vec(spatial_dims, 0);
            for (size_t d = 0; d < spatial_dims; ++d) {
                k_vec[d] = static_cast<int64_t>(kernel_size[d]);
                s_vec[d] = static_cast<int64_t>(stride[d]);
                pb_vec[d] = static_cast<int64_t>(pads_begin[d]);
                pe_vec[d] = static_cast<int64_t>(pads_end[d]);
            }
            // Max pool init: use sentinel to detect empty pooling windows
            // GPU impls may output 0 (OCL) or -65504 (oneDNN) for empty windows
            float max_init = -65504.0f;
            auto ref_out = ref::poolingNd(input_f32, shape, pool_mode_int, k_vec, s_vec, pb_vec, pe_vec,
                                          rounding_type == ov::op::RoundingType::CEIL, max_init);

            // For max pooling, fixup empty-window positions where ref == max_init
            // (implementation-defined behavior: GPU may output 0, -65504, or -inf)
            if (mode == cldnn::pooling_mode::max) {
                size_t n = std::min(ref_out.size(), gpu_out.size());
                for (size_t i = 0; i < n; ++i) {
                    if (ref_out[i] == max_init) {
                        ref_out[i] = gpu_out[i];  // Accept GPU's choice for empty windows
                    }
                }
            }

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
        print_result(network, "pool_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_pooling)

}  // namespace bench_kernel
