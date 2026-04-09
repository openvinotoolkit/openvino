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
#include <intel_gpu/primitives/group_normalization.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// GroupNormalization kernel benchmark
//
// Usage:
//   --group_normalization --dt=f16 --shapes=2x64x32x32 --num_groups=32 --epsilon=1e-5
//
// y = scale * (x - mean) / sqrt(var + eps) + bias, per group of channels.
// Input shape: [N, C, ...], scale/bias shape: [C]
// ============================================================================

class bench_group_normalization : public kernel_base {
public:
    std::string name() const override { return "group_normalization"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("GroupNormalization requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int64_t num_groups = config.num_groups > 0 ? config.num_groups : 1;
        double eps = 1e-5;
        if (!config.epsilon.empty()) {
            try { eps = std::stod(config.epsilon); } catch (...) {}
        }

        auto& in_shape = shapes[0];
        size_t rank = in_shape.size();
        int64_t C = (rank >= 2) ? in_shape[1] : 1;

        auto exec_config = make_exec_config(config, "gn_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        // Scale and bias: shape [C]
        cldnn::layout scale_layout(ov::PartialShape{C}, cldnn::data_types::f32, cldnn::format::bfyx);
        auto scale_mem = engine.allocate_memory(scale_layout);
        auto bias_mem = engine.allocate_memory(scale_layout);
        fill_memory_random(scale_mem, *stream, cldnn::data_types::f32);
        fill_memory_random(bias_mem, *stream, cldnn::data_types::f32);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::input_layout("scale", scale_layout));
        topology.add(cldnn::input_layout("bias", scale_layout));
        topology.add(cldnn::group_normalization("gn_prim",
            cldnn::input_info("input"), cldnn::input_info("scale"),
            cldnn::input_info("bias"), num_groups, eps));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);
        network.set_input_data("scale", scale_mem);
        network.set_input_data("bias", bias_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "gn_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto scale_f32 = read_memory_to_f32(scale_mem, *stream);
            auto bias_f32 = read_memory_to_f32(bias_mem, *stream);

            // Reference GroupNorm
            size_t N = in_shape[0];
            size_t channels = C;
            size_t spatial = 1;
            for (size_t d = 2; d < rank; ++d) spatial *= in_shape[d];
            size_t group_size = channels / num_groups;

            std::vector<float> ref_out(input_f32.size());
            for (size_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < num_groups; ++g) {
                    // Compute mean and variance for this group
                    size_t count = group_size * spatial;
                    double mean = 0;
                    for (size_t c = g * group_size; c < (size_t)(g + 1) * group_size; ++c) {
                        for (size_t s = 0; s < spatial; ++s) {
                            mean += input_f32[n * channels * spatial + c * spatial + s];
                        }
                    }
                    mean /= count;

                    double var = 0;
                    for (size_t c = g * group_size; c < (size_t)(g + 1) * group_size; ++c) {
                        for (size_t s = 0; s < spatial; ++s) {
                            double diff = input_f32[n * channels * spatial + c * spatial + s] - mean;
                            var += diff * diff;
                        }
                    }
                    var /= count;

                    double inv_std = 1.0 / std::sqrt(var + eps);
                    for (size_t c = g * group_size; c < (size_t)(g + 1) * group_size; ++c) {
                        for (size_t s = 0; s < spatial; ++s) {
                            size_t idx = n * channels * spatial + c * spatial + s;
                            ref_out[idx] = static_cast<float>(
                                (input_f32[idx] - mean) * inv_std * scale_f32[c] + bias_f32[c]);
                        }
                    }
                }
            }

            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            // GroupNorm accumulates many elements — relax tolerance significantly for f16
            atol = std::max(atol, 5e-2f);
            rtol = std::max(rtol, 1e-1f);
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
        print_result(network, "gn_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_group_normalization)

}  // namespace bench_kernel
