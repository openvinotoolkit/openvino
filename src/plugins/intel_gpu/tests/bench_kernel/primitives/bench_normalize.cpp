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
#include <intel_gpu/primitives/normalize.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Normalize (L2 norm) kernel benchmark
//
// Usage:
//   --normalize --dt=f16 --shapes=1x64x112x112 --across_spatial=1 --epsilon=1e-10
//
// L2 normalization with per-channel scaling.
// ============================================================================

class bench_normalize : public kernel_base {
public:
    std::string name() const override { return "normalize"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Normalize requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        bool across_spatial = (config.across_spatial != 0);
        float eps = 1e-10f;
        if (!config.epsilon.empty()) {
            try { eps = std::stof(config.epsilon); } catch (...) {}
        }

        auto& in_shape = shapes[0];
        size_t rank = in_shape.size();

        auto exec_config = make_exec_config(config, "normalize_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, get_input_format(config, 0, in_shape.size()));
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        // Scale: per-channel (feature dimension = in_shape[1] if rank >= 2, else 1)
        int64_t channels = (rank >= 2) ? in_shape[1] : 1;
        std::vector<ov::Dimension> scale_dims(rank, 1);
        if (rank >= 2) scale_dims[1] = channels;
        cldnn::layout scale_layout(ov::PartialShape(scale_dims), cldnn::data_types::f32, get_input_format(config, 1, rank));
        auto scale_mem = engine.allocate_memory(scale_layout);
        fill_memory_random(scale_mem, *stream, cldnn::data_types::f32);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::data("scale", scale_mem));
        topology.add(cldnn::normalize("normalize_prim",
            cldnn::input_info("input"), "scale", across_spatial, eps));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "normalize_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto scale_f32 = read_memory_to_f32(scale_mem, *stream);

            // Reference L2 normalize
            size_t batch = (rank >= 1) ? in_shape[0] : 1;
            size_t C = channels;
            size_t spatial = 1;
            for (size_t d = 2; d < rank; ++d) spatial *= in_shape[d];

            std::vector<float> ref_out(input_f32.size());
            for (size_t b = 0; b < batch; ++b) {
                if (across_spatial) {
                    // Norm across all channels and spatial
                    float sum_sq = 0;
                    size_t base = b * C * spatial;
                    for (size_t i = 0; i < C * spatial; ++i) {
                        sum_sq += input_f32[base + i] * input_f32[base + i];
                    }
                    float norm = std::sqrt(sum_sq + eps);
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t s = 0; s < spatial; ++s) {
                            size_t idx = base + c * spatial + s;
                            ref_out[idx] = input_f32[idx] / norm * scale_f32[c];
                        }
                    }
                } else {
                    // Norm per spatial position across channels
                    for (size_t s = 0; s < spatial; ++s) {
                        float sum_sq = 0;
                        for (size_t c = 0; c < C; ++c) {
                            size_t idx = b * C * spatial + c * spatial + s;
                            sum_sq += input_f32[idx] * input_f32[idx];
                        }
                        float norm = std::sqrt(sum_sq + eps);
                        for (size_t c = 0; c < C; ++c) {
                            size_t idx = b * C * spatial + c * spatial + s;
                            ref_out[idx] = input_f32[idx] / norm * scale_f32[c];
                        }
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
        print_result(network, "normalize_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_normalize)

}  // namespace bench_kernel
