// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/adaptive_pooling.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Adaptive Pooling kernel benchmark
//
// Usage:
//   --adaptive_pooling --dt=f16 --shapes=1x64x32x32 --adaptive_pool_mode=0
//       --adaptive_pool_out=1:64:7:7
//
// adaptive_pool_mode: 0=average, 1=max
// adaptive_pool_out:  colon-separated output sizes (e.g. 1:64:7:7)
// ============================================================================

class bench_adaptive_pooling : public kernel_base {
public:
    std::string name() const override { return "adaptive_pooling"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("AdaptivePooling requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        auto& in_shape = shapes[0];

        // Parse output sizes
        std::vector<int64_t> out_sizes;
        if (!config.adaptive_pool_out.empty()) {
            out_sizes = parse_colon_vec(config.adaptive_pool_out);
        }
        if (out_sizes.empty()) {
            // Default: same as input with spatial dims halved
            out_sizes = std::vector<int64_t>(in_shape.begin(), in_shape.end());
            for (size_t d = 2; d < out_sizes.size(); ++d) out_sizes[d] = std::max((int64_t)1, in_shape[d] / 2);
        }

        // Build output tensor
        cldnn::tensor out_tensor;
        if (out_sizes.size() >= 4) {
            out_tensor = cldnn::tensor(static_cast<int32_t>(out_sizes[0]),
                                       static_cast<int32_t>(out_sizes[1]),
                                       static_cast<int32_t>(out_sizes[3]),
                                       static_cast<int32_t>(out_sizes[2]));
        } else if (out_sizes.size() >= 2) {
            out_tensor = cldnn::tensor(static_cast<int32_t>(out_sizes[0]),
                                       static_cast<int32_t>(out_sizes[1]), 1, 1);
        }

        auto pool_mode = (config.adaptive_pool_mode == 1)
            ? cldnn::adaptive_pooling_mode::max
            : cldnn::adaptive_pooling_mode::average;

        auto exec_config = make_exec_config(config, "adaptive_pool_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));

        // Provide output_shape as a data input (for dynamic shape support)
        // Output shape tensor contains the spatial dims
        size_t spatial_dims = out_sizes.size() > 2 ? out_sizes.size() - 2 : 0;
        if (spatial_dims == 0) spatial_dims = 2;
        std::vector<int32_t> shape_data;
        for (size_t d = 2; d < out_sizes.size(); ++d) shape_data.push_back(static_cast<int32_t>(out_sizes[d]));
        if (shape_data.empty()) shape_data = {1, 1};

        ov::PartialShape shape_ps({static_cast<int64_t>(shape_data.size())});
        cldnn::layout shape_layout(shape_ps, cldnn::data_types::i32, cldnn::format::bfyx);
        auto shape_mem = engine.allocate_memory(shape_layout);
        {
            cldnn::mem_lock<int32_t> lock(shape_mem, *stream);
            for (size_t i = 0; i < shape_data.size(); ++i) lock[i] = shape_data[i];
        }
        topology.add(cldnn::data("output_shape", shape_mem));

        if (pool_mode == cldnn::adaptive_pooling_mode::average) {
            topology.add(cldnn::adaptive_pooling("adaptive_pool_prim",
                cldnn::input_info("input"), cldnn::input_info("output_shape")));
        } else {
            topology.add(cldnn::adaptive_pooling("adaptive_pool_prim",
                cldnn::input_info("input"), cldnn::input_info("output_shape"),
                cldnn::data_types::i64));
        }

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        [[maybe_unused]] acc_result acc_res;
        if (config.is_acc()) {
            throw bench_unimplemented("CPU reference not implemented");
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "adaptive_pool_prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_adaptive_pooling)

}  // namespace bench_kernel
