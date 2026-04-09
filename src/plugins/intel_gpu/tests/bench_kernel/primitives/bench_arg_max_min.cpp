// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/arg_max_min.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// ArgMaxMin (TopK) kernel benchmark
//
// Usage:
//   --arg_max_min --dt=f16 --shapes=1x1000 --topk_mode=0 --top_k=5 --axis=-1
//
// topk_mode: 0=max, 1=min
// top_k: number of top elements
// axis: dimension to operate on (-1=last)
// ============================================================================

class bench_arg_max_min : public kernel_base {
public:
    std::string name() const override { return "arg_max_min"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("ArgMaxMin requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        auto& in_shape = shapes[0];
        size_t rank = in_shape.size();

        // Parse config
        auto topk_mode = (config.topk_mode == 1) ? ov::op::TopKMode::MIN : ov::op::TopKMode::MAX;
        uint32_t top_k = std::max(config.top_k, 1);
        int64_t ax = config.axis;
        if (ax < 0) ax += static_cast<int64_t>(rank);

        auto exec_config = make_exec_config(config, "argmax_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::arg_max_min("argmax_prim",
            {cldnn::input_info("input")},
            topk_mode,
            top_k,
            ax,
            ov::op::TopKSortType::SORT_VALUES,
            false,  // values_first
            false,  // stable
            cldnn::data_types::f32,
            1));  // num_outputs

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "argmax_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            bool is_max = (config.topk_mode != 1);
            auto ref_out = ref::arg_max_min(input_f32, in_shape, ax, static_cast<int64_t>(top_k), is_max);
            float atol, rtol;
            get_default_tolerance(cldnn::data_types::f32, atol, rtol);
            atol = 1.0f;  // indices comparison
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
        print_result(network, "argmax_prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_arg_max_min)

}  // namespace bench_kernel
