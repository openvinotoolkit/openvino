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
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Gather kernel benchmark
//
// Usage:
//   --gather --dt=f16 dict_shape:indices_shape
//
// Gathers values from dict along axis 0 using indices.
//
// Example:
//   --gather --dt=f16 32000x4096:1x128     (embedding lookup)
//   --gather --dt=f16 10x20:5              (simple 1D gather)
// ============================================================================

class bench_gather : public kernel_base {
public:
    std::string name() const override { return "gather"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Gather requires 2 shapes (dict:indices). Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dict_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        // Gather axis: from config (verbose log), post-ops, or default 0
        int64_t axis = config.gather_axis;
        if (!config.attr_post_ops_str.empty()) {
            axis = std::stoll(config.attr_post_ops_str);
        }

        auto exec_config = make_exec_config(config, "gather_prim");
        auto stream = engine.create_stream(exec_config);

        // Dictionary input
        auto& dict_shape = shapes[0];
        ov::PartialShape dict_ps(std::vector<ov::Dimension>(dict_shape.begin(), dict_shape.end()));
        cldnn::layout dict_layout(dict_ps, dict_dt, get_input_format(config, 0, dict_shape.size()));
        auto dict_mem = engine.allocate_memory(dict_layout);
        fill_memory_random(dict_mem, *stream, dict_dt);

        // Indices input (always i32)
        auto& idx_shape = shapes[1];
        ov::PartialShape idx_ps(std::vector<ov::Dimension>(idx_shape.begin(), idx_shape.end()));
        cldnn::layout idx_layout(idx_ps, cldnn::data_types::i32, get_input_format(config, 1, idx_shape.size()));
        auto idx_mem = engine.allocate_memory(idx_layout);
        // Fill indices with valid values [0, dict_shape[axis])
        {
            int32_t max_val = static_cast<int32_t>(dict_shape[axis >= 0 ? axis : dict_shape.size() + axis]);
            auto ptr = idx_mem->lock(*stream, cldnn::mem_lock_type::write);
            auto* data = static_cast<int32_t*>(ptr);
            size_t count = idx_layout.count();
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<int32_t>(rand() % max_val);
            }
            idx_mem->unlock(*stream);
        }

        // Compute output shape
        int64_t input_rank = static_cast<int64_t>(dict_shape.size());
        int64_t norm_axis = axis >= 0 ? axis : input_rank + axis;
        ov::Shape output_ov_shape;
        for (int64_t i = 0; i < norm_axis; ++i)
            output_ov_shape.push_back(dict_shape[i]);
        for (auto d : idx_shape)
            output_ov_shape.push_back(d);
        for (int64_t i = norm_axis + 1; i < input_rank; ++i)
            output_ov_shape.push_back(dict_shape[i]);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("dict", dict_layout));
        topology.add(cldnn::input_layout("indices", idx_layout));
        topology.add(cldnn::gather("gather_prim",
            cldnn::input_info("dict"),
            cldnn::input_info("indices"),
            axis,
            input_rank,
            output_ov_shape,
            config.batch_dim,
            config.support_neg_ind != 0));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("dict", dict_mem);
        network.set_input_data("indices", idx_mem);

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
            auto gpu_out = read_network_output_f32(outputs, "gather_prim", *stream);
            auto dict_f32 = read_memory_to_f32(dict_mem, *stream);
            // Read indices as i32
            auto idx_f32 = read_memory_to_f32(idx_mem, *stream);
            std::vector<int32_t> indices(idx_f32.size());
            for (size_t i = 0; i < idx_f32.size(); ++i)
                indices[i] = static_cast<int32_t>(idx_f32[i]);

            auto ref_out = ref::gather(dict_f32, indices, dict_shape, idx_shape, axis);

            float atol, rtol;
            get_default_tolerance(dict_dt, atol, rtol);
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
        print_result(network, "gather_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_gather)

}  // namespace bench_kernel
