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
#include <intel_gpu/primitives/gather_elements.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// GatherElements kernel benchmark
//
// Usage:
//   --gather_elements --dt=f16 --shapes=4x4:4x2 --gather_axis=1
//
// Gathers elements from data along axis using indices (like torch.gather).
// shapes = data_shape:indices_shape (output_shape == indices_shape)
// ============================================================================

class bench_gather_elements : public kernel_base {
public:
    std::string name() const override { return "gather_elements"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("GatherElements requires 2 shapes (data:indices). Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int64_t axis = config.gather_axis;

        auto& data_shape = shapes[0];
        auto& indices_shape = shapes[1];
        size_t rank = data_shape.size();

        auto exec_config = make_exec_config(config, "gather_elements_prim");
        auto stream = engine.create_stream(exec_config);

        // Data input
        ov::PartialShape data_ps(std::vector<ov::Dimension>(data_shape.begin(), data_shape.end()));
        cldnn::layout data_layout(data_ps, dt, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(data_layout);
        fill_memory_random(data_mem, *stream, dt);

        // Indices input (i32, values in [0, data_shape[axis]))
        ov::PartialShape idx_ps(std::vector<ov::Dimension>(indices_shape.begin(), indices_shape.end()));
        cldnn::layout idx_layout(idx_ps, cldnn::data_types::i32, cldnn::format::bfyx);
        auto idx_mem = engine.allocate_memory(idx_layout);
        {
            cldnn::mem_lock<int32_t> lock(idx_mem, *stream);
            size_t total = 1;
            for (auto d : indices_shape) total *= d;
            int64_t axis_size = data_shape[axis >= 0 ? axis : axis + rank];
            for (size_t i = 0; i < total; ++i) {
                lock[i] = rand() % axis_size;
            }
        }

        // Output shape == indices shape
        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", data_layout));
        topology.add(cldnn::input_layout("indices", idx_layout));
        topology.add(cldnn::gather_elements("gather_elements_prim",
            cldnn::input_info("data"), cldnn::input_info("indices"),
            cldnn::format::bfyx, cldnn::tensor(), axis));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);
        network.set_input_data("indices", idx_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "gather_elements_prim", *stream);
            auto data_f32 = read_memory_to_f32(data_mem, *stream);

            // Read indices
            std::vector<int32_t> indices_data;
            {
                cldnn::mem_lock<int32_t> lock(idx_mem, *stream);
                size_t total = 1;
                for (auto d : indices_shape) total *= d;
                indices_data.assign(lock.data(), lock.data() + total);
            }

            // Reference implementation
            size_t out_total = 1;
            for (auto d : indices_shape) out_total *= d;

            std::vector<size_t> data_strides(rank, 1), out_strides(rank, 1);
            for (int d = (int)rank - 2; d >= 0; --d) {
                data_strides[d] = data_strides[d+1] * data_shape[d+1];
                out_strides[d] = out_strides[d+1] * indices_shape[d+1];
            }

            size_t ax = (axis >= 0) ? axis : axis + rank;
            std::vector<float> ref_out(out_total);
            for (size_t idx = 0; idx < out_total; ++idx) {
                size_t data_idx = 0, rem = idx;
                for (size_t d = 0; d < rank; ++d) {
                    size_t coord = rem / out_strides[d];
                    rem %= out_strides[d];
                    if (d == ax) {
                        data_idx += indices_data[idx] * data_strides[d];
                    } else {
                        data_idx += coord * data_strides[d];
                    }
                }
                ref_out[idx] = data_f32[data_idx];
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
        print_result(network, "gather_elements_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_gather_elements)

}  // namespace bench_kernel
