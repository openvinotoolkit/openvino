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
#include <intel_gpu/primitives/scatter_elements_update.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// ScatterElementsUpdate kernel benchmark
//
// Usage:
//   --scatter_elements_update --dt=f16 --shapes=4x4:2x2:2x2 --gather_axis=0
//
// Scatters updates into data at positions specified by indices along axis.
// shapes = data:indices:updates (indices and updates same shape)
// ============================================================================

class bench_scatter_elements_update : public kernel_base {
public:
    std::string name() const override { return "scatter_elements_update"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 3) {
            throw std::runtime_error("ScatterElementsUpdate requires 3 shapes (data:indices:updates). Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int64_t axis = config.gather_axis;

        auto& data_shape = shapes[0];
        auto& indices_shape = shapes[1];
        size_t rank = data_shape.size();

        auto exec_config = make_exec_config(config, "scatter_elem_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape data_ps(std::vector<ov::Dimension>(data_shape.begin(), data_shape.end()));
        cldnn::layout data_layout(data_ps, dt, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(data_layout);
        fill_memory_random(data_mem, *stream, dt);

        ov::PartialShape idx_ps(std::vector<ov::Dimension>(indices_shape.begin(), indices_shape.end()));
        cldnn::layout idx_layout(idx_ps, cldnn::data_types::i32, cldnn::format::bfyx);
        auto idx_mem = engine.allocate_memory(idx_layout);
        {
            cldnn::mem_lock<int32_t> lock(idx_mem, *stream);
            size_t total = 1;
            for (auto d : indices_shape) total *= d;
            size_t ax = (axis >= 0) ? axis : axis + rank;
            int64_t axis_size = data_shape[ax];
            // Generate unique indices per axis-slice to avoid GPU race conditions
            std::vector<int32_t> pool(axis_size);
            std::iota(pool.begin(), pool.end(), 0);
            std::shuffle(pool.begin(), pool.end(), std::mt19937(std::random_device{}()));
            for (size_t i = 0; i < total; ++i) lock[i] = pool[i % pool.size()];
        }

        ov::PartialShape upd_ps(std::vector<ov::Dimension>(indices_shape.begin(), indices_shape.end()));
        cldnn::layout upd_layout(upd_ps, dt, cldnn::format::bfyx);
        auto upd_mem = engine.allocate_memory(upd_layout);
        fill_memory_random(upd_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", data_layout));
        topology.add(cldnn::input_layout("indices", idx_layout));
        topology.add(cldnn::input_layout("updates", upd_layout));
        topology.add(cldnn::scatter_elements_update("scatter_elem_prim",
            cldnn::input_info("data"), cldnn::input_info("indices"),
            cldnn::input_info("updates"), axis));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);
        network.set_input_data("indices", idx_mem);
        network.set_input_data("updates", upd_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "scatter_elem_prim", *stream);
            auto data_f32 = read_memory_to_f32(data_mem, *stream);
            auto upd_f32 = read_memory_to_f32(upd_mem, *stream);
            std::vector<int32_t> idx_data;
            {
                cldnn::mem_lock<int32_t> lock(idx_mem, *stream);
                size_t total = 1;
                for (auto d : indices_shape) total *= d;
                idx_data.assign(lock.data(), lock.data() + total);
            }

            // Reference: copy data, then scatter
            std::vector<float> ref_out = data_f32;
            size_t ax = (axis >= 0) ? axis : axis + rank;

            std::vector<size_t> data_strides(rank, 1), idx_strides(rank, 1);
            for (int d = (int)rank - 2; d >= 0; --d) {
                data_strides[d] = data_strides[d+1] * data_shape[d+1];
                idx_strides[d] = idx_strides[d+1] * indices_shape[d+1];
            }

            size_t idx_total = 1;
            for (auto d : indices_shape) idx_total *= d;

            for (size_t i = 0; i < idx_total; ++i) {
                // Decompose flat index into coords
                size_t data_offset = 0, rem = i;
                for (size_t d = 0; d < rank; ++d) {
                    size_t coord = rem / idx_strides[d];
                    rem %= idx_strides[d];
                    if (d == ax) {
                        data_offset += idx_data[i] * data_strides[d];
                    } else {
                        data_offset += coord * data_strides[d];
                    }
                }
                ref_out[data_offset] = upd_f32[i];
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
        print_result(network, "scatter_elem_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_scatter_elements_update)

}  // namespace bench_kernel
