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
#include <intel_gpu/primitives/select.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Select kernel benchmark
//
// Usage:
//   --select --dt=u8:f16 --shapes=shape
//
// Performs elementwise select: output = mask ? input1 : input2
// mask is u8 (boolean), input1/input2 and output are the second dt.
// All tensors have the same shape.
//
// Example:
//   --select --dt=u8:f16 --shapes=1x1x128x128
//   --select --dt=u8:f16 --shapes=1x1x128x256
// ============================================================================

class bench_select : public kernel_base {
public:
    std::string name() const override { return "select"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Select requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        // mask type and data type
        cldnn::data_types mask_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::u8;
        cldnn::data_types data_dt = dts.size() > 1 ? dts[1] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "select_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];

        // Mask memory (u8, values 0 or 1)
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout mask_layout(ps, mask_dt, cldnn::format::bfyx);
        auto mask_mem = engine.allocate_memory(mask_layout);
        {
            // Fill mask with random 0/1
            cldnn::mem_lock<uint8_t> lock(mask_mem, *stream);
            size_t total = 1;
            for (auto d : shape) total *= d;
            for (size_t i = 0; i < total; ++i) {
                lock[i] = static_cast<uint8_t>(rand() % 2);
            }
        }

        // Input1 and Input2
        cldnn::layout data_layout(ps, data_dt, cldnn::format::bfyx);
        auto input1_mem = engine.allocate_memory(data_layout);
        auto input2_mem = engine.allocate_memory(data_layout);
        fill_memory_random(input1_mem, *stream, data_dt);
        fill_memory_random(input2_mem, *stream, data_dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("mask", mask_layout));
        topology.add(cldnn::input_layout("input1", data_layout));
        topology.add(cldnn::input_layout("input2", data_layout));
        topology.add(cldnn::select("select_prim",
            cldnn::input_info("mask"),
            cldnn::input_info("input1"),
            cldnn::input_info("input2")));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("mask", mask_mem);
        network.set_input_data("input1", input1_mem);
        network.set_input_data("input2", input2_mem);

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
            auto gpu_out = read_network_output_f32(outputs, "select_prim", *stream);
            auto input1_f32 = read_memory_to_f32(input1_mem, *stream);
            auto input2_f32 = read_memory_to_f32(input2_mem, *stream);

            // Read mask as u8
            std::vector<uint8_t> mask_data;
            {
                cldnn::mem_lock<uint8_t> lock(mask_mem, *stream);
                size_t total = 1;
                for (auto d : shape) total *= d;
                mask_data.assign(lock.data(), lock.data() + total);
            }

            auto ref_out = ref::select(mask_data, input1_f32, input2_f32);

            float atol, rtol;
            get_default_tolerance(data_dt, atol, rtol);
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
        print_result(network, "select_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_select)

}  // namespace bench_kernel
