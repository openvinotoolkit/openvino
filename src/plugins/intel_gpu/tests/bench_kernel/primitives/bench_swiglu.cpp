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
#include <intel_gpu/primitives/swiglu.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// SwiGLU (Swish Gated Linear Unit) kernel benchmark
//
// Usage:
//   --swiglu_standalone --dt=f16 shape
//
// SwiGLU splits input along last axis into two halves, applies swish to
// the gate half and multiplies with the up half.
// Input shape last dim must be 2x the output last dim.
//
// Example:
//   --swiglu_standalone --dt=f16 1x128x22016   (output: 1x128x11008)
//   --swiglu_standalone --dt=f16 1x1x22016
// ============================================================================

class bench_swiglu_standalone : public kernel_base {
public:
    std::string name() const override { return "swiglu"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("SwiGLU requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "swiglu_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout input_layout_desc(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout_desc);
        fill_memory_random(input_mem, *stream, dt);

        // SwiGLU splits along last axis: input last_dim = 2 * output last_dim
        int64_t axis = (config.split_axis >= 0) ? static_cast<int64_t>(config.split_axis)
                                                : static_cast<int64_t>(shape.size()) - 1;
        int64_t split_length = (config.split_length >= 0) ? static_cast<int64_t>(config.split_length)
                                                          : static_cast<int64_t>(shape.back()) / 2;
        auto glu_type = static_cast<ov::op::internal::GLU::GluType>(config.glu_type);
        size_t gate_idx = static_cast<size_t>(config.gate_idx);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout_desc));
        topology.add(cldnn::swiglu("swiglu_prim",
            cldnn::input_info("input"),
            axis,
            split_length,
            glu_type,
            gate_idx,
            cldnn::tensor()));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

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
            auto gpu_out = read_network_output_f32(outputs, "swiglu_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto ref_out = ref::swiglu(input_f32, shape, axis, split_length,
                                       config.glu_type, config.gate_idx);

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
        print_result(network, "swiglu_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_swiglu_standalone)

}  // namespace bench_kernel
