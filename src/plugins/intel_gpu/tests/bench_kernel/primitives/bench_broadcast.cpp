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
#include <intel_gpu/primitives/broadcast.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Broadcast kernel benchmark
//
// Usage:
//   --broadcast --dt=f16 --shapes=input_shape:target_shape
//
// Broadcasts input to target_shape using NUMPY broadcast rules.
//
// Example:
//   --broadcast --dt=u8 --shapes=1x1x128x128:1x1x128x128
//   --broadcast --dt=f16 --shapes=1x64x1:1x64x128
// ============================================================================

class bench_broadcast : public kernel_base {
public:
    std::string name() const override { return "broadcast"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Broadcast requires 2 shapes (input:target). Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "broadcast_prim");
        auto stream = engine.create_stream(exec_config);

        auto& input_shape = shapes[0];
        auto& target_shape = shapes[1];

        // Input memory
        ov::PartialShape input_ps(std::vector<ov::Dimension>(input_shape.begin(), input_shape.end()));
        cldnn::layout input_layout_desc(input_ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout_desc);
        fill_memory_random(input_mem, *stream, dt);

        // Target shape for broadcast
        ov::Shape target_ov(target_shape.begin(), target_shape.end());

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout_desc));

        auto bcast_prim = cldnn::broadcast("broadcast_prim",
            cldnn::input_info("input"),
            target_ov,
            ov::AxisSet{},
            ov::op::BroadcastType::NUMPY);
        bcast_prim.output_pshape = ov::PartialShape(target_ov);
        topology.add(bcast_prim);

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
            auto gpu_out = read_network_output_f32(outputs, "broadcast_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto ref_out = ref::broadcast(input_f32, input_shape, target_shape);

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
        print_result(network, "broadcast_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_broadcast)

}  // namespace bench_kernel
