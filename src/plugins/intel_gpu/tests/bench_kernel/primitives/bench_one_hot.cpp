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
#include <intel_gpu/primitives/one_hot.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_one_hot : public kernel_base {
public:
    std::string name() const override { return "one_hot"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("one_hot requires at least 1 shape (indices). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int64_t depth = config.one_hot_depth > 0 ? config.one_hot_depth : 10;
        int64_t oh_axis = config.axis == -1 ? static_cast<int64_t>(shapes[0].size()) : config.axis;
        float on_value = config.one_hot_on_value;
        float off_value = config.one_hot_off_value;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        cldnn::layout input_layout(ps, cldnn::data_types::i32, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, cldnn::data_types::i32);

        // Compute output shape
        auto& in_shape = shapes[0];
        std::vector<int64_t> out_dims;
        for (size_t i = 0; i < in_shape.size(); ++i) {
            if (static_cast<int64_t>(i) == oh_axis) out_dims.push_back(depth);
            out_dims.push_back(in_shape[i]);
        }
        if (oh_axis >= static_cast<int64_t>(in_shape.size())) out_dims.push_back(depth);

        // Pad to 4D for tensor
        while (out_dims.size() < 4) out_dims.insert(out_dims.begin(), 1);
        cldnn::tensor out_tensor(cldnn::batch(out_dims[0]), cldnn::feature(out_dims[1]),
                                 cldnn::spatial(out_dims[3], out_dims[2]));

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::one_hot("prim", cldnn::input_info("input"), out_tensor, dt,
                     oh_axis, depth, false, on_value, off_value));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto ref_out = ref::one_hot(input_f32, in_shape, depth, oh_axis, on_value, off_value);
            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
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
        print_result(network, "prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_one_hot)

}  // namespace bench_kernel
