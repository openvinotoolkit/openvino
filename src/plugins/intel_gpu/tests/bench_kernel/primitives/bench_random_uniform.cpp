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
#include <intel_gpu/primitives/random_uniform.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_random_uniform : public kernel_base {
public:
    std::string name() const override { return "random_uniform"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("random_uniform requires at least 1 shape (output). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];
        ov::Shape out_shape(shapes[0].begin(), shapes[0].end());

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        // min/max as scalar inputs
        auto make_scalar = [&](const std::string& name, float val) {
            ov::PartialShape ps({1});
            cldnn::layout lay(ps, dt, cldnn::format::bfyx);
            auto mem = engine.allocate_memory(lay);
            if (dt == cldnn::data_types::f32) {
                cldnn::mem_lock<float> lock(mem, *stream);
                lock[0] = val;
            } else {
                cldnn::mem_lock<ov::float16> lock(mem, *stream);
                lock[0] = ov::float16(val);
            }
            return mem;
        };
        auto min_mem = make_scalar("min", 0.0f);
        auto max_mem = make_scalar("max", 1.0f);

        // shape input as i64
        ov::PartialShape shape_ps({static_cast<int64_t>(out_shape.size())});
        cldnn::layout shape_lay(shape_ps, cldnn::data_types::i64, cldnn::format::bfyx);
        auto shape_mem = engine.allocate_memory(shape_lay);
        {
            cldnn::mem_lock<int64_t> lock(shape_mem, *stream);
            for (size_t i = 0; i < out_shape.size(); ++i)
                lock[i] = static_cast<int64_t>(out_shape[i]);
        }

        cldnn::topology topology;
        ov::PartialShape s1({1});
        cldnn::layout scalar_lay(s1, dt, cldnn::format::bfyx);
        topology.add(cldnn::data("shape", shape_mem));
        topology.add(cldnn::input_layout("min_val", scalar_lay));
        topology.add(cldnn::input_layout("max_val", scalar_lay));
        topology.add(cldnn::random_uniform("prim",
            {cldnn::input_info("shape"), cldnn::input_info("min_val"), cldnn::input_info("max_val")},
            dt, 0, 0, out_shape));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("min_val", min_mem);
        network.set_input_data("max_val", max_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto ref_out = ref::make_range_check_ref(gpu_out, 0.0f, 1.0f);
            float atol = 0.0f, rtol = 0.0f;  // range check: values must be in [0,1]
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

REGISTER_KERNEL(bench_random_uniform)

}  // namespace bench_kernel
