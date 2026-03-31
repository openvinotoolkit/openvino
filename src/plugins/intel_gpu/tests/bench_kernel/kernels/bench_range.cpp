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
#include <intel_gpu/primitives/range.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_range : public kernel_base {
public:
    std::string name() const override { return "range"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: not used directly. Uses range_start/stop/step config.
        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];

        float start_val = config.range_start;
        float stop_val = config.range_stop;
        float step_val = config.range_step != 0 ? config.range_step : 1.0f;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        // Create scalar data inputs for start/stop/step
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

        auto start_mem = make_scalar("start", start_val);
        auto stop_mem = make_scalar("stop", stop_val);
        auto step_mem = make_scalar("step", step_val);

        cldnn::topology topology;
        ov::PartialShape s1({1});
        cldnn::layout scalar_lay(s1, dt, cldnn::format::bfyx);
        topology.add(cldnn::input_layout("start", scalar_lay));
        topology.add(cldnn::input_layout("stop", scalar_lay));
        topology.add(cldnn::input_layout("step", scalar_lay));
        topology.add(cldnn::range("prim",
            {cldnn::input_info("start"), cldnn::input_info("stop"), cldnn::input_info("step")}, dt));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("start", start_mem);
        network.set_input_data("stop", stop_mem);
        network.set_input_data("step", step_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto ref_out = ref::range(start_val, stop_val, step_val);
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

REGISTER_KERNEL(bench_range)

}  // namespace bench_kernel
