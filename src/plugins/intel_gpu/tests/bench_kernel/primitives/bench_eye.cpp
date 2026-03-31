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
#include <intel_gpu/primitives/eye.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_eye : public kernel_base {
public:
    std::string name() const override { return "eye"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("eye requires at least 1 shape (output). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];
        int32_t shift = config.eye_diagonal;
        auto& out_shape = shapes[0];

        // Pad shape to 4D
        std::vector<int64_t> s4 = out_shape;
        while (s4.size() < 4) s4.insert(s4.begin(), 1);
        cldnn::tensor out_t(cldnn::batch(s4[0]), cldnn::feature(s4[1]),
                            cldnn::spatial(s4[3], s4[2]));

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        // Eye requires shape inputs (num_rows, num_cols, diagonal_idx, batch_shape)
        auto make_i32_scalar = [&](int32_t val) {
            ov::PartialShape ps({1});
            cldnn::layout lay(ps, cldnn::data_types::i32, cldnn::format::bfyx);
            auto mem = engine.allocate_memory(lay);
            cldnn::mem_lock<int32_t> lock(mem, *stream);
            lock[0] = val;
            return mem;
        };

        int64_t num_rows = out_shape.size() >= 2 ? out_shape[out_shape.size()-2] : out_shape[0];
        int64_t num_cols = out_shape.size() >= 2 ? out_shape[out_shape.size()-1] : out_shape[0];

        auto rows_mem = make_i32_scalar(static_cast<int32_t>(num_rows));
        auto cols_mem = make_i32_scalar(static_cast<int32_t>(num_cols));
        auto diag_mem = make_i32_scalar(shift);

        cldnn::topology topology;
        topology.add(cldnn::data("rows", rows_mem));
        topology.add(cldnn::data("cols", cols_mem));
        topology.add(cldnn::data("diag", diag_mem));
        topology.add(cldnn::eye("prim",
            {cldnn::input_info("rows"), cldnn::input_info("cols"), cldnn::input_info("diag")},
            out_t, shift, dt));

        cldnn::network network(engine, topology, exec_config);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto ref_out = ref::eye(num_rows, num_cols, static_cast<int64_t>(shift), out_shape);
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

REGISTER_KERNEL(bench_eye)

}  // namespace bench_kernel
