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
#include <intel_gpu/primitives/lstm_cell.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_lstm_cell : public kernel_base {
public:
    std::string name() const override { return "lstm_cell"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: x[N,input_size] : hidden[N,hidden_size] : cell[N,hidden_size]
        //       : W[4*hidden_size,input_size] : R[4*hidden_size,hidden_size] : B[4*hidden_size]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 6) throw std::runtime_error("lstm_cell requires 6 shapes (x:hidden:cell:W:R:B). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        auto make_layout = [&](size_t idx, cldnn::data_types d) {
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[idx].begin(), shapes[idx].end()));
            return cldnn::layout(ps, d, cldnn::format::bfyx);
        };

        cldnn::layout lay_x = make_layout(0, dt);
        cldnn::layout lay_h = make_layout(1, dt);
        cldnn::layout lay_c = make_layout(2, dt);
        cldnn::layout lay_w = make_layout(3, dt);
        cldnn::layout lay_r = make_layout(4, dt);
        cldnn::layout lay_b = make_layout(5, dt);

        auto x_mem = engine.allocate_memory(lay_x);
        auto h_mem = engine.allocate_memory(lay_h);
        auto c_mem = engine.allocate_memory(lay_c);
        auto w_mem = engine.allocate_memory(lay_w);
        auto r_mem = engine.allocate_memory(lay_r);
        auto b_mem = engine.allocate_memory(lay_b);

        fill_memory_random(x_mem, *stream, dt);
        fill_memory_random(h_mem, *stream, dt);
        fill_memory_random(c_mem, *stream, dt);
        fill_memory_random(w_mem, *stream, dt);
        fill_memory_random(r_mem, *stream, dt);
        fill_memory_random(b_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("x", lay_x));
        topology.add(cldnn::input_layout("hidden", lay_h));
        topology.add(cldnn::input_layout("cell", lay_c));
        topology.add(cldnn::input_layout("W", lay_w));
        topology.add(cldnn::input_layout("R", lay_r));
        topology.add(cldnn::input_layout("B", lay_b));

        topology.add(cldnn::lstm_cell("prim",
            cldnn::input_info("x"),
            cldnn::input_info("hidden"),
            cldnn::input_info("cell"),
            cldnn::input_info("W"),
            cldnn::input_info("R"),
            cldnn::input_info("B"),
            cldnn::input_info(""),  // seq_lengths (empty)
            0.0f,  // clip
            false,  // input_forget
            {cldnn::activation_func::logistic, cldnn::activation_func::hyperbolic_tan, cldnn::activation_func::hyperbolic_tan},
            {},  // activation_params
            cldnn::lstm_weights_order::iofz,
            ov::op::RecurrentSequenceDirection::FORWARD,
            2));  // num_outputs = 2 (hidden_state + cell_state)

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("x", x_mem);
        network.set_input_data("hidden", h_mem);
        network.set_input_data("cell", c_mem);
        network.set_input_data("W", w_mem);
        network.set_input_data("R", r_mem);
        network.set_input_data("B", b_mem);

        bool test_passed = true;
        perf_timer timer;
        auto wall_start = std::chrono::high_resolution_clock::now();

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto xf = read_memory_to_f32(x_mem, *stream);
            auto hf = read_memory_to_f32(h_mem, *stream);
            auto cf = read_memory_to_f32(c_mem, *stream);
            auto Wf = read_memory_to_f32(w_mem, *stream);
            auto Rf = read_memory_to_f32(r_mem, *stream);
            auto Bf = read_memory_to_f32(b_mem, *stream);
            int64_t batch_sz = shapes[0][0];
            int64_t in_sz = shapes[0][1];
            int64_t hid_sz = shapes[1][1];
            auto ref_out = ref::lstm_cell(xf, hf, cf, Wf, Rf, Bf, batch_sz, in_sz, hid_sz);
            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            atol *= 4; rtol *= 4;  // LSTM accumulates FP error
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

REGISTER_KERNEL(bench_lstm_cell)

}  // namespace bench_kernel
