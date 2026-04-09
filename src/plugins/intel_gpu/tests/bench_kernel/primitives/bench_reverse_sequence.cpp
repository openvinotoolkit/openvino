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
#include <intel_gpu/primitives/reverse_sequence.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_reverse_sequence : public kernel_base {
public:
    std::string name() const override { return "reverse_sequence"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("reverse_sequence requires 2 shapes (data:seq_lengths). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int32_t seq_axis = config.seq_axis;
        int32_t batch_axis = config.batch_dim;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, cldnn::data_types::i32, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(lay0);
        auto seq_mem = engine.allocate_memory(lay1);
        fill_memory_random(data_mem, *stream, dt);
        // seq_lengths should be valid (1 to dim_size)
        [[maybe_unused]] int max_seq = static_cast<int>(shapes[0][seq_axis]);
        fill_memory_random(seq_mem, *stream, cldnn::data_types::i32);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", lay0));
        topology.add(cldnn::input_layout("seq_lens", lay1));
        topology.add(cldnn::reverse_sequence("prim",
            cldnn::input_info("data"), cldnn::input_info("seq_lens"), seq_axis, batch_axis));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);
        network.set_input_data("seq_lens", seq_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto data_f32 = read_memory_to_f32(data_mem, *stream);
            auto seq_f32 = read_memory_to_f32(seq_mem, *stream);
            auto ref_out = ref::reverse_sequence(data_f32, seq_f32, shapes[0], static_cast<int64_t>(seq_axis), static_cast<int64_t>(batch_axis));
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

REGISTER_KERNEL(bench_reverse_sequence)

}  // namespace bench_kernel
