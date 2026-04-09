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
#include <intel_gpu/primitives/ctc_greedy_decoder.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_ctc_greedy_decoder : public kernel_base {
public:
    std::string name() const override { return "ctc_greedy_decoder"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: logits[T,N,C]:seq_len[N]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("ctc_greedy_decoder requires 2 shapes (logits:seq_len). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];
        uint32_t blank_index = 0;
        bool merge_repeated = true;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        auto logits_mem = engine.allocate_memory(lay0);
        auto seq_mem = engine.allocate_memory(lay1);
        fill_memory_random(logits_mem, *stream, dt);
        fill_memory_random(seq_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("logits", lay0));
        topology.add(cldnn::input_layout("seq_len", lay1));
        topology.add(cldnn::ctc_greedy_decoder("prim",
            {cldnn::input_info("logits"), cldnn::input_info("seq_len")},
            blank_index, merge_repeated, cldnn::data_types::i32, 1));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("logits", logits_mem);
        network.set_input_data("seq_len", seq_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        [[maybe_unused]] acc_result acc_res;
        if (config.is_acc()) {
            throw bench_unimplemented("CPU reference not implemented");
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

REGISTER_KERNEL(bench_ctc_greedy_decoder)

}  // namespace bench_kernel
