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
#include <intel_gpu/primitives/ctc_loss.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_ctc_loss : public kernel_base {
public:
    std::string name() const override { return "ctc_loss"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: log_probs[T,N,C] : targets[N,S] : input_lengths[N] : target_lengths[N] : blank_index[1]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 4) throw std::runtime_error("ctc_loss requires >=4 shapes. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        ov::PartialShape ps2(std::vector<ov::Dimension>(shapes[2].begin(), shapes[2].end()));
        ov::PartialShape ps3(std::vector<ov::Dimension>(shapes[3].begin(), shapes[3].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, cldnn::data_types::i32, cldnn::format::bfyx);
        cldnn::layout lay2(ps2, cldnn::data_types::i32, cldnn::format::bfyx);
        cldnn::layout lay3(ps3, cldnn::data_types::i32, cldnn::format::bfyx);

        auto log_probs_mem = engine.allocate_memory(lay0);
        auto targets_mem   = engine.allocate_memory(lay1);
        auto in_len_mem    = engine.allocate_memory(lay2);
        auto tgt_len_mem   = engine.allocate_memory(lay3);
        fill_memory_random(log_probs_mem, *stream, dt);
        // Targets: bounded to [0, C-1]
        {
            int64_t C = shapes[0].back();
            [[maybe_unused]] int64_t S = shapes[1].size() > 1 ? shapes[1].back() : shapes[1][0];
            cldnn::mem_lock<int32_t> tgt(targets_mem, *stream);
            for (size_t i = 0; i < tgt.size(); ++i)
                tgt[i] = static_cast<int32_t>(i % (C > 1 ? C - 1 : 1));  // valid class indices, exclude blank
        }
        // input_lengths: bounded to [1, T]
        {
            int64_t T = shapes[0][0];
            cldnn::mem_lock<int32_t> il(in_len_mem, *stream);
            for (size_t i = 0; i < il.size(); ++i)
                il[i] = static_cast<int32_t>(T);  // all sequences use full length
        }
        // target_lengths: bounded to [1, S] and <= T
        {
            int64_t T = shapes[0][0];
            int64_t S = shapes[1].size() > 1 ? shapes[1].back() : shapes[1][0];
            cldnn::mem_lock<int32_t> tl(tgt_len_mem, *stream);
            for (size_t i = 0; i < tl.size(); ++i)
                tl[i] = static_cast<int32_t>(std::min(S, T));
        }

        std::vector<cldnn::input_info> inputs = {
            cldnn::input_info("log_probs"), cldnn::input_info("targets"),
            cldnn::input_info("in_len"), cldnn::input_info("tgt_len")
        };

        // Optional blank_index input
        cldnn::memory::ptr blank_mem;
        if (shapes.size() >= 5) {
            ov::PartialShape ps4(std::vector<ov::Dimension>(shapes[4].begin(), shapes[4].end()));
            cldnn::layout lay4(ps4, cldnn::data_types::i32, cldnn::format::bfyx);
            blank_mem = engine.allocate_memory(lay4);
            fill_memory_random(blank_mem, *stream, cldnn::data_types::i32);
            inputs.push_back(cldnn::input_info("blank_idx"));
        }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("log_probs", lay0));
        topology.add(cldnn::input_layout("targets", lay1));
        topology.add(cldnn::input_layout("in_len", lay2));
        topology.add(cldnn::input_layout("tgt_len", lay3));
        if (blank_mem) {
            ov::PartialShape ps4(std::vector<ov::Dimension>(shapes[4].begin(), shapes[4].end()));
            cldnn::layout lay4(ps4, cldnn::data_types::i32, cldnn::format::bfyx);
            topology.add(cldnn::input_layout("blank_idx", lay4));
        }
        topology.add(cldnn::ctc_loss("prim", inputs, false, true, false));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("log_probs", log_probs_mem);
        network.set_input_data("targets", targets_mem);
        network.set_input_data("in_len", in_len_mem);
        network.set_input_data("tgt_len", tgt_len_mem);
        if (blank_mem) network.set_input_data("blank_idx", blank_mem);

        bool test_passed = true;
        perf_timer timer;
        auto wall_start = std::chrono::high_resolution_clock::now();

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

REGISTER_KERNEL(bench_ctc_loss)

}  // namespace bench_kernel
