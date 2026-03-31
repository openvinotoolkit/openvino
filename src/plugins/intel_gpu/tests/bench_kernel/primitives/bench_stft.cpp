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
#include <intel_gpu/primitives/stft.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_stft : public kernel_base {
public:
    std::string name() const override { return "STFT"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: signal : window  (frame_size and frame_step set automatically from window size)
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("stft requires 2 shapes (signal:window). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        // frame_size and frame_step must be scalars (rank 0)
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        cldnn::layout lay2(ov::PartialShape{}, cldnn::data_types::i64, cldnn::format::bfyx);
        cldnn::layout lay3(ov::PartialShape{}, cldnn::data_types::i64, cldnn::format::bfyx);

        auto signal_mem = engine.allocate_memory(lay0);
        auto window_mem = engine.allocate_memory(lay1);
        auto fsize_mem  = engine.allocate_memory(lay2);
        auto fstep_mem  = engine.allocate_memory(lay3);
        fill_memory_random(signal_mem, *stream, dt);
        fill_memory_random(window_mem, *stream, dt);
        // Set frame_size to window size, frame_step to 1
        { cldnn::mem_lock<int64_t> l(fsize_mem, *stream); l[0] = static_cast<int64_t>(shapes[1][0]); }
        { cldnn::mem_lock<int64_t> l(fstep_mem, *stream); l[0] = std::max(int64_t(1), static_cast<int64_t>(shapes[1][0]) / 2); }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("signal", lay0));
        topology.add(cldnn::input_layout("window", lay1));
        topology.add(cldnn::input_layout("frame_size", lay2));
        topology.add(cldnn::input_layout("frame_step", lay3));
        topology.add(cldnn::STFT("prim",
            cldnn::input_info("signal"), cldnn::input_info("window"),
            cldnn::input_info("frame_size"), cldnn::input_info("frame_step"),
            false /* transpose_frames */));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("signal", signal_mem);
        network.set_input_data("window", window_mem);
        network.set_input_data("frame_size", fsize_mem);
        network.set_input_data("frame_step", fstep_mem);

        bool test_passed = true;
        perf_timer timer;
        auto wall_start = std::chrono::high_resolution_clock::now();

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

REGISTER_KERNEL(bench_stft)

}  // namespace bench_kernel
