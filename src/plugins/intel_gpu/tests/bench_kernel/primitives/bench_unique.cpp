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
#include <intel_gpu/primitives/unique.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_unique_count : public kernel_base {
public:
    std::string name() const override { return "unique_count"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("unique_count requires 1 shape. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(lay0);
        fill_memory_random(data_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", lay0));
        topology.add(cldnn::unique_count("prim", cldnn::input_info("data"),
            true /* flattened */, config.axis));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);

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

REGISTER_KERNEL(bench_unique_count)

// unique_gather requires unique_count as a dependency - create a combined bench
class bench_unique_gather : public kernel_base {
public:
    std::string name() const override { return "unique_gather"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("unique_gather requires 1 shape. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim_gather");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(lay0);
        fill_memory_random(data_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", lay0));
        topology.add(cldnn::unique_count("prim_count", cldnn::input_info("data"),
            true /* flattened */, config.axis));

        std::vector<cldnn::input_info> gather_inputs = {
            cldnn::input_info("data"),
            cldnn::input_info("prim_count")
        };
        topology.add(cldnn::unique_gather("prim_gather", gather_inputs,
            true /* flattened */, config.axis, true /* sorted */,
            dt, cldnn::data_types::i64, cldnn::data_types::i64));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);

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
        print_result(network, "prim_gather", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_unique_gather)

}  // namespace bench_kernel
