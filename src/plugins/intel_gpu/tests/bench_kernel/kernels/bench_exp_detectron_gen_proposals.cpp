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
#include <intel_gpu/primitives/experimental_detectron_generate_proposals_single_image.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_exp_detectron_gen_proposals : public kernel_base {
public:
    std::string name() const override { return "experimental_detectron_generate_proposals_single_image"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: im_info[3] : anchors[A,4] : deltas[A,4] : scores[A]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 4) throw std::runtime_error("exp_detectron_gen_proposals requires 4 shapes. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        auto make_layout = [&](size_t idx, cldnn::data_types d) {
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[idx].begin(), shapes[idx].end()));
            return cldnn::layout(ps, d, cldnn::format::bfyx);
        };

        auto lay0 = make_layout(0, dt);
        auto lay1 = make_layout(1, dt);
        auto lay2 = make_layout(2, dt);
        auto lay3 = make_layout(3, dt);

        auto mem0 = engine.allocate_memory(lay0);
        auto mem1 = engine.allocate_memory(lay1);
        auto mem2 = engine.allocate_memory(lay2);
        auto mem3 = engine.allocate_memory(lay3);
        fill_memory_random(mem0, *stream, dt);
        fill_memory_random(mem1, *stream, dt);
        fill_memory_random(mem2, *stream, dt);
        fill_memory_random(mem3, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("im_info", lay0));
        topology.add(cldnn::input_layout("anchors", lay1));
        topology.add(cldnn::input_layout("deltas", lay2));
        topology.add(cldnn::input_layout("scores", lay3));
        topology.add(cldnn::experimental_detectron_generate_proposals_single_image("prim",
            cldnn::input_info("im_info"),
            cldnn::input_info("anchors"),
            cldnn::input_info("deltas"),
            cldnn::input_info("scores"),
            0.0f,   // min_size
            config.nms_iou_threshold,  // nms_threshold
            config.nms_top_k,          // pre_nms_count
            config.nms_top_k / 2));    // post_nms_count

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("im_info", mem0);
        network.set_input_data("anchors", mem1);
        network.set_input_data("deltas", mem2);
        network.set_input_data("scores", mem3);

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

REGISTER_KERNEL(bench_exp_detectron_gen_proposals)

}  // namespace bench_kernel
