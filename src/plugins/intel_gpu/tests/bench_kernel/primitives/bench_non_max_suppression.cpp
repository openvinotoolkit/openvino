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
#include <intel_gpu/primitives/non_max_suppression.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_non_max_suppression : public kernel_base {
public:
    std::string name() const override { return "non_max_suppression"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: boxes[N,num_boxes,4]:scores[N,num_classes,num_boxes]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("non_max_suppression requires 2 shapes (boxes:scores). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];
        int selected_indices_num = config.nms_top_k > 0 ? config.nms_top_k : 100;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        auto boxes_mem = engine.allocate_memory(lay0);
        auto scores_mem = engine.allocate_memory(lay1);
        fill_memory_random(boxes_mem, *stream, dt);
        fill_memory_random(scores_mem, *stream, dt);

        // iou/score threshold as const data
        ov::PartialShape s1({1});
        cldnn::layout scalar_lay(s1, cldnn::data_types::f32, cldnn::format::bfyx);
        auto iou_mem = engine.allocate_memory(scalar_lay);
        auto score_mem = engine.allocate_memory(scalar_lay);
        { cldnn::mem_lock<float> l(iou_mem, *stream); l[0] = config.nms_iou_threshold > 0 ? config.nms_iou_threshold : 0.5f; }
        { cldnn::mem_lock<float> l(score_mem, *stream); l[0] = config.nms_score_threshold > 0 ? config.nms_score_threshold : 0.0f; }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("boxes", lay0));
        topology.add(cldnn::input_layout("scores", lay1));
        topology.add(cldnn::data("iou_th", iou_mem));
        topology.add(cldnn::data("score_th", score_mem));
        topology.add(cldnn::non_max_suppression("prim",
            cldnn::input_info("boxes"), cldnn::input_info("scores"),
            selected_indices_num, false, true,
            "", "iou_th", "score_th"));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("boxes", boxes_mem);
        network.set_input_data("scores", scores_mem);

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

REGISTER_KERNEL(bench_non_max_suppression)

}  // namespace bench_kernel
