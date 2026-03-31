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
#include <intel_gpu/primitives/experimental_detectron_detection_output.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_exp_detectron_det_output : public kernel_base {
public:
    std::string name() const override { return "experimental_detectron_detection_output"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: rois[N,4] : deltas[N,num_classes*4] : scores[N,num_classes] : im_info[1,3]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 4) throw std::runtime_error("exp_detectron_detection_output requires 4 shapes. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        auto make_layout = [&](size_t idx, cldnn::data_types d) {
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[idx].begin(), shapes[idx].end()));
            return cldnn::layout(ps, d, cldnn::format::bfyx);
        };

        auto lay0 = make_layout(0, dt); // rois
        auto lay1 = make_layout(1, dt); // deltas
        auto lay2 = make_layout(2, dt); // scores
        auto lay3 = make_layout(3, dt); // im_info

        auto mem0 = engine.allocate_memory(lay0);
        auto mem1 = engine.allocate_memory(lay1);
        auto mem2 = engine.allocate_memory(lay2);
        auto mem3 = engine.allocate_memory(lay3);
        fill_memory_random(mem0, *stream, dt);
        fill_memory_random(mem1, *stream, dt);
        fill_memory_random(mem2, *stream, dt);
        fill_memory_random(mem3, *stream, dt);

        int num_classes = config.det_num_classes > 0 ? config.det_num_classes : 2;

        cldnn::topology topology;
        topology.add(cldnn::input_layout("rois", lay0));
        topology.add(cldnn::input_layout("deltas", lay1));
        topology.add(cldnn::input_layout("scores", lay2));
        topology.add(cldnn::input_layout("im_info", lay3));
        topology.add(cldnn::experimental_detectron_detection_output("prim",
            cldnn::input_info("rois"),
            cldnn::input_info("deltas"),
            cldnn::input_info("scores"),
            cldnn::input_info("im_info"),
            cldnn::input_info(""),  // output_classes (empty)
            cldnn::input_info(""),  // output_scores (empty)
            config.nms_score_threshold, // score_threshold
            config.nms_iou_threshold,   // nms_threshold
            num_classes,                // num_classes
            config.nms_top_k,           // post_nms_count
            config.nms_top_k * 2,       // max_detections_per_image
            true,                       // class_agnostic_box_regression
            4.135f,                     // max_delta_log_wh
            std::vector<float>{10.0f, 10.0f, 5.0f, 5.0f}));  // deltas_weights

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("rois", mem0);
        network.set_input_data("deltas", mem1);
        network.set_input_data("scores", mem2);
        network.set_input_data("im_info", mem3);

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

REGISTER_KERNEL(bench_exp_detectron_det_output)

}  // namespace bench_kernel
