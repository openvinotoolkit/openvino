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
#include <intel_gpu/primitives/proposal.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_proposal : public kernel_base {
public:
    std::string name() const override { return "proposal"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: cls_scores[N,2*A,H,W]:bbox_pred[N,4*A,H,W]:img_info[N,3or4]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 3) throw std::runtime_error("proposal requires 3 shapes (cls_scores:bbox_pred:img_info). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];
        int max_proposals = 300;
        float iou_th = 0.7f;
        int min_bbox = 16, feat_stride = 16, pre_nms = 6000, post_nms = 300;
        std::vector<float> ratios = {0.5f, 1.0f, 2.0f};
        std::vector<float> scales = {8.0f, 16.0f, 32.0f};

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        auto make_layout = [&](const std::vector<int64_t>& s, cldnn::data_types t) {
            ov::PartialShape ps(std::vector<ov::Dimension>(s.begin(), s.end()));
            return cldnn::layout(ps, t, cldnn::format::bfyx);
        };

        auto lay0 = make_layout(shapes[0], dt);
        auto lay1 = make_layout(shapes[1], dt);
        auto lay2 = make_layout(shapes[2], dt);
        auto mem0 = engine.allocate_memory(lay0);
        auto mem1 = engine.allocate_memory(lay1);
        auto mem2 = engine.allocate_memory(lay2);
        fill_memory_random(mem0, *stream, dt);
        fill_memory_random(mem1, *stream, dt);
        fill_memory_random(mem2, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("cls", lay0));
        topology.add(cldnn::input_layout("bbox", lay1));
        topology.add(cldnn::input_layout("img", lay2));
        topology.add(cldnn::proposal("prim",
            cldnn::input_info("cls"), cldnn::input_info("bbox"), cldnn::input_info("img"),
            max_proposals, iou_th, min_bbox, feat_stride,
            pre_nms, post_nms, ratios, scales));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("cls", mem0);
        network.set_input_data("bbox", mem1);
        network.set_input_data("img", mem2);

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

REGISTER_KERNEL(bench_proposal)

}  // namespace bench_kernel
