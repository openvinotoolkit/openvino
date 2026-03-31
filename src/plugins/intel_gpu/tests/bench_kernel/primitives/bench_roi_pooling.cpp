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
#include <intel_gpu/primitives/roi_pooling.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_roi_pooling : public kernel_base {
public:
    std::string name() const override { return "roi_pooling"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: feature_map[N,C,H,W]:rois[num_rois,5]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("roi_pooling requires 2 shapes (feature:rois). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int pooled_h = config.roi_pooled_h > 0 ? config.roi_pooled_h : 7;
        int pooled_w = config.roi_pooled_w > 0 ? config.roi_pooled_w : 7;
        float spatial_scale = config.roi_spatial_scale > 0 ? config.roi_spatial_scale : 1.0f / 16.0f;

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        auto feat_mem = engine.allocate_memory(lay0);
        auto rois_mem = engine.allocate_memory(lay1);
        fill_memory_random(feat_mem, *stream, dt);
        fill_memory_random(rois_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("feat", lay0));
        topology.add(cldnn::input_layout("rois", lay1));
        topology.add(cldnn::roi_pooling("prim",
            cldnn::input_info("feat"), cldnn::input_info("rois"),
            cldnn::pooling_mode::max, false,
            pooled_w, pooled_h, spatial_scale));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("feat", feat_mem);
        network.set_input_data("rois", rois_mem);

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

REGISTER_KERNEL(bench_roi_pooling)

}  // namespace bench_kernel
