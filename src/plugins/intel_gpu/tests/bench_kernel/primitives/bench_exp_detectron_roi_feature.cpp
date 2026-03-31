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
#include <intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_exp_detectron_roi_feat : public kernel_base {
public:
    std::string name() const override { return "experimental_detectron_roi_feature_extractor"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: rois[N,4] : feat_map1[C,H,W] [: feat_map2 ...]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("exp_detectron_roi_feature requires >=2 shapes (rois:feat_maps). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        std::vector<cldnn::input_info> inputs;
        std::vector<cldnn::memory::ptr> memories;

        cldnn::topology topology;

        for (size_t i = 0; i < shapes.size(); ++i) {
            std::string name = (i == 0) ? "rois" : "feat_" + std::to_string(i - 1);
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[i].begin(), shapes[i].end()));
            cldnn::layout lay(ps, dt, cldnn::format::bfyx);
            auto mem = engine.allocate_memory(lay);
            fill_memory_random(mem, *stream, dt);
            topology.add(cldnn::input_layout(name, lay));
            inputs.push_back(cldnn::input_info(name));
            memories.push_back(mem);
        }

        int output_dim = config.roi_pooled_h > 0 ? config.roi_pooled_h : 7;
        std::vector<int64_t> pyramid_scales;
        for (size_t i = 1; i < shapes.size(); ++i) pyramid_scales.push_back(4 * (1 << (i - 1)));
        int sampling_ratio = config.roi_sampling_ratio > 0 ? config.roi_sampling_ratio : 2;

        topology.add(cldnn::experimental_detectron_roi_feature_extractor("prim",
            inputs, output_dim, pyramid_scales, sampling_ratio, false));

        cldnn::network network(engine, topology, exec_config);
        for (size_t i = 0; i < shapes.size(); ++i) {
            std::string name = (i == 0) ? "rois" : "feat_" + std::to_string(i - 1);
            network.set_input_data(name, memories[i]);
        }

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

REGISTER_KERNEL(bench_exp_detectron_roi_feat)

}  // namespace bench_kernel
