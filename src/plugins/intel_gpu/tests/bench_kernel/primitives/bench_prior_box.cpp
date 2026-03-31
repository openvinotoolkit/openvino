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
#include <intel_gpu/primitives/prior_box.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_prior_box : public kernel_base {
public:
    std::string name() const override { return "prior_box"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        // shapes: feat_shape:img_shape (e.g., 38x38:300x300)
        if (shapes.size() < 2) throw std::runtime_error("prior_box requires 2 shapes (feature:image). Got: " + config.shapes_str);

        [[maybe_unused]] auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];

        auto& feat_s = shapes[0];
        auto& img_s = shapes[1];

        // Default SSD-style prior box params
        std::vector<float> min_sizes = {30.0f};
        std::vector<float> max_sizes = {60.0f};
        std::vector<float> aspect_ratios = {2.0f};
        std::vector<float> variance = {0.1f, 0.1f, 0.2f, 0.2f};
        float step = 0.0f;
        float offset = 0.5f;
        bool flip = true, clip = false;

        // PriorBox expects 1D inputs: output_size[2] (H,W) and image_size[2] (H,W)
        // Extract spatial dims from full shapes
        std::vector<int64_t> feat_spatial, img_spatial;
        if (feat_s.size() >= 4) { feat_spatial = {feat_s[2], feat_s[3]}; }
        else if (feat_s.size() == 2) { feat_spatial = feat_s; }
        else { feat_spatial = feat_s; }
        if (img_s.size() >= 4) { img_spatial = {img_s[2], img_s[3]}; }
        else if (img_s.size() == 2) { img_spatial = img_s; }
        else { img_spatial = img_s; }

        std::vector<int64_t> feat4 = feat_s; feat4.resize(4, 1);
        std::vector<int64_t> img4 = img_s; img4.resize(4, 1);
        cldnn::tensor out_t(cldnn::batch(feat4[0]), cldnn::feature(feat4[1]),
                            cldnn::spatial(feat4[3], feat4[2]));
        cldnn::tensor img_t(cldnn::batch(img4[0]), cldnn::feature(img4[1]),
                            cldnn::spatial(img4[3], img4[2]));

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        // Both inputs must be 1D tensors of shape [ndim] containing spatial dimensions
        ov::PartialShape ps_feat({static_cast<int64_t>(feat_spatial.size())});
        cldnn::layout feat_layout(ps_feat, cldnn::data_types::i64, cldnn::format::bfyx);
        auto feat_mem = engine.allocate_memory(feat_layout);
        { cldnn::mem_lock<int64_t> l(feat_mem, *stream); for (size_t i = 0; i < feat_spatial.size(); ++i) l[i] = feat_spatial[i]; }

        ov::PartialShape ps_img({static_cast<int64_t>(img_spatial.size())});
        cldnn::layout img_layout(ps_img, cldnn::data_types::i64, cldnn::format::bfyx);
        auto img_mem = engine.allocate_memory(img_layout);
        { cldnn::mem_lock<int64_t> l(img_mem, *stream); for (size_t i = 0; i < img_spatial.size(); ++i) l[i] = img_spatial[i]; }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("feat", feat_layout));
        topology.add(cldnn::input_layout("img", img_layout));
        topology.add(cldnn::prior_box("prim",
            {cldnn::input_info("feat"), cldnn::input_info("img")}, out_t, img_t,
            min_sizes, max_sizes, aspect_ratios,
            flip, clip, variance, step, offset));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("feat", feat_mem);
        network.set_input_data("img", img_mem);

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

REGISTER_KERNEL(bench_prior_box)

}  // namespace bench_kernel
