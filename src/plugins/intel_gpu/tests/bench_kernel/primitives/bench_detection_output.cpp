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
#include <intel_gpu/primitives/detection_output.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// DetectionOutput kernel benchmark
//
// Usage:
//   --detection_output --dt=f32 --shapes=1x7668:1x2091:1x2x7668
//       --det_num_classes=21 --det_keep_top_k=200 --det_nms_threshold=0.45
//       --det_confidence_threshold=0.01 --det_top_k=400
//       --det_code_type=1  (0=corner, 1=center_size, 2=corner_size)
//       --det_share_location=1
//
// shapes: location_pred : confidence_pred : prior_boxes
// ============================================================================

class bench_detection_output : public kernel_base {
public:
    std::string name() const override { return "detection_output"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 3) {
            throw std::runtime_error("DetectionOutput requires 3 shapes (loc:conf:priors). Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f32 : config.data_types[0];

        // Trim trailing 1s from shapes (verbose log may produce 4D shapes, but
        // DetectionOutput requires 2D for loc/conf and 3D for priors)
        auto trim_trailing_ones = [](std::vector<int64_t>& shape, size_t min_rank) {
            while (shape.size() > min_rank && shape.back() == 1) shape.pop_back();
        };
        auto loc_shape = shapes[0];     // [N, num_loc_classes * 4]
        auto conf_shape = shapes[1];    // [N, num_classes * num_priors]
        auto prior_shape = shapes[2];   // [1, 2, num_priors * prior_info_size]
        trim_trailing_ones(loc_shape, 2);
        trim_trailing_ones(conf_shape, 2);
        trim_trailing_ones(prior_shape, 3);

        auto exec_config = make_exec_config(config, "det_output_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape loc_ps(std::vector<ov::Dimension>(loc_shape.begin(), loc_shape.end()));
        ov::PartialShape conf_ps(std::vector<ov::Dimension>(conf_shape.begin(), conf_shape.end()));
        ov::PartialShape prior_ps(std::vector<ov::Dimension>(prior_shape.begin(), prior_shape.end()));

        cldnn::layout loc_layout(loc_ps, dt, cldnn::format::bfyx);
        cldnn::layout conf_layout(conf_ps, dt, cldnn::format::bfyx);
        cldnn::layout prior_layout(prior_ps, dt, cldnn::format::bfyx);

        auto loc_mem = engine.allocate_memory(loc_layout);
        auto conf_mem = engine.allocate_memory(conf_layout);
        auto prior_mem = engine.allocate_memory(prior_layout);

        fill_memory_random(loc_mem, *stream, dt);
        fill_memory_random(conf_mem, *stream, dt);
        fill_memory_random(prior_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("loc", loc_layout));
        topology.add(cldnn::input_layout("conf", conf_layout));
        topology.add(cldnn::input_layout("priors", prior_layout));

        // Parse detection output params
        auto code_type = cldnn::prior_box_code_type::center_size;
        if (config.det_code_type == 0) code_type = cldnn::prior_box_code_type::corner;
        else if (config.det_code_type == 2) code_type = cldnn::prior_box_code_type::corner_size;

        // Infer variance_encoded_in_target from priors shape when not explicitly set.
        // Prior shape is [N, 1, data] when variance is encoded in target (no separate variance),
        // and [N, 2, data] when variance is separate (dim[1]=2 means [priors, variances]).
        bool variance_encoded = (config.det_variance_encoded != 0);
        if (config.det_variance_encoded == 0 && prior_shape.size() >= 2 && prior_shape[1] == 1) {
            variance_encoded = true;
        }

        topology.add(cldnn::detection_output("det_output_prim",
            {cldnn::input_info("loc"), cldnn::input_info("conf"), cldnn::input_info("priors")},
            config.det_num_classes,
            static_cast<uint32_t>(config.det_keep_top_k),
            config.det_share_location != 0,
            config.det_background_label_id,
            config.det_nms_threshold,
            config.det_top_k,
            1.0f,  // eta
            code_type,
            variance_encoded,
            config.det_confidence_threshold));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("loc", loc_mem);
        network.set_input_data("conf", conf_mem);
        network.set_input_data("priors", prior_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "det_output_prim", *stream);
            if (gpu_out.empty()) test_passed = false;
            else {
                // DetectionOutput is NMS-based — verify output values are in reasonable range
                auto ref_out = ref::make_range_check_ref(gpu_out, -1.0f, 2.0f);
                float atol = 2.0f, rtol = 1.0f;
                acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
                reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
            }
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "det_output_prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_detection_output)

}  // namespace bench_kernel
