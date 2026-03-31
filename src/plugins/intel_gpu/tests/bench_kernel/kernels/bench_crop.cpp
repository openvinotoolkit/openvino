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
#include <intel_gpu/primitives/crop.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Crop (Slice/StridedSlice) kernel benchmark
//
// Usage:
//   --crop --dt=f16 input_shape:crop_shape
//
// Crops (slices) from offset (0,0,...) with size crop_shape
//
// Example:
//   --crop --dt=f16 1x128x8192:1x128x4096   (slice last dim in half)
//   --crop --dt=f16 1x32x128:1x32x64
// ============================================================================

class bench_crop : public kernel_base {
public:
    std::string name() const override { return "crop"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Crop requires 2 shapes (input:crop_size). Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "crop_prim");
        auto stream = engine.create_stream(exec_config);

        auto& input_shape = shapes[0];
        auto& crop_shape = shapes[1];

        // Use original dimensionality with ov::PartialShape (no 4D padding needed)
        ov::PartialShape input_ps(std::vector<ov::Dimension>(input_shape.begin(), input_shape.end()));
        cldnn::layout input_layout_desc(input_ps, dt, get_input_format(config, 0, input_shape.size()));
        auto input_mem = engine.allocate_memory(input_layout_desc);
        fill_memory_random(input_mem, *stream, dt);

        // Convert shape to cldnn::tensor matching tensor_from_dims logic
        auto shape_to_tensor = [](const std::vector<int64_t>& dims, int def = 1) -> cldnn::tensor {
            switch (dims.size()) {
            case 1: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(def, def));
            case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, def));
            case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, dims[2]));
            case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
            case 5: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[4], dims[3], dims[2]));
            case 6: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[5], dims[4], dims[3], dims[2]));
            default: throw std::runtime_error("Unsupported dims size: " + std::to_string(dims.size()));
            }
        };

        cldnn::tensor ref_tensor = shape_to_tensor(crop_shape);
        // Parse offsets from config (from verbose log) or default to zeros
        std::vector<int64_t> offset_vec(crop_shape.size(), 0);
        if (!config.offsets.empty()) {
            offset_vec = parse_colon_vec(config.offsets);
            offset_vec.resize(crop_shape.size(), 0);
        }
        cldnn::tensor off_tensor = shape_to_tensor(offset_vec, 0);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout_desc));
        topology.add(cldnn::crop("crop_prim",
            cldnn::input_info("input"),
            ref_tensor,
            off_tensor));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        // Execute and collect results
        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        // Accuracy mode
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "crop_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto ref_out = ref::crop(input_f32, input_shape, crop_shape, offset_vec);

            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
            has_acc = true;
            reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
            if (!acc_res.pass) test_passed = false;
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            has_perf = true;
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "crop_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_crop)

}  // namespace bench_kernel
