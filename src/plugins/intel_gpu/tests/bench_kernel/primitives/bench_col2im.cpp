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
#include <intel_gpu/primitives/col2im.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Col2Im kernel benchmark
//
// Usage:
//   --col2im --dt=f16 --shapes=1x27x16
//       --col2im_output_shape=4x4 --col2im_kernel_shape=3x3
//       --strides=1x1 --dilations=1x1
//
// shapes: input [N, C*KH*KW, L] where L = number of sliding blocks
// col2im_output_shape: spatial output shape (H:W), colon-separated
// col2im_kernel_shape: kernel size (KH:KW), colon-separated
// ============================================================================

class bench_col2im : public kernel_base {
public:
    std::string name() const override { return "col2im"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Col2Im requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        auto& in_shape = shapes[0];

        // Parse col2im params
        auto out_shape_vec = parse_x_vec(config.col2im_output_shape);
        auto kernel_shape_vec = parse_x_vec(config.col2im_kernel_shape);
        auto strides_vec = parse_x_vec(config.strides);
        auto dilations_vec = parse_x_vec(config.dilations);
        auto pad_begin_vec = parse_x_vec(config.padding_begin);
        auto pad_end_vec = parse_x_vec(config.padding_end);

        // Defaults
        if (out_shape_vec.empty()) out_shape_vec = {4, 4};
        if (kernel_shape_vec.empty()) kernel_shape_vec = {3, 3};
        if (strides_vec.empty()) strides_vec = {1, 1};
        if (dilations_vec.empty()) dilations_vec = {1, 1};
        if (pad_begin_vec.empty()) pad_begin_vec = {0, 0};
        if (pad_end_vec.empty()) pad_end_vec = {0, 0};

        ov::Shape output_shape(out_shape_vec.begin(), out_shape_vec.end());
        ov::Shape kernel_shape(kernel_shape_vec.begin(), kernel_shape_vec.end());
        ov::Strides str(strides_vec.begin(), strides_vec.end());
        ov::Strides dil(dilations_vec.begin(), dilations_vec.end());
        ov::CoordinateDiff pb(pad_begin_vec.begin(), pad_begin_vec.end());
        ov::CoordinateDiff pe(pad_end_vec.begin(), pad_end_vec.end());

        auto exec_config = make_exec_config(config, "col2im_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::col2im("col2im_prim",
            cldnn::input_info("input"),
            str, dil, pb, pe,
            output_shape, kernel_shape));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "col2im_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            std::vector<int64_t> osz(out_shape_vec.begin(), out_shape_vec.end());
            std::vector<int64_t> ksz(kernel_shape_vec.begin(), kernel_shape_vec.end());
            std::vector<int64_t> str(strides_vec.begin(), strides_vec.end());
            std::vector<int64_t> dil(dilations_vec.begin(), dilations_vec.end());
            std::vector<int64_t> pad(pad_begin_vec.begin(), pad_begin_vec.end());
            auto ref_out = ref::col2im(input_f32, in_shape, osz, ksz, str, dil, pad);
            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
            reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
            if (!acc_res.pass) test_passed = false;
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "col2im_prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_col2im)

}  // namespace bench_kernel
