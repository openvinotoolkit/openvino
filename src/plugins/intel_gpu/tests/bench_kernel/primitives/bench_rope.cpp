// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/rope.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// RoPE (Rotary Position Embedding) kernel benchmark
//
// Usage:
//   --rope --dt=f16 --shapes=BxSxHxD:1x1xSxD:1x1xSxD
//
// Standard LLM RoPE: input[B,S,H,D] + cos[1,1,S,D] + sin[1,1,S,D]
//   -> output[B,H,S,D] (with output_trans0213)
//
// RotateHalf formula (non-interleaved):
//   For each (b,s,h) and r in [0, half_rotary_ndims):
//     out[r]      = cos[s,r]*in[r]      - sin[s,r]*in[r+half]
//     out[r+half] = cos[s,r+half]*in[r+half] + sin[s,r+half]*in[r]
//
// Example:
//   --rope --dt=f16 --shapes=1x128x32x128:1x1x128x128:1x1x128x128
//   --rope --dt=f16 --shapes=1x128x8x128:1x1x128x128:1x1x128x128
// ============================================================================

class bench_rope : public kernel_base {
public:
    std::string name() const override { return "rope"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 3) {
            throw std::runtime_error("RoPE requires 3 shapes (input:cos:sin). Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "rope_prim");
        auto stream = engine.create_stream(exec_config);

        auto& input_shape = shapes[0];   // [B, S, H, D]
        auto& cos_shape = shapes[1];     // [1, 1, S, D]
        auto& sin_shape = shapes[2];     // [1, 1, S, D]

        if (input_shape.size() != 4) {
            throw std::runtime_error("RoPE input must be 4D [B,S,H,D]. Got rank=" + std::to_string(input_shape.size()));
        }

        int64_t head_cnt = config.head_cnt > 0 ? config.head_cnt : input_shape[2];
        int64_t head_size = config.head_size > 0 ? config.head_size : input_shape[3];
        int64_t rotary_ndims = config.rotary_ndims > 0 ? config.rotary_ndims : head_size;

        // Allocate input memory
        ov::PartialShape input_ps(std::vector<ov::Dimension>(input_shape.begin(), input_shape.end()));
        cldnn::layout input_layout_desc(input_ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout_desc);
        fill_memory_random(input_mem, *stream, dt);

        // Cos table
        ov::PartialShape cos_ps(std::vector<ov::Dimension>(cos_shape.begin(), cos_shape.end()));
        cldnn::layout cos_layout(cos_ps, dt, cldnn::format::bfyx);
        auto cos_mem = engine.allocate_memory(cos_layout);
        fill_memory_random(cos_mem, *stream, dt);

        // Sin table
        ov::PartialShape sin_ps(std::vector<ov::Dimension>(sin_shape.begin(), sin_shape.end()));
        cldnn::layout sin_layout(sin_ps, dt, cldnn::format::bfyx);
        auto sin_mem = engine.allocate_memory(sin_layout);
        fill_memory_random(sin_mem, *stream, dt);

        // RoPE config: standard LLama-style RotateHalf with input transpose
        // LLama/Mistral/Qwen: input [B,S,H,D] -> output [B,H,S,D]
        // input_trans0213 = true means ENABLE_TRANSPOSE in kernel,
        //   which reads input as [B,S,H,D] and writes output as [B,H,S,D]
        cldnn::RoPE::Config rope_config;
        rope_config.rotary_ndims = static_cast<size_t>(rotary_ndims);
        rope_config.head_cnt = static_cast<size_t>(head_cnt);
        rope_config.head_size = static_cast<size_t>(head_size);
        rope_config.input_trans0213 = (config.input_trans0213 != 0);
        rope_config.output_trans0213 = false;
        rope_config.is_interleaved = (config.is_interleaved != 0);
        rope_config.is_chatglm = (config.is_chatglm != 0);
        rope_config.is_qwen = (config.is_qwen != 0);
        rope_config.support_2d_rope = false;
        rope_config.support_3d_rope = false;
        rope_config.use_rope_cache = false;
        rope_config.is_ltx_video = false;
        rope_config.slice_start = static_cast<size_t>(config.slice_start);
        rope_config.slice_stop = static_cast<size_t>(config.slice_stop);
        rope_config.gather_position_arg_id = config.gather_rank;

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout_desc));
        topology.add(cldnn::input_layout("cos", cos_layout));
        topology.add(cldnn::input_layout("sin", sin_layout));
        topology.add(cldnn::rope("rope_prim",
            {cldnn::input_info("input"), cldnn::input_info("cos"), cldnn::input_info("sin")},
            rope_config));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);
        network.set_input_data("cos", cos_mem);
        network.set_input_data("sin", sin_mem);

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
            auto gpu_out = read_network_output_f32(outputs, "rope_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto cos_f32 = read_memory_to_f32(cos_mem, *stream);
            auto sin_f32 = read_memory_to_f32(sin_mem, *stream);
            auto ref_out = ref::rope_rotate_half(input_f32, cos_f32, sin_f32,
                                                  input_shape, cos_shape,
                                                  static_cast<int64_t>(rotary_ndims),
                                                  true /*output_trans0213*/);

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
        print_result(network, "rope_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_rope)

}  // namespace bench_kernel
