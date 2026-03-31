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
#include <intel_gpu/primitives/quantize.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Quantize (FakeQuantize) kernel benchmark
//
// Usage:
//   --quantize --dt=f16 --shapes=1x64x112x112 --levels=256
//
// FakeQuantize: clamp input to [input_low, input_high], then map to
// [output_low, output_high] with `levels` discrete values.
// input_low/high/output_low/high are per-channel scalars (shape [1,C,1,1]).
// ============================================================================

class bench_quantize : public kernel_base {
public:
    std::string name() const override { return "quantize"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Quantize requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        int levels = config.levels > 0 ? config.levels : 256;

        auto& in_shape = shapes[0];
        size_t rank = in_shape.size();
        int64_t C = (rank >= 2) ? in_shape[1] : 1;

        // Output data type: prefer user-specified, else default based on levels
        cldnn::data_types out_dt = (config.data_types.size() > 1) ? config.data_types[1]
                                 : (levels <= 256) ? cldnn::data_types::u8 : dt;

        auto exec_config = make_exec_config(config, "quantize_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, get_input_format(config, 0, rank));
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        // Per-channel limit tensors: [1, C, 1, ..., 1] matching input rank
        std::vector<int64_t> limit_dims = {1, C};
        for (size_t d = 2; d < rank; ++d) limit_dims.push_back(1);
        ov::PartialShape limit_ps(std::vector<ov::Dimension>(limit_dims.begin(), limit_dims.end()));
        cldnn::layout limit_layout(limit_ps, cldnn::data_types::f32, get_input_format(config, 1, rank));

        auto in_lo_mem = engine.allocate_memory(limit_layout);
        auto in_hi_mem = engine.allocate_memory(limit_layout);
        auto out_lo_mem = engine.allocate_memory(limit_layout);
        auto out_hi_mem = engine.allocate_memory(limit_layout);

        // Fill limits: in_lo=-1, in_hi=1, out range depends on output type
        float out_lo_val = 0.0f, out_hi_val = static_cast<float>(levels - 1);
        if (out_dt == cldnn::data_types::i8) {
            out_lo_val = -128.0f;
            out_hi_val = 127.0f;
        }
        {
            cldnn::mem_lock<float> lo(in_lo_mem, *stream);
            cldnn::mem_lock<float> hi(in_hi_mem, *stream);
            cldnn::mem_lock<float> olo(out_lo_mem, *stream);
            cldnn::mem_lock<float> ohi(out_hi_mem, *stream);
            for (int64_t c = 0; c < C; ++c) {
                lo[c] = -1.0f;
                hi[c] = 1.0f;
                olo[c] = out_lo_val;
                ohi[c] = out_hi_val;
            }
        }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::data("in_lo", in_lo_mem));
        topology.add(cldnn::data("in_hi", in_hi_mem));
        topology.add(cldnn::data("out_lo", out_lo_mem));
        topology.add(cldnn::data("out_hi", out_hi_mem));
        topology.add(cldnn::quantize("quantize_prim",
            cldnn::input_info("input"),
            cldnn::input_info("in_lo"), cldnn::input_info("in_hi"),
            cldnn::input_info("out_lo"), cldnn::input_info("out_hi"),
            levels, out_dt));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "quantize_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);

            // Reference FakeQuantize
            size_t total = input_f32.size();
            [[maybe_unused]] size_t spatial = 1;
            for (size_t d = 2; d < rank; ++d) spatial *= in_shape[d];

            std::vector<float> ref_out(total);
            for (size_t i = 0; i < total; ++i) {
                // size_t c = (rank >= 2) ? ((i / spatial) % C) : 0;
                float x = input_f32[i];
                float il = -1.0f, ih = 1.0f;
                float ol = out_lo_val, oh = out_hi_val;
                // Clamp
                x = std::min(std::max(x, il), ih);
                // Linear map
                float val = std::nearbyint((x - il) / (ih - il) * (oh - ol)) + ol;
                ref_out[i] = val;
            }

            float atol = 1.0f, rtol = 0.0f;  // Quantized values should match exactly (±1 rounding)
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
        print_result(network, "quantize_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_quantize)

}  // namespace bench_kernel
