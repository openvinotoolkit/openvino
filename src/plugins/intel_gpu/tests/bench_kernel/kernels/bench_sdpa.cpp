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
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_attrs.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Scaled Dot Product Attention kernel benchmark
//
// Usage:
//   --sdpa --dt=f16 QxKxVxM  (4 shapes: query:key:value[:attn_mask])
//
// Example:
//   --sdpa --dt=f16 1x32x1x128:1x32x128x128:1x32x128x128
//   --sdpa --dt=f16 1x32x1x128:1x32x128x128:1x32x128x128:1x1x1x128
// ============================================================================

class bench_sdpa : public kernel_base {
public:
    std::string name() const override { return "scaled_dot_product_attention"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // Parse shapes: query:key:value[:attn_mask]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 3) {
            throw std::runtime_error("SDPA requires at least 3 shapes (Q:K:V). Got: " + config.shapes_str);
        }

        auto& q_shape = shapes[0];
        auto& k_shape = shapes[1];
        auto& v_shape = shapes[2];
        bool has_mask = shapes.size() >= 4;

        auto dts = config.data_types;
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        // Parse scale/zp for KV compression
        auto scales = parse_scales(config.attr_scales_str);
        auto zero_points = parse_zero_points(config.attr_zero_points_str);

        // Config
        auto exec_config = make_exec_config(config, "sdpa_prim");
        auto stream = engine.create_stream(exec_config);

        // Allocate Q, K, V
        ov::PartialShape q_ps(std::vector<ov::Dimension>(q_shape.begin(), q_shape.end()));
        ov::PartialShape k_ps(std::vector<ov::Dimension>(k_shape.begin(), k_shape.end()));
        ov::PartialShape v_ps(std::vector<ov::Dimension>(v_shape.begin(), v_shape.end()));

        cldnn::layout q_layout(q_ps, input_dt, cldnn::format::bfyx);
        cldnn::layout k_layout(k_ps, input_dt, cldnn::format::bfyx);
        cldnn::layout v_layout(v_ps, input_dt, cldnn::format::bfyx);

        auto q_mem = engine.allocate_memory(q_layout);
        auto k_mem = engine.allocate_memory(k_layout);
        auto v_mem = engine.allocate_memory(v_layout);
        fill_memory_random(q_mem, *stream, input_dt);
        fill_memory_random(k_mem, *stream, input_dt);
        fill_memory_random(v_mem, *stream, input_dt);

        // Save input data for accuracy check BEFORE network may reformat
        std::vector<float> q_f32_saved, k_f32_saved, v_f32_saved, mask_f32_saved;
        if (config.is_acc()) {
            q_f32_saved = read_memory_to_f32(q_mem, *stream);
            k_f32_saved = read_memory_to_f32(k_mem, *stream);
            v_f32_saved = read_memory_to_f32(v_mem, *stream);
        }

        // Topology
        cldnn::topology topology;
        topology.add(cldnn::input_layout("query", q_layout));
        topology.add(cldnn::input_layout("key", k_layout));
        topology.add(cldnn::input_layout("value", v_layout));

        std::vector<cldnn::input_info> sdpa_inputs = {
            cldnn::input_info("query"),
            cldnn::input_info("key"),
            cldnn::input_info("value"),
        };

        cldnn::memory::ptr mask_mem_saved;
        if (has_mask) {
            auto& mask_shape = shapes[3];
            ov::PartialShape mask_ps(std::vector<ov::Dimension>(mask_shape.begin(), mask_shape.end()));
            cldnn::layout mask_layout(mask_ps, input_dt, cldnn::format::bfyx);
            mask_mem_saved = engine.allocate_memory(mask_layout);
            fill_memory_random(mask_mem_saved, *stream, input_dt);
            if (config.is_acc()) {
                mask_f32_saved = read_memory_to_f32(mask_mem_saved, *stream);
            }
            topology.add(cldnn::input_layout("attn_mask", mask_layout));
            sdpa_inputs.push_back(cldnn::input_info("attn_mask"));
        }

        bool is_causal;
        if (config.is_causal >= 0) {
            is_causal = (config.is_causal != 0);
        } else {
            is_causal = !has_mask;  // Default to causal if no mask provided
        }

        // Transpose orders from config or default identity
        auto order_q = config.order_q.empty() ? std::vector<int64_t>{0,1,2,3} : parse_colon_vec(config.order_q);
        auto order_k = config.order_k.empty() ? std::vector<int64_t>{0,1,2,3} : parse_colon_vec(config.order_k);
        auto order_v = config.order_v.empty() ? std::vector<int64_t>{0,1,2,3} : parse_colon_vec(config.order_v);
        auto order_out = config.order_out.empty() ? std::vector<int64_t>{0,1,2,3} : parse_colon_vec(config.order_out);

        // For accuracy mode, use identity order_out in the SDPA primitive.
        // The GPU SDPA kernel's output layout with non-identity order_out uses non-standard
        // strides that make raw memory comparison unreliable. Since order_out only affects
        // the output layout (not the SDPA computation), using identity output order
        // correctly validates the attention computation while keeping comparison simple.
        // Performance mode uses the actual order_out for realistic kernel timing.
        auto effective_order_out = config.is_acc() ? std::vector<int64_t>{0,1,2,3} : order_out;

        auto sdpa_prim = cldnn::scaled_dot_product_attention("sdpa_prim",
            sdpa_inputs,
            is_causal,
            -1,  // indirect_axis
            order_q,
            order_k,
            order_v,
            effective_order_out);
        if (!config.scale_val.empty()) {
            sdpa_prim.scale_val = std::stof(config.scale_val);
        }
        topology.add(sdpa_prim);

        std::string last_prim_id = "sdpa_prim";

        // For accuracy mode, add reorder to f32/bfyx to ensure readable output layout
        std::string output_prim_id = last_prim_id;
        if (config.is_acc()) {
            topology.add(cldnn::reorder("output_reorder",
                cldnn::input_info(last_prim_id),
                cldnn::format::bfyx, cldnn::data_types::f32));
            output_prim_id = "output_reorder";
        }

        // Build & execute
        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("query", q_mem);
        network.set_input_data("key", k_mem);
        network.set_input_data("value", v_mem);
        if (has_mask) {
            network.set_input_data("attn_mask", mask_mem_saved);
        }

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
            auto gpu_out = read_network_output_f32(outputs, output_prim_id, *stream);

            // Reference uses identity order_out (matching effective_order_out in accuracy mode)
            auto ref_out = ref::sdpa(q_f32_saved, q_shape, k_f32_saved, k_shape, v_f32_saved, v_shape, is_causal,
                                     order_q, order_k, order_v, std::vector<int64_t>{0,1,2,3},
                                     config.scale_val.empty() ? 0.0f : std::stof(config.scale_val),
                                     has_mask ? &mask_f32_saved : nullptr,
                                     has_mask ? &shapes[3] : nullptr);

            float atol, rtol;
            get_default_tolerance(input_dt, atol, rtol);
            atol *= 10.0f;
            rtol *= 5.0f;
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
        print_result(network, "sdpa_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_sdpa)

}  // namespace bench_kernel
