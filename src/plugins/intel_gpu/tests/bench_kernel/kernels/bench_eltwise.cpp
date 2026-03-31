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
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/activation.hpp>

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
// Eltwise kernel benchmark
//
// Usage:
//   --eltwise --dt=f16 --attr-post-ops=sum shape0:shape1
//
// Example:
//   --eltwise --dt=f16 --attr-post-ops=sum 1x64x56x56:1x64x56x56
//   --eltwise --dt=f16 --attr-post-ops=prod 1x4096:1x4096
// ============================================================================

class bench_eltwise : public kernel_base {
public:
    std::string name() const override { return "eltwise"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Eltwise requires at least 2 shapes. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        bool int_output_dt = (dt == cldnn::data_types::i8 || dt == cldnn::data_types::u8 ||
                      dt == cldnn::data_types::i32 || dt == cldnn::data_types::i64);

        // Determine eltwise mode: config.eltwise_mode takes priority (direct from verbose log),
        // then fall back to post-ops string, then default to sum
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        bool has_quantize_post_op = false;
        for (const auto& po : post_ops) {
            if (po.kind == post_op_kind::quantize) {
                has_quantize_post_op = true;
                break;
            }
        }
        bool skip_activation_chain = int_output_dt && has_quantize_post_op;
        cldnn::eltwise_mode mode = cldnn::eltwise_mode::sum;
        if (config.eltwise_mode >= 0) {
            mode = static_cast<cldnn::eltwise_mode>(config.eltwise_mode);
        } else if (!post_ops.empty() && post_ops[0].kind == post_op_kind::eltwise) {
            mode = map_eltwise_mode(post_ops[0].elt_mode);
        }

        auto exec_config = make_exec_config(config, "eltwise_prim");
        auto stream = engine.create_stream(exec_config);

        cldnn::topology topology;
        std::vector<cldnn::input_info> inputs;
        std::vector<cldnn::memory::ptr> memories;

        for (size_t i = 0; i < shapes.size(); ++i) {
            std::string id = "input" + std::to_string(i);
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[i].begin(), shapes[i].end()));
            // Use per-input data type: dt[0] for input0, dt[1] for input1 (if available)
            cldnn::data_types input_dt = (dts.size() > i) ? dts[i] : dt;
            cldnn::layout layout(ps, input_dt, get_input_format(config, i, shapes[i].size()));
            auto mem = engine.allocate_memory(layout);
            fill_memory_random(mem, *stream, input_dt);
            topology.add(cldnn::input_layout(id, layout));
            inputs.push_back(cldnn::input_info(id));
            memories.push_back(mem);
        }

        topology.add(cldnn::eltwise("eltwise_prim", inputs, mode,
            {}, dt,
            ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY),
            config.pythondiv != 0));
        std::string last_prim_id = "eltwise_prim";

        // Determine which post-ops to apply after the eltwise primitive.
        // If eltwise_mode comes from post_ops[0], that entry is "consumed" as the
        // main eltwise mode; remaining entries are chained as GPU primitives.
        size_t post_op_start = 0;
        if (config.eltwise_mode < 0 && !post_ops.empty() && post_ops[0].kind == post_op_kind::eltwise) {
            post_op_start = 1;
        }

        // Chain activation / quantize post-ops in the GPU topology
        for (size_t pi = post_op_start; pi < post_ops.size(); ++pi) {
            const auto& po = post_ops[pi];
            if (po.kind == post_op_kind::activation) {
                if (skip_activation_chain) {
                    continue;
                }
                auto act = map_activation(po.act_func, po.alpha);
                std::string act_id = "act_" + std::to_string(pi);
                topology.add(cldnn::activation(act_id,
                    cldnn::input_info(last_prim_id), act, {po.alpha, po.beta}));
                last_prim_id = act_id;
            }
        }

        cldnn::network network(engine, topology, exec_config);
        for (size_t i = 0; i < memories.size(); ++i) {
            network.set_input_data("input" + std::to_string(i), memories[i]);
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
            auto gpu_out = read_network_output_f32(outputs, last_prim_id, *stream);

            // Read inputs
            std::vector<float> in0 = read_memory_to_f32(memories[0], *stream);
            std::vector<float> in1 = read_memory_to_f32(memories[1], *stream);
            auto ref_out = ref::eltwise(in0, in1, mode, shapes[0], shapes[1]);

            // GPU with pythondiv uses floor-division for integer output types.
            // Keep floating-point outputs as true division.
            if (mode == cldnn::eltwise_mode::div && config.pythondiv != 0 && int_output_dt) {
                for (auto& v : ref_out) {
                    v = std::floor(v);
                }
            }

            // GPU eltwise output is quantized to the output dtype (e.g., u8 saturate).
            // Apply the same quantization to reference so comparison is valid.
            ref::apply_quantize_ref(ref_out, dt);

            // Apply activation post-ops to reference (e.g. relu)
            for (size_t pi = post_op_start; pi < post_ops.size(); ++pi) {
                const auto& po = post_ops[pi];
                if (po.kind == post_op_kind::activation) {
                    if (skip_activation_chain) {
                        continue;
                    }
                    auto act = map_activation(po.act_func, po.alpha);
                    for (auto& v : ref_out) {
                        v = ref::apply_activation(v, act, po.alpha, po.beta);
                    }
                }
            }

            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);

            // When output is integer (e.g. i32) but an input is float (e.g. f16),
            // GPU and CPU reference may differ by ±1 due to intermediate f16
            // truncation vs f32 rounding before floor/cast.  This is within spec.
            bool has_int_output = (dt == cldnn::data_types::i32 || dt == cldnn::data_types::i64 ||
                                   dt == cldnn::data_types::u8  || dt == cldnn::data_types::i8);
            bool has_float_input = false;
            for (size_t i = 0; i < dts.size(); ++i) {
                auto idt = (dts.size() > i) ? dts[i] : dt;
                if (idt == cldnn::data_types::f16 || idt == cldnn::data_types::f32) {
                    has_float_input = true;
                    break;
                }
            }
            if (has_int_output && has_float_input) {
                atol = std::max(atol, 1.0f);
            }

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
        print_result(network, "eltwise_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_eltwise)

// ============================================================================
// Activation kernel benchmark
//
// Usage:
//   --activation --dt=f16 --attr-post-ops=relu shape
//
// Example:
//   --activation --dt=f16 --attr-post-ops=relu 1x4096
//   --activation --dt=f16 --attr-post-ops=gelu_erf 1x64x56x56
// ============================================================================

class bench_activation : public kernel_base {
public:
    std::string name() const override { return "activation"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Activation requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        // Get activation function from post-ops
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        cldnn::activation_func act = cldnn::activation_func::relu;
        float alpha = 0.0f, beta = 0.0f;
        if (!post_ops.empty() && post_ops[0].kind == post_op_kind::activation) {
            act = map_activation(post_ops[0].act_func, post_ops[0].alpha);
            alpha = post_ops[0].alpha;
            beta = post_ops[0].beta;
        }

        auto exec_config = make_exec_config(config, "act_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout layout(ps, dt, get_input_format(config, 0, shape.size()));
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        topology.add(cldnn::activation("act_prim",
            cldnn::input_info("input"), act, {alpha, beta}));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", mem);

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
            auto gpu_out = read_network_output_f32(outputs, "act_prim", *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            auto ref_out = ref::activation(input_f32, act, alpha, beta);

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
        print_result(network, "act_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_activation)

}  // namespace bench_kernel
