// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>

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
#include <intel_gpu/primitives/quantize.hpp>

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
        cldnn::data_types default_input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        // The primitive's own output dtype follows the first dtype token (legacy behavior).
        // If fused post-ops include quantize, final output dtype is derived from post-ops.
        cldnn::data_types primitive_output_dt = default_input_dt;

        // Determine eltwise mode: config.eltwise_mode takes priority (direct from verbose log),
        // then fall back to post-ops string, then default to sum
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        cldnn::data_types final_output_dt = get_terminal_post_op_dtype(post_ops, primitive_output_dt);
        bool int_output_dt = (final_output_dt == cldnn::data_types::i8 || final_output_dt == cldnn::data_types::u8 ||
              final_output_dt == cldnn::data_types::i32 || final_output_dt == cldnn::data_types::i64);
        bool has_quantize_post_op = false;
        for (const auto& po : post_ops) {
            if (po.kind == post_op_kind::quantize) {
                has_quantize_post_op = true;
                break;
            }
        }
        bool skip_activation_chain = int_output_dt && has_quantize_post_op;

        auto parse_colon_floats = [](const std::string& s) {
            std::vector<float> out;
            if (s.empty()) return out;
            std::istringstream iss(s);
            std::string tok;
            while (std::getline(iss, tok, ':')) {
                out.push_back(std::stof(tok));
            }
            return out;
        };

        auto parse_stride_tensors = [](const std::string& s) {
            std::vector<cldnn::tensor> out;
            if (s.empty()) return out;
            std::istringstream tensor_iss(s);
            std::string tensor_tok;
            while (std::getline(tensor_iss, tensor_tok, ';')) {
                if (tensor_tok.empty()) {
                    continue;
                }
                std::vector<cldnn::tensor::value_type> dims;
                std::istringstream dim_iss(tensor_tok);
                std::string dim_tok;
                while (std::getline(dim_iss, dim_tok, ':')) {
                    dims.push_back(static_cast<cldnn::tensor::value_type>(std::stoll(dim_tok)));
                }
                out.emplace_back(dims);
            }
            return out;
        };

        auto to_broadcast_type = [](const std::string& s) {
            if (s == "none" || s == "NONE" || s == "0")
                return ov::op::AutoBroadcastType::NONE;
            if (s == "pdpd" || s == "PDPD" || s == "2")
                return ov::op::AutoBroadcastType::PDPD;
            return ov::op::AutoBroadcastType::NUMPY;
        };

        auto coeffs = parse_colon_floats(config.eltwise_coefficients);
        auto stride_vec = parse_stride_tensors(config.eltwise_stride);
        ov::op::AutoBroadcastSpec elt_broadcast_spec(
            to_broadcast_type(config.eltwise_broadcast_type),
            config.eltwise_broadcast_axis);
        cldnn::eltwise_mode mode = cldnn::eltwise_mode::sum;
        if (config.eltwise_mode >= 0) {
            mode = static_cast<cldnn::eltwise_mode>(config.eltwise_mode);
        } else if (!post_ops.empty() && post_ops[0].kind == post_op_kind::eltwise) {
            mode = map_eltwise_mode(post_ops[0].elt_mode);
        }

        // If eltwise mode is explicitly provided from verbose and fused eltwise post-ops
        // are present, trailing inputs can represent post-op tensors rather than
        // base eltwise operands.
        size_t post_op_start = 0;
        if (config.eltwise_mode < 0 && !post_ops.empty() && post_ops[0].kind == post_op_kind::eltwise) {
            post_op_start = 1;
        }
        size_t fused_eltwise_post_count = 0;
        for (size_t pi = post_op_start; pi < post_ops.size(); ++pi) {
            if (post_ops[pi].kind == post_op_kind::eltwise) {
                fused_eltwise_post_count++;
            }
        }

        auto exec_config = make_exec_config(config, "eltwise_prim");
        auto stream = engine.create_stream(exec_config);

        cldnn::topology topology;
        std::vector<cldnn::memory::ptr> memories;
        const size_t total_input_count = shapes.size();
        size_t base_input_count = total_input_count;
        if (config.eltwise_mode >= 0 && fused_eltwise_post_count > 0 && total_input_count >= 2 + fused_eltwise_post_count) {
            base_input_count = total_input_count - fused_eltwise_post_count;
        }
        base_input_count = std::max<size_t>(2, std::min(base_input_count, total_input_count));

        for (size_t i = 0; i < total_input_count; ++i) {
            std::string id = "input" + std::to_string(i);
            ov::PartialShape ps(std::vector<ov::Dimension>(shapes[i].begin(), shapes[i].end()));
            // Use per-input data type: dt[0] for input0, dt[1] for input1 (if available)
            cldnn::data_types input_dt = (dts.size() > i) ? dts[i] : default_input_dt;
            cldnn::layout layout(ps, input_dt, get_input_format(config, i, shapes[i].size()));
            auto mem = engine.allocate_memory(layout);
            fill_memory_random(mem, *stream, input_dt);
            topology.add(cldnn::input_layout(id, layout));
            memories.push_back(mem);
        }

        // Eltwise primitive path is binary-oriented in shape inference.
        // Support N inputs by chaining binary eltwise ops left-to-right.
        if (!stride_vec.empty()) {
            if (base_input_count != 2 || !coeffs.empty() || config.pythondiv != 0) {
                throw std::runtime_error("eltwise_stride supports only binary eltwise without coefficients/pythondiv");
            }
            topology.add(cldnn::eltwise("eltwise_prim",
                cldnn::input_info("input0"), cldnn::input_info("input1"),
                stride_vec, mode, elt_broadcast_spec));
        } else {
            topology.add(cldnn::eltwise("eltwise_prim", {cldnn::input_info("input0"), cldnn::input_info("input1")}, mode,
                coeffs, primitive_output_dt,
                elt_broadcast_spec,
                config.pythondiv != 0));
        }
        std::string last_prim_id = "eltwise_prim";
        for (size_t i = 2; i < base_input_count; ++i) {
            std::string elt_chain_id = "eltwise_chain_" + std::to_string(i);
            topology.add(cldnn::eltwise(elt_chain_id,
                {cldnn::input_info(last_prim_id), cldnn::input_info("input" + std::to_string(i))},
                mode,
                {}, primitive_output_dt,
                elt_broadcast_spec,
                config.pythondiv != 0));
            last_prim_id = elt_chain_id;
        }

        auto compute_broadcast_shape = [](const std::vector<int64_t>& a,
                                          const std::vector<int64_t>& b) {
            size_t rank = std::max(a.size(), b.size());
            std::vector<int64_t> out(rank, 1);
            for (size_t i = 0; i < rank; ++i) {
                int64_t av = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
                int64_t bv = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
                out[i] = std::max(av, bv);
            }
            return out;
        };
        std::vector<int64_t> out_shape = shapes[0];
        for (size_t i = 1; i < base_input_count; ++i) {
            out_shape = compute_broadcast_shape(out_shape, shapes[i]);
        }
        std::map<int, ref::elt_ref_data> elt_data_map;

        // Chain post-ops in the GPU topology and keep the runtime chain for CPU reference.
        std::vector<post_op_entry> runtime_post_ops;
        int runtime_post_op_idx = 0;
        size_t fused_eltwise_input_used = 0;
        for (size_t pi = post_op_start; pi < post_ops.size(); ++pi) {
            const auto& po = post_ops[pi];
            switch (po.kind) {
                case post_op_kind::activation: {
                    if (skip_activation_chain) {
                        break;
                    }
                    auto act = map_activation(po.act_func, po.alpha);
                    std::string act_id = "act_" + std::to_string(pi);
                    topology.add(cldnn::activation(act_id,
                        cldnn::input_info(last_prim_id), act, {po.alpha, po.beta}));
                    last_prim_id = act_id;
                    runtime_post_ops.push_back(po);
                    runtime_post_op_idx++;
                    break;
                }
                case post_op_kind::eltwise: {
                    std::vector<int64_t> elt_shape = out_shape;
                    if (po.elt_broadcast == broadcast_spec::per_oc) {
                        elt_shape.assign(out_shape.size(), 1);
                        if (!elt_shape.empty()) elt_shape[0] = out_shape[0];
                        if (elt_shape.size() > 1) elt_shape[1] = out_shape[1];
                    } else if (po.elt_broadcast == broadcast_spec::per_tensor) {
                        elt_shape.assign(out_shape.size(), 1);
                    }

                    std::string elt_data_id;
                    cldnn::input_info elt_input;
                    size_t post_tensor_idx = base_input_count + fused_eltwise_input_used;
                    if (post_tensor_idx < memories.size()) {
                        // Use trailing logged input as fused eltwise tensor.
                        elt_data_id = "input" + std::to_string(post_tensor_idx);
                        elt_input = cldnn::input_info(elt_data_id);
                        if (config.is_acc()) {
                            elt_data_map[runtime_post_op_idx] = {read_memory_to_f32(memories[post_tensor_idx], *stream), shapes[post_tensor_idx]};
                        }
                        fused_eltwise_input_used++;
                    } else {
                        // Fallback for manually-crafted commands without explicit fused tensor inputs.
                        elt_data_id = "elt_data_" + std::to_string(pi);
                        ov::PartialShape elt_ps(std::vector<ov::Dimension>(elt_shape.begin(), elt_shape.end()));
                        cldnn::layout elt_layout(elt_ps, po.elt_dt, get_format_for_rank(elt_shape.size()));
                        auto elt_mem = engine.allocate_memory(elt_layout);
                        fill_memory_random(elt_mem, *stream, po.elt_dt);
                        topology.add(cldnn::data(elt_data_id, elt_mem));
                        elt_input = cldnn::input_info(elt_data_id);
                        if (config.is_acc()) {
                            elt_data_map[runtime_post_op_idx] = {read_memory_to_f32(elt_mem, *stream), elt_shape};
                        }
                    }

                    std::string elt_id = "elt_post_" + std::to_string(pi);
                    topology.add(cldnn::eltwise(elt_id,
                        {cldnn::input_info(last_prim_id), elt_input},
                        map_eltwise_mode(po.elt_mode)));
                    last_prim_id = elt_id;
                    runtime_post_ops.push_back(po);
                    runtime_post_op_idx++;
                    break;
                }
                case post_op_kind::quantize: {
                    float lo = 0.0f;
                    float hi = 255.0f;
                    int levels = 256;
                    switch (po.quant_out_dt) {
                        case cldnn::data_types::i8: lo = -128.0f; hi = 127.0f; levels = 256; break;
                        case cldnn::data_types::u8: lo = 0.0f; hi = 255.0f; levels = 256; break;
                        case cldnn::data_types::i4: lo = -8.0f; hi = 7.0f; levels = 16; break;
                        case cldnn::data_types::u4: lo = 0.0f; hi = 15.0f; levels = 16; break;
                        default: break;
                    }

                    auto make_scalar = [&](const std::string& name, float val) {
                        cldnn::layout scalar_layout({1, 1, 1, 1}, cldnn::data_types::f32, cldnn::format::bfyx);
                        auto mem = engine.allocate_memory(scalar_layout);
                        { auto lock = cldnn::mem_lock<float>(mem, *stream); lock[0] = val; }
                        topology.add(cldnn::data(name, mem));
                    };

                    std::string q_id = "quant_" + std::to_string(pi);
                    std::string in_lo = q_id + "_in_lo";
                    std::string in_hi = q_id + "_in_hi";
                    std::string out_lo = q_id + "_out_lo";
                    std::string out_hi = q_id + "_out_hi";
                    make_scalar(in_lo, lo);
                    make_scalar(in_hi, hi);
                    make_scalar(out_lo, lo);
                    make_scalar(out_hi, hi);

                    topology.add(cldnn::quantize(q_id,
                        cldnn::input_info(last_prim_id),
                        cldnn::input_info(in_lo), cldnn::input_info(in_hi),
                        cldnn::input_info(out_lo), cldnn::input_info(out_hi),
                        levels, po.quant_out_dt));
                    last_prim_id = q_id;
                    runtime_post_ops.push_back(po);
                    runtime_post_op_idx++;
                    break;
                }
                default:
                    break;
            }
        }

        add_terminal_post_op_consumer_reorder(topology, config, runtime_post_ops, last_prim_id, primitive_output_dt);

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

            // Read all inputs and fold N-ary eltwise reference left-to-right.
            // This mirrors the N-input eltwise primitive execution path.
            std::vector<std::vector<float>> input_vals;
            input_vals.reserve(memories.size());
            for (size_t i = 0; i < memories.size(); ++i) {
                input_vals.push_back(read_memory_to_f32(memories[i], *stream));
            }

            auto ref_out = input_vals[0];
            std::vector<int64_t> ref_shape = shapes[0];
            for (size_t i = 1; i < base_input_count; ++i) {
                ref_out = ref::eltwise(ref_out, input_vals[i], mode, ref_shape, shapes[i]);
                ref_shape = compute_broadcast_shape(ref_shape, shapes[i]);
            }

            // GPU with pythondiv uses floor-division for integer output types.
            // Keep floating-point outputs as true division.
            if (mode == cldnn::eltwise_mode::div && config.pythondiv != 0 && int_output_dt) {
                for (auto& v : ref_out) {
                    v = std::floor(v);
                }
            }

            // GPU eltwise output is quantized to the output dtype (e.g., u8 saturate).
            // Apply the same quantization to reference so comparison is valid.
            ref::apply_quantize_ref(ref_out, primitive_output_dt);

            bool can_check = ref::apply_post_ops_ref(ref_out, runtime_post_ops, out_shape, elt_data_map);
            if (!can_check) {
                test_passed = false;
            }

            float atol, rtol;
            get_default_tolerance(final_output_dt, atol, rtol);

            // When output is integer (e.g. i32) but an input is float (e.g. f16),
            // GPU and CPU reference may differ by ±1 due to intermediate f16
            // truncation vs f32 rounding before floor/cast.  This is within spec.
            bool has_int_output = (final_output_dt == cldnn::data_types::i32 || final_output_dt == cldnn::data_types::i64 ||
                                   final_output_dt == cldnn::data_types::u8  || final_output_dt == cldnn::data_types::i8);
            bool has_float_input = false;
            for (size_t i = 0; i < memories.size(); ++i) {
                auto idt = (dts.size() > i) ? dts[i] : default_input_dt;
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
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        cldnn::data_types output_dt = (dts.size() > 1) ? dts.back() : input_dt;

        // Get base activation function from post-ops
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        cldnn::activation_func act = cldnn::activation_func::relu;
        float alpha = 0.0f, beta = 0.0f;
        size_t post_op_start = 0;
        if (!post_ops.empty() && post_ops[0].kind == post_op_kind::activation) {
            act = map_activation(post_ops[0].act_func, post_ops[0].alpha);
            alpha = post_ops[0].alpha;
            beta = post_ops[0].beta;
            post_op_start = 1;
        }

        auto exec_config = make_exec_config(config, "act_prim");
        auto stream = engine.create_stream(exec_config);

        auto& shape = shapes[0];
        ov::PartialShape ps(std::vector<ov::Dimension>(shape.begin(), shape.end()));
        cldnn::layout layout(ps, input_dt, get_input_format(config, 0, shape.size()));
        auto mem = engine.allocate_memory(layout);
        fill_memory_random(mem, *stream, input_dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", layout));
        topology.add(cldnn::activation("act_prim",
            cldnn::input_info("input"), act, {alpha, beta}));
        std::string last_prim_id = "act_prim";
        std::vector<int64_t> out_shape = shape;
        std::map<int, ref::elt_ref_data> elt_data_map;
        std::vector<post_op_entry> runtime_post_ops;

        int runtime_post_op_idx = 0;
        for (size_t pi = post_op_start; pi < post_ops.size(); ++pi) {
            const auto& po = post_ops[pi];
            switch (po.kind) {
                case post_op_kind::activation: {
                    std::string po_id = "act_post_" + std::to_string(pi);
                    topology.add(cldnn::activation(po_id,
                        cldnn::input_info(last_prim_id),
                        map_activation(po.act_func, po.alpha),
                        {po.alpha, po.beta}));
                    last_prim_id = po_id;
                    runtime_post_ops.push_back(po);
                    runtime_post_op_idx++;
                    break;
                }
                case post_op_kind::eltwise: {
                    std::vector<int64_t> elt_shape = out_shape;
                    if (po.elt_broadcast == broadcast_spec::per_oc) {
                        elt_shape.assign(out_shape.size(), 1);
                        if (!elt_shape.empty()) elt_shape[0] = out_shape[0];
                        if (elt_shape.size() > 1) elt_shape[1] = out_shape[1];
                    } else if (po.elt_broadcast == broadcast_spec::per_tensor) {
                        elt_shape.assign(out_shape.size(), 1);
                    }

                    std::string elt_data_id = "elt_data_" + std::to_string(pi);
                    ov::PartialShape elt_ps(std::vector<ov::Dimension>(elt_shape.begin(), elt_shape.end()));
                    cldnn::layout elt_layout(elt_ps, po.elt_dt, get_format_for_rank(elt_shape.size()));
                    auto elt_mem = engine.allocate_memory(elt_layout);
                    fill_memory_random(elt_mem, *stream, po.elt_dt);
                    if (config.is_acc()) {
                        elt_data_map[runtime_post_op_idx] = {read_memory_to_f32(elt_mem, *stream), elt_shape};
                    }
                    topology.add(cldnn::data(elt_data_id, elt_mem));

                    std::string po_id = "elt_post_" + std::to_string(pi);
                    topology.add(cldnn::eltwise(po_id,
                        {cldnn::input_info(last_prim_id), cldnn::input_info(elt_data_id)},
                        map_eltwise_mode(po.elt_mode)));
                    last_prim_id = po_id;
                    runtime_post_ops.push_back(po);
                    runtime_post_op_idx++;
                    break;
                }
                case post_op_kind::quantize: {
                    float lo = 0.0f;
                    float hi = 255.0f;
                    int levels = 256;
                    switch (po.quant_out_dt) {
                        case cldnn::data_types::i8: lo = -128.0f; hi = 127.0f; levels = 256; break;
                        case cldnn::data_types::u8: lo = 0.0f; hi = 255.0f; levels = 256; break;
                        case cldnn::data_types::i4: lo = -8.0f; hi = 7.0f; levels = 16; break;
                        case cldnn::data_types::u4: lo = 0.0f; hi = 15.0f; levels = 16; break;
                        default: break;
                    }

                    auto make_scalar = [&](const std::string& name, float val) {
                        cldnn::layout scalar_layout({1, 1, 1, 1}, cldnn::data_types::f32, cldnn::format::bfyx);
                        auto mem = engine.allocate_memory(scalar_layout);
                        { auto lock = cldnn::mem_lock<float>(mem, *stream); lock[0] = val; }
                        topology.add(cldnn::data(name, mem));
                    };

                    std::string q_id = "quant_" + std::to_string(pi);
                    std::string in_lo = q_id + "_in_lo";
                    std::string in_hi = q_id + "_in_hi";
                    std::string out_lo = q_id + "_out_lo";
                    std::string out_hi = q_id + "_out_hi";
                    make_scalar(in_lo, lo);
                    make_scalar(in_hi, hi);
                    make_scalar(out_lo, lo);
                    make_scalar(out_hi, hi);

                    topology.add(cldnn::quantize(q_id,
                        cldnn::input_info(last_prim_id),
                        cldnn::input_info(in_lo), cldnn::input_info(in_hi),
                        cldnn::input_info(out_lo), cldnn::input_info(out_hi),
                        levels, po.quant_out_dt));
                    last_prim_id = q_id;
                    runtime_post_ops.push_back(po);
                    runtime_post_op_idx++;
                    break;
                }
                default:
                    break;
            }
        }

        add_terminal_post_op_consumer_reorder(topology, config, runtime_post_ops, last_prim_id, output_dt);

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
            auto gpu_out = read_network_output_f32(outputs, last_prim_id, *stream);
            auto input_f32 = read_memory_to_f32(mem, *stream);
            auto ref_out = ref::activation(input_f32, act, alpha, beta);

            // Activation output follows primitive output dtype.
            ref::apply_quantize_ref(ref_out, output_dt);

            bool can_check = ref::apply_post_ops_ref(ref_out, runtime_post_ops, out_shape, elt_data_map);
            if (!can_check) {
                test_passed = false;
            }

            float atol, rtol;
            get_default_tolerance(output_dt, atol, rtol);
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
