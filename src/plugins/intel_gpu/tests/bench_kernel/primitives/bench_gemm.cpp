// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/reorder.hpp>

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
// Gemm kernel benchmark
// ============================================================================

class bench_gemm : public kernel_base {
public:
    std::string name() const override { return "gemm"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // Parse shapes: input0_shape:input1_shape[:input2_shape]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Gemm requires at least 2 shapes (input0:input1). Got: " + config.shapes_str);
        }

        auto input0_shape = shapes[0];
        auto input1_shape = shapes[1];
        bool has_input2 = shapes.size() >= 3;

        // Strip trailing 1-dimensions (verbose log pads to 4D)
        auto strip_trailing_ones = [](std::vector<int64_t>& shape) {
            while (shape.size() > 2 && shape.back() == 1) {
                shape.pop_back();
            }
        };
        strip_trailing_ones(input0_shape);
        strip_trailing_ones(input1_shape);

        // Parse data types: input0:input1:output (3 types) or input:output (2 types)
        auto dts = config.data_types;
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        cldnn::data_types weight_dt = dts.size() > 1 ? dts[1] : input_dt;
        cldnn::data_types output_dt = dts.size() > 2 ? dts[2] : (dts.size() > 1 ? dts[1] : input_dt);

        // Parse post-ops
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        resolve_quantize_dtypes(post_ops, output_dt, input_dt);

        auto exec_config = make_exec_config(config, "gemm_prim");
        auto stream = engine.create_stream(exec_config);

        // Allocate memories
        ov::PartialShape input0_ps(std::vector<ov::Dimension>(input0_shape.begin(), input0_shape.end()));
        ov::PartialShape input1_ps(std::vector<ov::Dimension>(input1_shape.begin(), input1_shape.end()));
        auto fmt = get_input_format(config, 0, std::max(input0_shape.size(), input1_shape.size()));
        cldnn::layout input0_layout(input0_ps, input_dt, get_input_format(config, 0, input0_shape.size()));
        cldnn::layout input1_layout(input1_ps, weight_dt, get_input_format(config, 1, input1_shape.size()));

        auto input0_mem = engine.allocate_memory(input0_layout);
        auto input1_mem = engine.allocate_memory(input1_layout);
        fill_memory_random(input0_mem, *stream, input_dt);
        fill_memory_random(input1_mem, *stream, weight_dt);

        // Save input data for accuracy check BEFORE network may reformat
        std::vector<float> a_f32_saved, b_f32_saved, c_f32_saved;
        if (config.is_acc()) {
            a_f32_saved = read_memory_to_f32(input0_mem, *stream);
            b_f32_saved = read_memory_to_f32(input1_mem, *stream);
        }

        // Build topology
        cldnn::topology topology;
        topology.add(cldnn::input_layout("input0", input0_layout));
        topology.add(cldnn::input_layout("input1", input1_layout));

        std::vector<cldnn::input_info> gemm_inputs = {
            cldnn::input_info("input0"),
            cldnn::input_info("input1")
        };

        // Optional third input (for beta * C)
        cldnn::memory::ptr input2_mem;
        if (has_input2) {
            auto& input2_shape = shapes[2];
            ov::PartialShape input2_ps(std::vector<ov::Dimension>(input2_shape.begin(), input2_shape.end()));
            cldnn::layout input2_layout(input2_ps, input_dt, get_input_format(config, 2, input2_shape.size()));
            input2_mem = engine.allocate_memory(input2_layout);
            fill_memory_random(input2_mem, *stream, input_dt);
            if (config.is_acc()) {
                c_f32_saved = read_memory_to_f32(input2_mem, *stream);
            }
            topology.add(cldnn::input_layout("input2", input2_layout));
            gemm_inputs.push_back(cldnn::input_info("input2"));
        }

        // Use order-based constructor if gemm_order0/order1 are provided,
        // otherwise fall back to legacy transpose_a/transpose_b boolean constructor
        size_t rank0 = input0_shape.size();
        size_t rank1 = input1_shape.size();

        if (!config.gemm_order0.empty() || !config.gemm_order1.empty()) {
            // Parse arbitrary transpose orders
            auto order0 = config.gemm_order0.empty()
                ? std::vector<int64_t>()  // default identity will be built below
                : parse_colon_vec(config.gemm_order0);
            auto order1 = config.gemm_order1.empty()
                ? std::vector<int64_t>()
                : parse_colon_vec(config.gemm_order1);
            auto order_out = config.gemm_order_out.empty()
                ? std::vector<int64_t>()
                : parse_colon_vec(config.gemm_order_out);

            // Build identity order for missing orders
            if (order0.empty()) {
                order0.resize(rank0);
                std::iota(order0.begin(), order0.end(), 0);
            }
            if (order1.empty()) {
                order1.resize(rank1);
                std::iota(order1.begin(), order1.end(), 0);
            }

            topology.add(cldnn::gemm("gemm_prim",
                gemm_inputs,
                output_dt,
                order0,
                order1,
                order_out,
                1.0f,   // alpha
                has_input2 ? 1.0f : 0.0f));  // beta
        } else {
            topology.add(cldnn::gemm("gemm_prim",
                gemm_inputs,
                output_dt,
                static_cast<uint32_t>(config.transpose_a),
                static_cast<uint32_t>(config.transpose_b),
                1.0f,   // alpha
                has_input2 ? 1.0f : 0.0f,  // beta
                rank0,
                rank1));
        }

        std::string last_prim_id = "gemm_prim";

        // Add post-ops
        int post_op_idx = 0;
        std::map<int, ref::elt_ref_data> elt_data_map;  // for accuracy check
        for (const auto& po : post_ops) {
            std::string po_id = "post_op_" + std::to_string(post_op_idx);

            switch (po.kind) {
                case post_op_kind::activation: {
                    topology.add(cldnn::activation(po_id,
                        cldnn::input_info(last_prim_id),
                        map_activation(po.act_func, po.alpha),
                        {po.alpha, po.beta}));
                    last_prim_id = po_id;
                    break;
                }
                case post_op_kind::eltwise: {
                    // Determine output shape for eltwise
                    // Gemm output shape must consider permutation orders
                    // After permute, matmul dims: A_permuted[..., M, K] * B_permuted[..., K, N] -> [..., M, N]
                    std::vector<int64_t> out_shape;
                    {
                        // Apply permutation orders to get effective shapes
                        auto apply_order = [](const std::vector<int64_t>& shape, const std::string& order_str) {
                            if (order_str.empty()) return shape;
                            std::vector<int64_t> order;
                            std::istringstream iss(order_str);
                            std::string tok;
                            while (std::getline(iss, tok, ':')) order.push_back(std::stoi(tok));
                            if (order.size() != shape.size()) return shape;
                            std::vector<int64_t> permuted(shape.size());
                            for (size_t i = 0; i < shape.size(); ++i)
                                permuted[i] = shape[order[i]];
                            return permuted;
                        };
                        auto perm0 = apply_order(input0_shape, config.gemm_order0);
                        auto perm1 = apply_order(input1_shape, config.gemm_order1);
                        // Output shape = batch dims from perm0, M from perm0[-2], N from perm1[-1]
                        out_shape = perm0;
                        if (out_shape.size() >= 1 && perm1.size() >= 1)
                            out_shape.back() = perm1.back();
                    }

                    std::vector<int64_t> elt_shape;
                    if (po.elt_broadcast == broadcast_spec::per_oc) {
                        elt_shape.resize(out_shape.size(), 1);
                        elt_shape.back() = out_shape.back();
                    } else if (po.elt_broadcast == broadcast_spec::per_tensor) {
                        elt_shape.resize(out_shape.size(), 1);
                    } else {
                        elt_shape = out_shape;
                    }

                    std::string elt_data_id = "elt_data_" + std::to_string(post_op_idx);
                    ov::PartialShape elt_ps(std::vector<ov::Dimension>(elt_shape.begin(), elt_shape.end()));
                    cldnn::layout elt_layout(elt_ps, po.elt_dt, get_format_for_rank(elt_shape.size()));
                    auto elt_mem = engine.allocate_memory(elt_layout);
                    fill_memory_random(elt_mem, *stream, po.elt_dt);

                    // Save eltwise data for accuracy check
                    if (config.is_acc()) {
                        elt_data_map[post_op_idx] = {read_memory_to_f32(elt_mem, *stream), elt_shape};
                    }

                    topology.add(cldnn::data(elt_data_id, elt_mem));

                    topology.add(cldnn::eltwise(po_id,
                        {cldnn::input_info(last_prim_id), cldnn::input_info(elt_data_id)},
                        map_eltwise_mode(po.elt_mode)));
                    last_prim_id = po_id;
                    break;
                }
                case post_op_kind::quantize: {
                    topology.add(cldnn::reorder(po_id,
                        cldnn::input_info(last_prim_id),
                        fmt, po.quant_out_dt));
                    last_prim_id = po_id;
                    // If more post-ops follow, reorder back to f16 to match
                    // actual model behavior where fused ops compute in float
                    if (post_op_idx + 1 < static_cast<int>(post_ops.size())) {
                        std::string dequant_id = po_id + "_dequant";
                        topology.add(cldnn::reorder(dequant_id,
                            cldnn::input_info(last_prim_id),
                            fmt, cldnn::data_types::f16));
                        last_prim_id = dequant_id;
                    }
                    break;
                }
                default:
                    break;
            }
            post_op_idx++;
        }

        // Keep the final post-op node non-output in perf mode so prepare_primitive_fusing
        // can consider post-op fusion for terminal chains.
        add_terminal_post_op_consumer_reorder(topology, config, post_ops, last_prim_id, output_dt);

        // For accuracy mode, add reorder to f32/bfyx to ensure readable output layout
        std::string output_prim_id = last_prim_id;
        if (config.is_acc()) {
            topology.add(cldnn::reorder("output_reorder",
                cldnn::input_info(last_prim_id),
                fmt, cldnn::data_types::f32));
            output_prim_id = "output_reorder";
        }

        // Build & execute
        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input0", input0_mem);
        network.set_input_data("input1", input1_mem);
        if (has_input2) {
            network.set_input_data("input2", input2_mem);
        }

        // Execute and collect results
        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        bool acc_unimplemented = false;
        acc_result acc_res;
        bool has_acc = false;
        perf_timer timer;
        bool has_perf = false;

        // Accuracy mode
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, output_prim_id, *stream);

            // Apply input permutations if order-based gemm is used
            std::vector<float> a_data = a_f32_saved;
            std::vector<int64_t> a_shape = input0_shape;
            std::vector<float> b_data = b_f32_saved;
            std::vector<int64_t> b_shape = input1_shape;

            if (!config.gemm_order0.empty()) {
                auto order0 = parse_colon_vec(config.gemm_order0);
                if (!ref::is_identity_order(order0)) {
                    std::vector<float> permuted;
                    std::vector<int64_t> permuted_shape;
                    ref::permute_tensor(a_f32_saved, input0_shape, order0, permuted, permuted_shape);
                    a_data = std::move(permuted);
                    a_shape = std::move(permuted_shape);
                }
            }
            if (!config.gemm_order1.empty()) {
                auto order1 = parse_colon_vec(config.gemm_order1);
                if (!ref::is_identity_order(order1)) {
                    std::vector<float> permuted;
                    std::vector<int64_t> permuted_shape;
                    ref::permute_tensor(b_f32_saved, input1_shape, order1, permuted, permuted_shape);
                    b_data = std::move(permuted);
                    b_shape = std::move(permuted_shape);
                }
            }

            // After permutation, inputs are in standard layout: A[...,M,K], B[...,K,N]
            // If using order-based permutation, no additional transpose needed.
            // If no orders specified, use transpose_a/transpose_b flags.
            bool use_transpose_a = config.gemm_order0.empty() && config.transpose_a != 0;
            bool use_transpose_b = config.gemm_order1.empty() && config.transpose_b != 0;
            auto ref_out = ref::gemm(a_data, a_shape, b_data, b_shape, use_transpose_a, use_transpose_b);

            // If 3-input (C = A*B + C), add C to reference
            if (has_input2) {
                for (size_t i = 0; i < ref_out.size() && i < c_f32_saved.size(); ++i) {
                    ref_out[i] += c_f32_saved[i];
                }
            }

            // Gemm output shape: [..., M, N] from permuted shapes
            size_t rk = a_shape.size();
            int64_t M_out = use_transpose_a ? a_shape[rk - 1] : a_shape[rk - 2];
            int64_t N_out = use_transpose_b ? b_shape[rk - 2] : b_shape[rk - 1];
            std::vector<int64_t> out_shape;
            for (size_t i = 0; i < rk - 2; ++i) out_shape.push_back(a_shape[i]);
            out_shape.push_back(M_out);
            out_shape.push_back(N_out);

            // Apply output permutation if specified
            if (!config.gemm_order_out.empty()) {
                auto order_out = parse_colon_vec(config.gemm_order_out);
                if (!ref::is_identity_order(order_out)) {
                    std::vector<float> permuted;
                    std::vector<int64_t> permuted_shape;
                    ref::permute_tensor(ref_out, out_shape, order_out, permuted, permuted_shape);
                    ref_out = std::move(permuted);
                    out_shape = std::move(permuted_shape);
                }
            }

            // Apply post-ops to reference
            bool can_check = ref::apply_post_ops_ref(ref_out, post_ops, out_shape, elt_data_map);
            if (can_check) {
                float atol, rtol;
                get_default_tolerance(input_dt, atol, rtol);
                // For mixed int-input / float-output (e.g., u8:i8:f16), use the
                // wider of input and output tolerances to account for accumulation
                // error in integer matmul followed by float conversion/post-ops.
                if (output_dt != input_dt) {
                    float oatol, ortol;
                    get_default_tolerance(output_dt, oatol, ortol);
                    atol = std::max(atol, oatol);
                    rtol = std::max(rtol, ortol);
                }
                int quant_count = static_cast<int>(std::count_if(post_ops.begin(), post_ops.end(),
                    [](const post_op_entry& po) { return po.kind == post_op_kind::quantize; }));
                if (quant_count > 0) { atol = std::max(atol, static_cast<float>(quant_count)); }
                acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
                has_acc = true;
                reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
                if (!acc_res.pass) test_passed = false;
            } else {
                acc_unimplemented = true;
            }
        }

        // Performance mode
        if (config.is_perf()) {
            run_perf(network, config, timer);
            has_perf = true;
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "gemm_prim", config, test_passed, acc_unimplemented, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed, acc_unimplemented);
    }
};

REGISTER_KERNEL(bench_gemm)

}  // namespace bench_kernel
