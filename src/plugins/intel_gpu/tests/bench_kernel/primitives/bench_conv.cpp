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
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/reorder.hpp>
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
// Convolution kernel benchmark
//
// Usage:
//   --conv --dt=f16 NxCxHxW:OxCxKHxKW [--strides=1x1] [--dilation=1x1] [--padding=0x0] [--groups=1]
//
// Example:
//   --conv --dt=f16 1x64x56x56:64x64x3x3
//   --conv --dt=f16 1x3x224x224:64x3x7x7 --strides=2x2 --padding=3x3
// ============================================================================

class bench_conv : public kernel_base {
public:
    std::string name() const override { return "convolution"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // Parse shapes: input_shape:weight_shape
        // Input: NxCxHxW, Weight: OxCxKHxKW (or OxCxKDxKHxKW for 3D)
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Conv requires at least 2 shapes (input:weights). Got: " + config.shapes_str);
        }

        auto& input_shape = shapes[0];
        auto& weight_shape = shapes[1];

        // Skip convolutions with very large reference computation in accuracy mode.
        // Estimate: input_volume * kernel_spatial_volume (approximate FLOPs for naive reference).
        // Threshold is adjustable — 100M ref ops ≈ 30s on a typical single-thread CPU.
        static constexpr int64_t CONV_REF_OPS_SKIP_THRESHOLD = 100'000'000LL;
        if (config.is_acc()) {
            int64_t ref_ops = 1;
            for (auto d : input_shape) ref_ops *= d;
            for (size_t i = 2; i < weight_shape.size(); ++i)
                ref_ops *= weight_shape[i];
            if (ref_ops > CONV_REF_OPS_SKIP_THRESHOLD) {
                print_skip_result(config, "Conv reference too slow (" + std::to_string(ref_ops) + " ref ops)");
                throw bench_skipped("Conv reference too slow");
            }
        }

        // Data types
        auto dts = config.data_types;
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        cldnn::data_types weight_dt = dts.size() > 1 ? dts[1] : input_dt;
        // Converter now preserves all input dtypes and appends output dtype.
        // output index = number of input tensors from --shapes.
        const size_t input_tensor_count = shapes.size();
        cldnn::data_types output_dt = dts.size() > input_tensor_count ? dts[input_tensor_count]
                                           : (dts.size() > 2 ? dts[2] : input_dt);

        // Conv parameters - extract from shapes string after extra ':'
        // Default values
        size_t spatial_dims = input_shape.size() - 2;  // 2 for 2D, 3 for 3D
        std::vector<size_t> strides(spatial_dims, 1);
        std::vector<size_t> dilations(spatial_dims, 1);
        std::vector<ptrdiff_t> padding_begin(spatial_dims, 0);
        std::vector<ptrdiff_t> padding_end(spatial_dims, 0);
        uint32_t groups = static_cast<uint32_t>(config.groups);

        // Override from config if provided
        if (!config.strides.empty()) {
            auto v = parse_x_vec(config.strides);
            if (v.size() == spatial_dims) strides = v;
        }
        if (!config.dilations.empty()) {
            auto v = parse_x_vec(config.dilations);
            if (v.size() == spatial_dims) dilations = v;
        }
        if (!config.padding_begin.empty()) {
            auto v = parse_x_svec(config.padding_begin);
            if (v.size() == spatial_dims) padding_begin = v;
        }
        if (!config.padding_end.empty()) {
            auto v = parse_x_svec(config.padding_end);
            if (v.size() == spatial_dims) padding_end = v;
        }

        // Parse post-ops
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        resolve_quantize_dtypes(post_ops, output_dt, input_dt);
        bool has_quantize_post_op = std::any_of(post_ops.begin(), post_ops.end(),
            [](const post_op_entry& po) { return po.kind == post_op_kind::quantize; });

        auto exec_config = make_exec_config(config, "conv_prim");
        auto stream = engine.create_stream(exec_config);

        // Build layouts (select format based on tensor rank)
        auto fmt = get_input_format(config, 0, input_shape.size());
        auto wfmt = get_input_format(config, 1, weight_shape.size());
        ov::PartialShape input_ps(std::vector<ov::Dimension>(input_shape.begin(), input_shape.end()));
        ov::PartialShape weight_ps(std::vector<ov::Dimension>(weight_shape.begin(), weight_shape.end()));
        cldnn::layout input_layout(input_ps, input_dt, fmt);
        cldnn::layout weight_layout(weight_ps, weight_dt, wfmt);

        // Allocate
        auto input_mem = engine.allocate_memory(input_layout);
        auto weight_mem = engine.allocate_memory(weight_layout);
        fill_memory_random(input_mem, *stream, input_dt);
        fill_memory_random(weight_mem, *stream, weight_dt);

        // Save input data for accuracy check BEFORE network may reformat
        std::vector<float> input_f32_saved, weight_f32_saved;
        if (config.is_acc()) {
            input_f32_saved = read_memory_to_f32(input_mem, *stream);
            weight_f32_saved = read_memory_to_f32(weight_mem, *stream);
        }

        // Topology
        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::data("weights", weight_mem));

        // Optional extra inputs from verbose log (in2+):
        // bias, weights_zero_points, activations_zero_points, compensation
        std::vector<std::string> aux_ids;
        std::vector<float> bias_f32_saved;
        std::vector<int64_t> bias_shape_saved;
        for (size_t aux_idx = 2; aux_idx < shapes.size(); ++aux_idx) {
            const auto& aux_shape = shapes[aux_idx];
            cldnn::data_types aux_dt = dts.size() > aux_idx ? dts[aux_idx] : cldnn::data_types::f32;
            auto aux_fmt = get_input_format(config, aux_idx, aux_shape.size());
            ov::PartialShape aux_ps(std::vector<ov::Dimension>(aux_shape.begin(), aux_shape.end()));
            cldnn::layout aux_layout(aux_ps, aux_dt, aux_fmt);
            auto aux_mem = engine.allocate_memory(aux_layout);
            fill_memory_random(aux_mem, *stream, aux_dt);

            if (config.is_acc() && aux_idx == 2) {
                bias_f32_saved = read_memory_to_f32(aux_mem, *stream);
                bias_shape_saved = aux_shape;
            }

            std::string aux_id = "aux_" + std::to_string(aux_idx - 2);
            topology.add(cldnn::data(aux_id, aux_mem));
            aux_ids.push_back(aux_id);
        }

        std::string bias_id = aux_ids.size() > 0 ? aux_ids[0] : "";
        std::string w_zp_id = "";
        std::string a_zp_id = "";
        std::string compensation_id = "";

        ov::Strides ov_strides(strides.begin(), strides.end());
        ov::Strides ov_dilations(dilations.begin(), dilations.end());
        ov::CoordinateDiff ov_padding_begin(padding_begin.begin(), padding_begin.end());
        ov::CoordinateDiff ov_padding_end(padding_end.begin(), padding_end.end());

        // Keep conv output in compute domain when quantize post-op exists so quantize can be represented
        // explicitly and optimized/fused by graph passes.
        bool force_conv_output_dtype = (output_dt != input_dt) && !has_quantize_post_op;
        if (!bias_id.empty() || !w_zp_id.empty() || !a_zp_id.empty() || !compensation_id.empty() || force_conv_output_dtype) {
            topology.add(cldnn::convolution("conv_prim",
                cldnn::input_info("input"),
                "weights",
                bias_id,
                w_zp_id,
                a_zp_id,
                compensation_id,
                groups,
                ov_strides,
                ov_dilations,
                ov_padding_begin,
                ov_padding_end,
                config.grouped_weights_shape != 0,
                force_conv_output_dtype ? output_dt : input_dt));
        } else {
            topology.add(cldnn::convolution("conv_prim",
                cldnn::input_info("input"),
                "weights",
                "",  // no bias
                groups,
                ov_strides,
                ov_dilations,
                ov_padding_begin,
                ov_padding_end,
                config.grouped_weights_shape != 0));
        }

        std::string last_prim_id = "conv_prim";

        // Post-ops
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
                    // Conv output shape: [N, OC, spatial_out...]
                    // Must account for stride, dilation, and padding
                    // Use standard weight shape for kernel dims
                    auto elt_weight_shape = (config.grouped_weights_shape != 0)
                        ? convert_grouped_weight_shape(weight_shape) : weight_shape;
                    int64_t elt_OC = elt_weight_shape[0];
                    std::vector<int64_t> conv_out_shape = {input_shape[0], elt_OC};
                    for (size_t d = 0; d < spatial_dims; ++d) {
                        int64_t effective_k = static_cast<int64_t>(dilations[d]) * (elt_weight_shape[d + 2] - 1) + 1;
                        int64_t out_d = (input_shape[d + 2] + static_cast<int64_t>(padding_begin[d])
                                         + static_cast<int64_t>(padding_end[d]) - effective_k) / static_cast<int64_t>(strides[d]) + 1;
                        conv_out_shape.push_back(out_d);
                    }
                    std::vector<int64_t> elt_shape = conv_out_shape;  // full output shape
                    if (po.elt_broadcast == broadcast_spec::per_oc) {
                        elt_shape = std::vector<int64_t>(conv_out_shape.size(), 1);
                        elt_shape[0] = conv_out_shape[0];
                        elt_shape[1] = conv_out_shape[1];
                    } else if (po.elt_broadcast == broadcast_spec::per_tensor) {
                        elt_shape = std::vector<int64_t>(conv_out_shape.size(), 1);
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
                    bool is_intermediate = (post_op_idx + 1 < static_cast<int>(post_ops.size()));
                    float lo = 0, hi = 0;
                    int levels = 256;
                    switch (po.quant_out_dt) {
                        case cldnn::data_types::i8:  lo = -128; hi = 127; levels = 256; break;
                        case cldnn::data_types::u8:  lo = 0;    hi = 255; levels = 256; break;
                        case cldnn::data_types::i4:  lo = -8;   hi = 7;   levels = 16;  break;
                        case cldnn::data_types::u4:  lo = 0;    hi = 15;  levels = 16;  break;
                        default: lo = -128; hi = 127; levels = 256; break;
                    }
                    auto make_scalar = [&](const std::string& name, float val) {
                        cldnn::layout scalar_layout({1, 1, 1, 1}, cldnn::data_types::f32, cldnn::format::bfyx);
                        auto mem = engine.allocate_memory(scalar_layout);
                        { auto lock = cldnn::mem_lock<float>(mem, *stream); lock[0] = val; }
                        topology.add(cldnn::data(name, mem));
                    };

                    std::string in_lo_id = po_id + "_in_lo";
                    std::string in_hi_id = po_id + "_in_hi";
                    std::string out_lo_id = po_id + "_out_lo";
                    std::string out_hi_id = po_id + "_out_hi";
                    make_scalar(in_lo_id, lo);
                    make_scalar(in_hi_id, hi);
                    make_scalar(out_lo_id, lo);
                    make_scalar(out_hi_id, hi);

                    auto fq_out_dt = is_intermediate
                        ? ((output_dt == cldnn::data_types::f32) ? cldnn::data_types::f32 : cldnn::data_types::f16)
                        : po.quant_out_dt;
                    topology.add(cldnn::quantize(po_id,
                        cldnn::input_info(last_prim_id),
                        cldnn::input_info(in_lo_id), cldnn::input_info(in_hi_id),
                        cldnn::input_info(out_lo_id), cldnn::input_info(out_hi_id),
                        levels, fq_out_dt));
                    last_prim_id = po_id;
                    break;
                }
                default: break;
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
        network.set_input_data("input", input_mem);

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
            // Convert grouped weight shape to standard form for reference
            auto ref_weight_shape = (config.grouped_weights_shape != 0)
                ? convert_grouped_weight_shape(weight_shape) : weight_shape;
            auto ref_out = ref::convNd(input_f32_saved, input_shape, weight_f32_saved, ref_weight_shape,
                                       groups, strides, dilations, padding_begin, padding_end);

            // Compute output shape (with stride/dilation/padding)
            // OC: for grouped [G, OC/G, ...] → G*OC/G; for standard [OC, IC, ...] → OC
            int64_t OC = (config.grouped_weights_shape != 0)
                ? weight_shape[0] * weight_shape[1] : weight_shape[0];
            std::vector<int64_t> out_shape = {input_shape[0], OC};
            for (size_t d = 0; d < spatial_dims; ++d) {
                int64_t effective_k = static_cast<int64_t>(dilations[d]) * (ref_weight_shape[d + 2] - 1) + 1;
                int64_t out_d = (input_shape[d + 2] + static_cast<int64_t>(padding_begin[d])
                                 + static_cast<int64_t>(padding_end[d]) - effective_k) / static_cast<int64_t>(strides[d]) + 1;
                out_shape.push_back(out_d);
            }

            // Add bias term to reference output when bias tensor is present.
            if (!bias_f32_saved.empty() && out_shape.size() >= 2) {
                int64_t N = out_shape[0];
                int64_t C = out_shape[1];
                size_t spatial = 1;
                for (size_t d = 2; d < out_shape.size(); ++d)
                    spatial *= static_cast<size_t>(out_shape[d]);

                auto get_bias = [&](int64_t n, int64_t c) {
                    if (bias_f32_saved.size() == 1)
                        return bias_f32_saved[0];
                    if (bias_shape_saved.size() >= 2 && bias_shape_saved[1] == C &&
                        bias_f32_saved.size() == static_cast<size_t>(C))
                        return bias_f32_saved[static_cast<size_t>(c)];
                    if (bias_shape_saved.size() >= 2 && bias_shape_saved[0] == N && bias_shape_saved[1] == C &&
                        bias_f32_saved.size() == static_cast<size_t>(N * C))
                        return bias_f32_saved[static_cast<size_t>(n * C + c)];
                    return bias_f32_saved[static_cast<size_t>(c) % bias_f32_saved.size()];
                };

                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        float b = get_bias(n, c);
                        size_t base = static_cast<size_t>((n * C + c) * static_cast<int64_t>(spatial));
                        for (size_t s = 0; s < spatial; ++s)
                            ref_out[base + s] += b;
                    }
                }
            }

            // Apply post-ops to reference
            bool can_check = ref::apply_post_ops_ref(ref_out, post_ops, out_shape, elt_data_map);
            if (can_check) {
                float atol, rtol;
                get_default_tolerance(input_dt, atol, rtol);
                int quant_count = static_cast<int>(std::count_if(post_ops.begin(), post_ops.end(),
                    [](const post_op_entry& po) { return po.kind == post_op_kind::quantize; }));
                if (quant_count > 0) { atol = std::max(atol, static_cast<float>(quant_count)); }
                bool has_mixed_quant_chain = quant_count > 0 && std::any_of(post_ops.begin(), post_ops.end(),
                    [](const post_op_entry& po) {
                        return po.kind == post_op_kind::eltwise ||
                               (po.kind == post_op_kind::activation && po.act_func == activation_func::clamp);
                    });
                // Quantized mixed chains may differ by ~1 LSB on a tiny fraction of elements
                // due to backend rounding order. Allow small slack to avoid false negatives.
                if (has_mixed_quant_chain) {
                    atol = std::max(atol, 1.1f);
                }
                // Exponential-family post-ops amplify small conv errors
                bool has_exp_postop = std::any_of(post_ops.begin(), post_ops.end(),
                    [](const post_op_entry& po) {
                        return po.kind == post_op_kind::activation &&
                            (po.act_func == activation_func::exp ||
                             po.act_func == activation_func::softplus);
                    });
                if (has_exp_postop) {
                    atol = std::max(atol, 1.0f);
                    rtol = std::max(rtol, 0.1f);
                }
                // Scale f16 tolerance by kernel spatial volume to account for
                // MAC accumulation error. Each MAC adds up to 1 f16 ULP of
                // rounding error; for large kernels (e.g., 5x5 = 25 MACs) the
                // accumulated error can exceed the default atol of 5e-3.
                if (input_dt == cldnn::data_types::f16) {
                    size_t kernel_volume = 1;
                    for (size_t d = 2; d < weight_shape.size(); ++d)
                        kernel_volume *= weight_shape[d];
                    if (kernel_volume > 9) {
                        float f16_eps = 1.0f / 1024.0f;  // 2^-10
                        float mac_atol = static_cast<float>(kernel_volume) * f16_eps;
                        atol = std::max(atol, mac_atol);
                    }
                }
                // Grouped 5D convolutions have higher numerical variance
                if (groups > 1 && spatial_dims >= 3) {
                    atol = std::max(atol, 0.5f);
                    rtol = std::max(rtol, 0.05f);
                }
                float threshold = 0.0f;
                // Allow small mismatch ratio for exp post-ops (f16 amplification)
                if (has_exp_postop) threshold = 0.01f;
                if (has_mixed_quant_chain) threshold = std::max(threshold, 1e-5f);
                acc_res = compare_f32(gpu_out, ref_out, atol, rtol, threshold);
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
        print_result(network, "conv_prim", config, test_passed, acc_unimplemented, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed, acc_unimplemented);
    }
};

REGISTER_KERNEL(bench_conv)

}  // namespace bench_kernel
