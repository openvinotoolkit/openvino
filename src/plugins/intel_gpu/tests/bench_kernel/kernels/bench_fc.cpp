// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
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
// Fully Connected kernel benchmark
// ============================================================================

class bench_fc : public kernel_base {
public:
    std::string name() const override { return "fully_connected"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // Parse shapes: input_shape:weight_shape[:bias_shape]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("FC requires at least 2 shapes (input:weights). Got: " + config.shapes_str);
        }

        // Copy shapes (don't modify original) and strip trailing 1-dimensions
        // Verbose log pads FC shapes to 4D (e.g., [1,128,768,1] -> [1,128,768])
        // FC primitive expects original ranks for proper MatMul computation
        auto input_shape = shapes[0];
        auto weight_shape = shapes[1];
        auto strip_trailing_ones = [](std::vector<int64_t>& shape) {
            while (shape.size() > 2 && shape.back() == 1) {
                shape.pop_back();
            }
        };
        strip_trailing_ones(input_shape);
        strip_trailing_ones(weight_shape);

        // Flatten input leading dimensions when input rank > weight rank
        // e.g., input 1x1x1x512, weight 29x512 -> input 1x512
        if (input_shape.size() > weight_shape.size()) {
            // Strip leading 1s first
            while (input_shape.size() > weight_shape.size() && input_shape.front() == 1) {
                input_shape.erase(input_shape.begin());
            }
            // If still larger, merge remaining leading dims
            if (input_shape.size() > weight_shape.size()) {
                size_t target_rank = weight_shape.size();
                int64_t merged = 1;
                while (input_shape.size() > target_rank) {
                    merged *= input_shape.front();
                    input_shape.erase(input_shape.begin());
                }
                input_shape.front() *= merged;
            }
        }

        // Parse data types
        auto dts = config.data_types;
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        cldnn::data_types weight_dt = dts.size() > 1 ? dts[1] : input_dt;
        cldnn::data_types output_dt = dts.size() > 2 ? dts[2] : input_dt;

        bool compressed_weights = (weight_dt != input_dt) &&
                                   (weight_dt == cldnn::data_types::i4 ||
                                    weight_dt == cldnn::data_types::u4 ||
                                    weight_dt == cldnn::data_types::i8 ||
                                    weight_dt == cldnn::data_types::u8);

        // Parse attributes
        auto scales = parse_scales(config.attr_scales_str);
        auto zero_points = parse_zero_points(config.attr_zero_points_str);
        auto post_ops = parse_post_ops(config.attr_post_ops_str);
        resolve_quantize_dtypes(post_ops, output_dt, input_dt);

        auto exec_config = make_exec_config(config, "fc_prim");
        auto stream = engine.create_stream(exec_config);

        // Build input layout
        ov::PartialShape input_ps(std::vector<ov::Dimension>(input_shape.begin(), input_shape.end()));
        auto fmt = get_input_format(config, 0, input_shape.size());
        cldnn::layout input_layout(input_ps, input_dt, fmt);

        // Build weight layout
        ov::PartialShape weight_ps(std::vector<ov::Dimension>(weight_shape.begin(), weight_shape.end()));
        cldnn::layout weight_layout(weight_ps, weight_dt, get_input_format(config, 1, weight_shape.size()));

        // Allocate memories
        auto input_mem = engine.allocate_memory(input_layout);
        auto weight_mem = engine.allocate_memory(weight_layout);
        fill_memory_random(input_mem, *stream, input_dt);
        fill_memory_random(weight_mem, *stream, weight_dt);

        // Save input data for accuracy check BEFORE network may reformat
        std::vector<float> input_f32_saved, weight_f32_saved;
        std::vector<float> scale_f32_saved, zp_f32_saved;
        std::vector<int64_t> saved_scale_shape, saved_zp_shape;
        if (config.is_acc()) {
            input_f32_saved = read_memory_to_f32(input_mem, *stream);
            weight_f32_saved = read_memory_to_f32(weight_mem, *stream);
        }

        size_t output_dim_size = input_shape.size();
        size_t weights_rank = weight_shape.size();

        // Build topology
        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::data("weights", weight_mem));

        std::string last_prim_id;

        if (compressed_weights) {
            // Find weight scale/zp in attributes
            const scale_entry* wei_scale = nullptr;
            const zero_point_entry* wei_zp = nullptr;
            for (auto& s : scales) {
                if (s.arg == attr_arg::wei) { wei_scale = &s; break; }
            }
            for (auto& z : zero_points) {
                if (z.arg == attr_arg::wei) { wei_zp = &z; break; }
            }

            // Compute decompression scale shape
            std::vector<int64_t> scale_shape;
            if (wei_scale) {
                if (wei_scale->policy == attr_policy::per_oc) {
                    scale_shape = {weight_shape[0], 1};
                } else if (wei_scale->policy == attr_policy::per_ocic && !wei_scale->groups.empty()) {
                    int64_t oc = weight_shape[0];
                    int64_t group = wei_scale->groups.back();
                    int64_t ic_groups = (weight_shape[1] + group - 1) / group;
                    scale_shape = {oc, ic_groups};
                } else {
                    scale_shape = {1, 1};
                }
            } else {
                scale_shape = {weight_shape[0], 1};
            }

            cldnn::data_types scale_dt = wei_scale ? wei_scale->dt : cldnn::data_types::f16;
            ov::PartialShape scale_ps(std::vector<ov::Dimension>(scale_shape.begin(), scale_shape.end()));
            cldnn::layout scale_layout(scale_ps, scale_dt, cldnn::format::bfyx);
            auto scale_mem = engine.allocate_memory(scale_layout);
            fill_memory_random(scale_mem, *stream, scale_dt);
            if (config.is_acc()) {
                scale_f32_saved = read_memory_to_f32(scale_mem, *stream);
                saved_scale_shape = scale_shape;
            }
            topology.add(cldnn::data("decompression_scale", scale_mem));

            std::string zp_id = "";
            if (wei_zp) {
                std::vector<int64_t> zp_shape = scale_shape;
                ov::PartialShape zp_ps(std::vector<ov::Dimension>(zp_shape.begin(), zp_shape.end()));
                // GPU converts u4/i4 ZP to u8/i8 internally, so use u8/i8 for ZP memory
                cldnn::data_types zp_alloc_dt = wei_zp->dt;
                if (zp_alloc_dt == cldnn::data_types::u4) zp_alloc_dt = cldnn::data_types::u8;
                if (zp_alloc_dt == cldnn::data_types::i4) zp_alloc_dt = cldnn::data_types::i8;
                cldnn::layout zp_layout(zp_ps, zp_alloc_dt, cldnn::format::bfyx);
                auto zp_mem = engine.allocate_memory(zp_layout);
                fill_memory_random(zp_mem, *stream, zp_alloc_dt);
                if (config.is_acc()) {
                    zp_f32_saved = read_memory_to_f32(zp_mem, *stream);
                    saved_zp_shape = scale_shape;
                }
                topology.add(cldnn::data("decompression_zp", zp_mem));
                zp_id = "decompression_zp";
            }

            // Check for activation scale/zp (dynamic quantization)
            const scale_entry* act_scale = nullptr;
            const zero_point_entry* act_zp = nullptr;
            for (auto& s : scales) {
                if (s.arg == attr_arg::src) { act_scale = &s; break; }
            }
            for (auto& z : zero_points) {
                if (z.arg == attr_arg::src) { act_zp = &z; break; }
            }

            if (act_scale) {
                // Dynamic quantized activation FC
                int64_t batch = 1;
                for (size_t i = 0; i < input_shape.size() - 1; ++i) batch *= input_shape[i];
                std::vector<int64_t> act_scale_shape = {batch, 1};
                ov::PartialShape act_s_ps(std::vector<ov::Dimension>(act_scale_shape.begin(), act_scale_shape.end()));
                cldnn::layout act_s_layout(act_s_ps, act_scale->dt, cldnn::format::bfyx);
                auto act_s_mem = engine.allocate_memory(act_s_layout);
                fill_memory_random(act_s_mem, *stream, act_scale->dt);

                cldnn::input_info act_scale_info("act_scale");
                cldnn::input_info act_zp_info("", 0);
                cldnn::input_info act_precomp_info("", 0);

                topology.add(cldnn::data("act_scale", act_s_mem));

                if (act_zp) {
                    cldnn::layout act_zp_layout(act_s_ps, act_zp->dt, cldnn::format::bfyx);
                    auto act_zp_mem = engine.allocate_memory(act_zp_layout);
                    fill_memory_random(act_zp_mem, *stream, act_zp->dt);
                    topology.add(cldnn::data("act_zp", act_zp_mem));
                    act_zp_info = cldnn::input_info("act_zp");
                }

                topology.add(cldnn::fully_connected("fc_prim",
                    cldnn::input_info("input"), "weights", "",
                    "decompression_scale", zp_id,
                    act_scale_info, act_zp_info, act_precomp_info,
                    output_dt, output_dim_size, weights_rank));
            } else {
                topology.add(cldnn::fully_connected("fc_prim",
                    cldnn::input_info("input"), "weights", "",
                    "decompression_scale", zp_id,
                    output_dt, output_dim_size, weights_rank));
            }
            last_prim_id = "fc_prim";
        } else {
            // Standard FC (no compression)
            topology.add(cldnn::fully_connected("fc_prim",
                cldnn::input_info("input"), "weights", "",
                output_dt, output_dim_size, weights_rank));
            last_prim_id = "fc_prim";
        }

        // Add post-ops to topology
        std::map<int, ref::elt_ref_data> elt_data_map;  // for accuracy check
        int post_op_idx = 0;
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
                    // Determine eltwise input shape based on broadcast spec
                    std::vector<int64_t> elt_shape;
                    // Output shape: same as FC output
                    // For per_oc: {1, OC} where OC = weight_shape[0]
                    // For per_tensor: {1, 1}
                    // For none: same as output = {batch, OC}
                    int64_t elt_oc = weight_shape.size() >= 2 ? weight_shape[weight_shape.size() - 2] : weight_shape[0];
                    if (po.elt_broadcast == broadcast_spec::per_oc) {
                        elt_shape = {1, elt_oc};
                    } else if (po.elt_broadcast == broadcast_spec::per_tensor) {
                        elt_shape = {1, 1};
                    } else {
                        // Same shape as output
                        elt_shape = input_shape;
                        elt_shape.back() = weight_shape[0];
                    }

                    std::string elt_data_id = "elt_data_" + std::to_string(post_op_idx);
                    ov::PartialShape elt_ps(std::vector<ov::Dimension>(elt_shape.begin(), elt_shape.end()));
                    cldnn::layout elt_layout(elt_ps, po.elt_dt, get_format_for_rank(elt_shape.size()));
                    auto elt_mem = engine.allocate_memory(elt_layout);
                    fill_memory_random(elt_mem, *stream, po.elt_dt);
                    topology.add(cldnn::data(elt_data_id, elt_mem));

                    // Save eltwise data for accuracy reference
                    elt_data_map[post_op_idx] = {read_memory_to_f32(elt_mem, *stream), elt_shape};

                    topology.add(cldnn::eltwise(po_id,
                        {cldnn::input_info(last_prim_id), cldnn::input_info(elt_data_id)},
                        map_eltwise_mode(po.elt_mode)));
                    last_prim_id = po_id;
                    break;
                }
                case post_op_kind::quantize: {
                    bool is_intermediate = (post_op_idx + 1 < static_cast<int>(post_ops.size()));
                    if (is_intermediate) {
                        // Intermediate quantize: use FakeQuantize to keep data in float.
                        // This avoids dtype round-trip (reorder i8->f32) mismatch vs reference.
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
                        std::string in_lo_id  = po_id + "_in_lo";
                        std::string in_hi_id  = po_id + "_in_hi";
                        std::string out_lo_id = po_id + "_out_lo";
                        std::string out_hi_id = po_id + "_out_hi";
                        make_scalar(in_lo_id, lo);
                        make_scalar(in_hi_id, hi);
                        make_scalar(out_lo_id, lo);
                        make_scalar(out_hi_id, hi);
                        auto fq_out_dt = (output_dt == cldnn::data_types::f32) ?
                            cldnn::data_types::f32 : cldnn::data_types::f16;
                        topology.add(cldnn::quantize(po_id,
                            cldnn::input_info(last_prim_id),
                            cldnn::input_info(in_lo_id), cldnn::input_info(in_hi_id),
                            cldnn::input_info(out_lo_id), cldnn::input_info(out_hi_id),
                            levels, fq_out_dt));
                        last_prim_id = po_id;
                    } else {
                        // Last quantize: reorder to actual target dtype
                        topology.add(cldnn::reorder(po_id,
                            cldnn::input_info(last_prim_id),
                            fmt, po.quant_out_dt));
                        last_prim_id = po_id;
                    }
                    break;
                }
                case post_op_kind::swiglu: {
                    // SwiGLU is handled specially; skip for now
                    // TODO: add swiglu primitive support
                    break;
                }
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

        // Build network - catch unimplemented cases (e.g., 4D matmul not supported)
        std::unique_ptr<cldnn::network> network_ptr;
        try {
            network_ptr = std::make_unique<cldnn::network>(engine, topology, exec_config);
        } catch (const std::exception& e) {
            std::string msg = e.what();
            if (msg.find("not implemented") != std::string::npos ||
                msg.find("not supported") != std::string::npos ||
                msg.find("Failed to select implementation") != std::string::npos) {
                throw bench_unimplemented(msg);
            }
            throw;
        }
        auto& network = *network_ptr;
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

            // Decompress weights if compressed, otherwise use raw
            std::vector<float> ref_weights;
            if (compressed_weights) {
                ref_weights = ref::decompress_weights(
                    weight_f32_saved, weight_shape,
                    scale_f32_saved, saved_scale_shape,
                    zp_f32_saved, saved_zp_shape);
            } else {
                ref_weights = weight_f32_saved;
            }

            auto ref_out = ref::fc(input_f32_saved, input_shape, ref_weights, weight_shape);

            // FC output shape: same rank as input with last dim = OC
            // For 2D weight [OC, IC] or 3D weight [batch, OC, IC], OC is at dim [-2]
            int64_t fc_oc = weight_shape.size() >= 2 ? weight_shape[weight_shape.size() - 2] : weight_shape[0];
            std::vector<int64_t> out_shape = input_shape;
            out_shape.back() = fc_oc;

            // Apply post-ops to reference
            bool can_check = ref::apply_post_ops_ref(ref_out, post_ops, out_shape, elt_data_map);
            if (can_check) {
                float atol, rtol;
                get_default_tolerance(input_dt, atol, rtol);
                // For mixed int-input / float-output, widen tolerance
                if (output_dt != input_dt) {
                    float oatol, ortol;
                    get_default_tolerance(output_dt, oatol, ortol);
                    atol = std::max(atol, oatol);
                    rtol = std::max(rtol, ortol);
                }
                if (compressed_weights) {
                    atol = std::max(atol, 1.0f);
                    rtol = std::max(rtol, 0.5f);
                }
                int quant_count = static_cast<int>(std::count_if(post_ops.begin(), post_ops.end(),
                    [](const post_op_entry& po) { return po.kind == post_op_kind::quantize; }));
                if (quant_count > 0) { atol = std::max(atol, static_cast<float>(quant_count)); }
                {
                    float threshold = (compressed_weights || quant_count > 0) ? 0.01f : 0.0f;
                    acc_res = compare_f32(gpu_out, ref_out, atol, rtol, threshold);
                    has_acc = true;
                    reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
                    if (!acc_res.pass) {
                        test_passed = false;
                    }
                }
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
        print_result(network, "fc_prim", config, test_passed, acc_unimplemented, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed, acc_unimplemented);
    }
};

REGISTER_KERNEL(bench_fc)

}  // namespace bench_kernel
