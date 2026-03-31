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
#include <intel_gpu/primitives/strided_slice.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// StridedSlice kernel benchmark
//
// Usage:
//   --strided_slice --dt=f16 --shapes=input_shape:output_shape
//
// Performs a strided slice from offset 0 with stride 1 producing output_shape.
// This mimics the most common LLM usage (e.g., slicing QKV projections).
//
// Example:
//   --strided_slice --dt=f16 --shapes=1x128x5120:1x128x1024
//   --strided_slice --dt=f16 --shapes=1x8x128x128:1x8x128x32
// ============================================================================

class bench_strided_slice : public kernel_base {
public:
    std::string name() const override { return "strided_slice"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("StridedSlice requires 2 shapes (input:output). Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;

        auto exec_config = make_exec_config(config, "strided_slice_prim");
        auto stream = engine.create_stream(exec_config);

        auto& input_shape = shapes[0];
        auto& output_shape = shapes[1];
        size_t input_rank = input_shape.size();

        // Input memory
        ov::PartialShape input_ps(std::vector<ov::Dimension>(input_shape.begin(), input_shape.end()));
        cldnn::layout input_layout_desc(input_ps, dt, get_input_format(config, 0, input_rank));
        auto input_mem = engine.allocate_memory(input_layout_desc);
        fill_memory_random(input_mem, *stream, dt);

        // Parse begin/end/strides (in original axis space, including new axes)
        // The orig_rank may differ from input_rank when new_axis_mask is used
        std::vector<int64_t> begin_vec, end_vec, strides_vec;
        std::vector<int64_t> begin_mask_vec, end_mask_vec;
        std::vector<int64_t> new_axis_mask_vec, shrink_axis_mask_vec;

        if (!config.ss_begin.empty()) {
            begin_vec = parse_colon_vec(config.ss_begin);
        }
        if (!config.ss_end.empty()) {
            end_vec = parse_colon_vec(config.ss_end);
        }
        if (!config.ss_strides.empty()) {
            strides_vec = parse_colon_vec(config.ss_strides);
        }
        if (!config.begin_mask.empty()) {
            begin_mask_vec = parse_colon_vec(config.begin_mask);
        }
        if (!config.end_mask.empty()) {
            end_mask_vec = parse_colon_vec(config.end_mask);
        }
        if (!config.new_axis_mask.empty()) {
            new_axis_mask_vec = parse_colon_vec(config.new_axis_mask);
        }
        if (!config.shrink_axis_mask.empty()) {
            shrink_axis_mask_vec = parse_colon_vec(config.shrink_axis_mask);
        }

        // Determine the original axis rank (the space of begin/end/strides)
        size_t orig_rank = begin_vec.size();
        if (orig_rank == 0) orig_rank = input_rank;

        // Pad vectors to orig_rank
        begin_vec.resize(orig_rank, 0);
        end_vec.resize(orig_rank, 0);
        strides_vec.resize(orig_rank, 1);
        begin_mask_vec.resize(orig_rank, 0);
        end_mask_vec.resize(orig_rank, 0);
        new_axis_mask_vec.resize(orig_rank, 0);
        shrink_axis_mask_vec.resize(orig_rank, 0);
        std::vector<int64_t> ellipsis_mask(orig_rank, 0);

        if (config.new_axis_mask.empty() && orig_rank > input_rank) {
            size_t implied_new_axes = orig_rank - input_rank;
            for (size_t i = 0; i < implied_new_axes; ++i) {
                new_axis_mask_vec[i] = 1;
            }
        }

        const bool has_explicit_end = !config.ss_end.empty();

        // Process dimensions: for each original axis, determine if it maps to an
        // input dimension or is a new axis, and compute effective begin/stride/size
        struct dim_info {
            size_t input_dim;   // index into input_shape (SIZE_MAX if new axis)
            int64_t eff_begin;  // effective start coordinate in input
            int64_t eff_stride; // effective stride
            int64_t eff_size;   // number of elements along this dim
            bool is_shrink;     // removed from output due to shrink_axis_mask
        };
        std::vector<dim_info> all_dims;
        size_t cur_input_dim = 0;

        for (size_t d = 0; d < orig_rank; ++d) {
            dim_info info;
            bool is_new = new_axis_mask_vec[d] != 0;
            bool is_shrink = shrink_axis_mask_vec[d] != 0;

            if (is_new) {
                info.input_dim = SIZE_MAX;
                info.eff_begin = 0;
                info.eff_stride = 1;
                info.eff_size = 1;
                info.is_shrink = false;
            } else {
                info.input_dim = cur_input_dim;
                int64_t dim_size = (cur_input_dim < input_rank) ? input_shape[cur_input_dim] : 1;
                int64_t s = strides_vec[d];
                if (s == 0) s = 1;

                // Effective begin
                int64_t b;
                if (begin_mask_vec[d] != 0) {
                    b = (s > 0) ? 0 : (dim_size - 1);
                } else {
                    b = begin_vec[d];
                    if (b < 0) b += dim_size;
                    if (s > 0) {
                        b = std::max<int64_t>(0, std::min(b, dim_size - 1));
                    } else {
                        b = std::max<int64_t>(0, std::min(b, dim_size - 1));
                    }
                }

                // Effective end
                int64_t e;
                if (!has_explicit_end || end_mask_vec[d] != 0) {
                    e = (s > 0) ? dim_size : -1;
                } else {
                    e = end_vec[d];
                    if (e < 0) e += dim_size;
                    if (s > 0) {
                        e = std::max<int64_t>(0, std::min(e, dim_size));
                    } else {
                        e = std::max<int64_t>(-1, std::min(e, dim_size - 1));
                    }
                }

                // Compute size
                int64_t sz;
                if (s > 0) {
                    sz = (e > b) ? ((e - b + s - 1) / s) : 0;
                } else {
                    sz = (b > e) ? ((b - e + (-s) - 1) / (-s)) : 0;
                }
                if (is_shrink) sz = 1;

                info.eff_begin = b;
                info.eff_stride = s;
                info.eff_size = sz;
                info.is_shrink = is_shrink;
                cur_input_dim++;
            }
            all_dims.push_back(info);
        }

        // Handle remaining input dims beyond orig_rank (pass through as identity)
        while (cur_input_dim < input_rank) {
            dim_info info;
            info.input_dim = cur_input_dim;
            info.eff_begin = 0;
            info.eff_stride = 1;
            info.eff_size = input_shape[cur_input_dim];
            info.is_shrink = false;
            all_dims.push_back(info);
            cur_input_dim++;
        }

        // Build the "full" iteration shape (all dims, before shrink removal)
        std::vector<int64_t> full_shape;
        for (auto& di : all_dims) {
            full_shape.push_back(di.eff_size);
        }
        size_t full_rank = all_dims.size();

        // Create begin/end/strides data memories (i32)
        auto make_i32_data = [&](const std::string& name, const std::vector<int64_t>& vals) {
            ov::PartialShape ps({static_cast<int64_t>(vals.size())});
            cldnn::layout layout(ps, cldnn::data_types::i32, cldnn::format::bfyx);
            auto mem = engine.allocate_memory(layout);
            {
                cldnn::mem_lock<int32_t> lock(mem, *stream);
                for (size_t i = 0; i < vals.size(); ++i) {
                    lock[i] = static_cast<int32_t>(vals[i]);
                }
            }
            return mem;
        };

        auto begin_mem = make_i32_data("begin", begin_vec);
        auto end_mem = make_i32_data("end", end_vec);
        auto strides_mem = make_i32_data("strides", strides_vec);

        ov::Shape out_shape_ov(output_shape.begin(), output_shape.end());

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout_desc));
        topology.add(cldnn::data("begin", begin_mem));
        topology.add(cldnn::data("end", end_mem));
        topology.add(cldnn::data("strides", strides_mem));

        auto ss_prim = cldnn::strided_slice("strided_slice_prim",
            {cldnn::input_info("input"), cldnn::input_info("begin"),
             cldnn::input_info("end"), cldnn::input_info("strides")},
            begin_vec,
            end_vec,
            strides_vec,
            begin_mask_vec,
            end_mask_vec,
            new_axis_mask_vec,
            shrink_axis_mask_vec,
            ellipsis_mask,
            out_shape_ov);

        topology.add(ss_prim);

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
            auto gpu_out = read_network_output_f32(outputs, "strided_slice_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);

            // Compute reference: iterate over full shape, map each coord to input
            // Compute input strides (for flat indexing)
            std::vector<int64_t> in_strides(input_rank, 1);
            for (int i = static_cast<int>(input_rank) - 2; i >= 0; --i) {
                in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
            }

            // Compute full output strides for multi-dim iteration
            std::vector<int64_t> full_strides(full_rank, 1);
            for (int i = static_cast<int>(full_rank) - 2; i >= 0; --i) {
                full_strides[i] = full_strides[i + 1] * full_shape[i + 1];
            }

            size_t out_total = 1;
            for (auto s : output_shape) out_total *= s;

            size_t full_total = 1;
            for (auto s : full_shape) full_total *= s;

            size_t in_total = 1;
            for (auto s : input_shape) in_total *= s;

            std::vector<float> ref_out(out_total);

            for (size_t fi = 0; fi < full_total; ++fi) {
                // Decompose flat index into per-dim coordinates in full shape
                size_t in_idx = 0;
                bool valid = true;
                size_t rem = fi;
                for (size_t d = 0; d < full_rank; ++d) {
                    int64_t coord = rem / full_strides[d];
                    rem %= full_strides[d];

                    if (all_dims[d].input_dim == SIZE_MAX) {
                        // New axis - no input dimension, skip
                        continue;
                    }
                    size_t idim = all_dims[d].input_dim;
                    int64_t in_coord = all_dims[d].eff_begin + coord * all_dims[d].eff_stride;
                    if (in_coord < 0 || in_coord >= input_shape[idim]) {
                        valid = false;
                        break;
                    }
                    in_idx += in_coord * in_strides[idim];
                }

                if (fi < out_total) {
                    ref_out[fi] = (valid && in_idx < in_total) ? input_f32[in_idx] : 0.0f;
                }
            }

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
        print_result(network, "strided_slice_prim", config, test_passed, false, wall_ms,
                     has_acc ? &acc_res : nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_strided_slice)

}  // namespace bench_kernel
