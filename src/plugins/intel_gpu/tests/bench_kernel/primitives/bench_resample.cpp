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
#include <intel_gpu/primitives/resample.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Resample (Interpolate) kernel benchmark
//
// Usage:
//   --resample --dt=f16 --shapes=1x64x32x32 --resample_sizes=1:64:64:64
//       --resample_mode=0  (0=nearest, 1=linear, 2=cubic)
//
// Resizes input tensor using interpolation.
// ============================================================================

class bench_resample : public kernel_base {
public:
    std::string name() const override { return "resample"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) {
            throw std::runtime_error("Resample requires at least 1 shape. Got: " + config.shapes_str);
        }

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto& in_shape = shapes[0];
        size_t rank = in_shape.size();

        // Parse output sizes
        std::vector<int64_t> out_sizes;
        if (!config.resample_sizes.empty()) {
            out_sizes = parse_colon_vec(config.resample_sizes);
        }
        if (out_sizes.empty()) {
            // Default: double spatial dims
            out_sizes = std::vector<int64_t>(in_shape.begin(), in_shape.end());
            for (size_t d = 2; d < rank; ++d) out_sizes[d] *= 2;
        } else if (out_sizes.size() < rank) {
            // Sizes are for spatial dims only; prepend batch/channel from input shape
            size_t num_spatial = out_sizes.size();
            std::vector<int64_t> full_sizes(in_shape.begin(), in_shape.end());
            for (size_t d = 0; d < num_spatial; ++d) {
                full_sizes[rank - num_spatial + d] = out_sizes[d];
            }
            out_sizes = full_sizes;
        }
        out_sizes.resize(rank, 1);

        // Compute scales
        std::vector<float> scales(rank);
        for (size_t d = 0; d < rank; ++d) {
            scales[d] = (in_shape[d] > 0) ? static_cast<float>(out_sizes[d]) / in_shape[d] : 1.0f;
        }

        // Axes: all dims
        std::vector<int64_t> axes(rank);
        for (size_t d = 0; d < rank; ++d) axes[d] = d;

        // Interpolation mode
        auto mode = cldnn::resample::InterpolateOp::InterpolateMode::NEAREST;
        if (config.resample_mode == 1) mode = cldnn::resample::InterpolateOp::InterpolateMode::LINEAR_ONNX;
        else if (config.resample_mode == 2) mode = cldnn::resample::InterpolateOp::InterpolateMode::CUBIC;

        auto exec_config = make_exec_config(config, "resample_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(ps, dt, get_input_format(config, 0, in_shape.size()));
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::resample("resample_prim",
            cldnn::input_info("input"),
            out_sizes, scales, axes,
            {0}, {0},  // pads_begin, pads_end
            0,  // antialias
            -0.75f,  // cube_coeff
            mode,
            cldnn::resample::InterpolateOp::ShapeCalcMode::SIZES,
            cldnn::resample::InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
            cldnn::resample::InterpolateOp::NearestMode::ROUND_PREFER_FLOOR));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;
        bool has_perf = false;

        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "resample_prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);

            // Reference: nearest-neighbor interpolation only
            if (mode == cldnn::resample::InterpolateOp::InterpolateMode::NEAREST) {
                size_t out_total = 1;
                for (auto d : out_sizes) out_total *= d;

                std::vector<size_t> in_strides(rank, 1), out_strides(rank, 1);
                for (int d = (int)rank - 2; d >= 0; --d) {
                    in_strides[d] = in_strides[d+1] * in_shape[d+1];
                    out_strides[d] = out_strides[d+1] * out_sizes[d+1];
                }

                std::vector<float> ref_out(out_total);
                for (size_t idx = 0; idx < out_total; ++idx) {
                    size_t in_idx = 0, rem = idx;
                    for (size_t d = 0; d < rank; ++d) {
                        size_t coord = rem / out_strides[d];
                        rem %= out_strides[d];
                        size_t in_coord = static_cast<size_t>(
                            std::min((int64_t)(coord / scales[d]), in_shape[d] - 1));
                        in_idx += in_coord * in_strides[d];
                    }
                    ref_out[idx] = input_f32[in_idx];
                }

                float atol, rtol;
                get_default_tolerance(dt, atol, rtol);
                auto acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
                reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
                if (!acc_res.pass) test_passed = false;
            } else if (mode == cldnn::resample::InterpolateOp::InterpolateMode::LINEAR_ONNX && rank == 4) {
                // Bilinear interpolation reference (4D: N,C,H,W)
                int64_t N = in_shape[0], C = in_shape[1], IH = in_shape[2], IW = in_shape[3];
                int64_t OH = out_sizes[2], OW = out_sizes[3];
                size_t out_total = static_cast<size_t>(N * C * OH * OW);
                std::vector<float> ref_out(out_total);

                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t oh = 0; oh < OH; ++oh) {
                            for (int64_t ow = 0; ow < OW; ++ow) {
                                // half_pixel coordinate transform
                                float iy = (OH > 1) ? ((oh + 0.5f) / scales[2] - 0.5f) : 0.0f;
                                float ix = (OW > 1) ? ((ow + 0.5f) / scales[3] - 0.5f) : 0.0f;

                                int64_t iy0 = static_cast<int64_t>(std::floor(iy));
                                int64_t ix0 = static_cast<int64_t>(std::floor(ix));
                                int64_t iy1 = iy0 + 1;
                                int64_t ix1 = ix0 + 1;
                                float dy = iy - iy0;
                                float dx = ix - ix0;

                                auto clamp = [](int64_t v, int64_t lo, int64_t hi) {
                                    return std::max(lo, std::min(v, hi));
                                };
                                int64_t y0 = clamp(iy0, (int64_t)0, IH - 1);
                                int64_t y1 = clamp(iy1, (int64_t)0, IH - 1);
                                int64_t x0 = clamp(ix0, (int64_t)0, IW - 1);
                                int64_t x1 = clamp(ix1, (int64_t)0, IW - 1);

                                size_t base = (n * C + c) * IH;
                                float v00 = input_f32[(base + y0) * IW + x0];
                                float v01 = input_f32[(base + y0) * IW + x1];
                                float v10 = input_f32[(base + y1) * IW + x0];
                                float v11 = input_f32[(base + y1) * IW + x1];

                                float val = v00 * (1-dy) * (1-dx) + v01 * (1-dy) * dx
                                          + v10 * dy * (1-dx)     + v11 * dy * dx;
                                ref_out[((n * C + c) * OH + oh) * OW + ow] = val;
                            }
                        }
                    }
                }

                float atol, rtol;
                get_default_tolerance(dt, atol, rtol);
                atol *= 2.0f;  // bilinear has more rounding
                auto acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
                reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
                if (!acc_res.pass) test_passed = false;
            } else {
                throw bench_unimplemented();
            }
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            has_perf = true;
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "resample_prim", config, test_passed, false, wall_ms,
                     nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_resample)

}  // namespace bench_kernel
