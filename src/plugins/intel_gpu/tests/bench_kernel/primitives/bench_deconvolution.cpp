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
#include <intel_gpu/primitives/deconvolution.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

// ============================================================================
// Deconvolution (ConvTranspose) kernel benchmark
//
// Usage:
//   --deconvolution --dt=f16 --shapes=1x64x16x16:64x32x3x3
//       --strides=2:2 --groups=1
//
// shapes = input:weights (BFYX : OC_IC_KH_KW)
// Output size is computed from input/weights/strides/padding.
// ============================================================================

class bench_deconvolution : public kernel_base {
public:
    std::string name() const override { return "deconvolution"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) {
            throw std::runtime_error("Deconvolution requires 2 shapes (input:weights). Got: " + config.shapes_str);
        }

        auto dts = config.data_types;
        cldnn::data_types input_dt = dts.size() > 0 ? dts[0] : cldnn::data_types::f16;
        cldnn::data_types weight_dt = dts.size() > 1 ? dts[1] : input_dt;

        auto& in_shape = shapes[0];
        auto& w_shape = shapes[1];

        uint32_t groups = std::max(config.groups, 1);
        size_t spatial_dims = in_shape.size() - 2;

        // Parse strides, dilations, padding
        std::vector<size_t> strides_v(spatial_dims, 1);
        std::vector<size_t> dilations_v(spatial_dims, 1);
        std::vector<ptrdiff_t> pad_begin(spatial_dims, 0);
        std::vector<ptrdiff_t> pad_end(spatial_dims, 0);

        if (!config.strides.empty()) {
            auto sv = parse_x_vec(config.strides);
            for (size_t i = 0; i < std::min(sv.size(), strides_v.size()); ++i) strides_v[i] = sv[i];
        }
        if (!config.dilations.empty()) {
            auto dv = parse_x_vec(config.dilations);
            for (size_t i = 0; i < std::min(dv.size(), dilations_v.size()); ++i) dilations_v[i] = dv[i];
        }
        if (!config.padding_begin.empty()) {
            auto pv = parse_x_vec(config.padding_begin);
            for (size_t i = 0; i < std::min(pv.size(), pad_begin.size()); ++i)
                pad_begin[i] = static_cast<ptrdiff_t>(pv[i]);
        }
        if (!config.padding_end.empty()) {
            auto pv = parse_x_vec(config.padding_end);
            for (size_t i = 0; i < std::min(pv.size(), pad_end.size()); ++i)
                pad_end[i] = static_cast<ptrdiff_t>(pv[i]);
        }

        auto exec_config = make_exec_config(config, "deconv_prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape in_ps(std::vector<ov::Dimension>(in_shape.begin(), in_shape.end()));
        cldnn::layout input_layout(in_ps, input_dt, get_input_format(config, 0, in_shape.size()));
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, input_dt);

        ov::PartialShape w_ps(std::vector<ov::Dimension>(w_shape.begin(), w_shape.end()));
        cldnn::layout weight_layout(w_ps, weight_dt, get_input_format(config, 1, w_shape.size()));
        auto weight_mem = engine.allocate_memory(weight_layout);
        fill_memory_random(weight_mem, *stream, weight_dt);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::data("weights", weight_mem));

        ov::Strides str(strides_v.begin(), strides_v.end());
        ov::CoordinateDiff pb(pad_begin.begin(), pad_begin.end());
        ov::CoordinateDiff pe(pad_end.begin(), pad_end.end());
        ov::Strides dil(dilations_v.begin(), dilations_v.end());
        ov::CoordinateDiff out_pad(spatial_dims, 0);

        if (groups > 1) {
            topology.add(cldnn::deconvolution("deconv_prim",
                cldnn::input_info("input"), "weights", "",
                static_cast<uint32_t>(groups), str, pb, dil,
                pb, pe, out_pad,
                config.grouped_weights_shape != 0));
        } else {
            topology.add(cldnn::deconvolution("deconv_prim",
                cldnn::input_info("input"), "weights",
                str, pb, dil));
        }

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;
        bool has_perf = false;

        [[maybe_unused]] acc_result acc_res;
        if (config.is_acc()) {
            throw bench_unimplemented("CPU reference not implemented");
        }

        if (config.is_perf()) {
            run_perf(network, config, timer);
            has_perf = true;
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "deconv_prim", config, test_passed, false, wall_ms,
                     nullptr, has_perf ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_deconvolution)

}  // namespace bench_kernel
