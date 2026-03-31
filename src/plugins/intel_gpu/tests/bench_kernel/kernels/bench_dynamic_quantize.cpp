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
#include <intel_gpu/primitives/dynamic_quantize.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_dynamic_quantize : public kernel_base {
public:
    std::string name() const override { return "dynamic_quantize"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.empty()) throw std::runtime_error("dynamic_quantize requires 1 shape. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        auto data_mem = engine.allocate_memory(lay0);
        fill_memory_random(data_mem, *stream, dt);

        // Create default symmetric quantization attributes
        cldnn::dynamic_quantize::Attributes attrs;
        attrs.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        attrs.quantization_dt = ov::element::i8;
        attrs.scale_dt = ov::element::f16;
        attrs.zp_dt = ov::element::dynamic;
        attrs.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        // Default group sizes: quantize along last dimension
        for (size_t i = 0; i < shapes[0].size(); ++i) {
            if (i < shapes[0].size() - 1)
                attrs.group_sizes.push_back(1);
            else
                attrs.group_sizes.push_back(static_cast<uint64_t>(shapes[0][i]));
        }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("data", lay0));
        topology.add(cldnn::dynamic_quantize("prim",
            cldnn::input_info("data"), attrs, shapes[0].size()));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("data", data_mem);

        bool test_passed = true;
        perf_timer timer;
        auto wall_start = std::chrono::high_resolution_clock::now();

        [[maybe_unused]] acc_result acc_res;
        if (config.is_acc()) {
            throw bench_unimplemented("CPU reference not implemented");
        }
        if (config.is_perf()) {
            run_perf(network, config, timer);
            reported_timer_ = timer;
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        print_result(network, "prim", config, test_passed, false, wall_ms,
                     nullptr, !timer.empty() ? &timer : nullptr);
        finalize_result(test_passed);
    }
};

REGISTER_KERNEL(bench_dynamic_quantize)

}  // namespace bench_kernel
