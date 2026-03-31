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
#include <intel_gpu/primitives/space_to_batch.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_space_to_batch : public kernel_base {
public:
    std::string name() const override { return "space_to_batch"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: input:block_shape:pads_begin:pads_end
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 4) throw std::runtime_error("space_to_batch requires 4 shapes (input:block:pads_begin:pads_end). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        cldnn::layout input_layout(ps, dt, cldnn::format::bfyx);
        auto input_mem = engine.allocate_memory(input_layout);
        fill_memory_random(input_mem, *stream, dt);

        auto to_tensor = [](const std::vector<int64_t>& v) -> cldnn::tensor {
            std::vector<int64_t> v4 = v;
            v4.resize(4, 1);
            return cldnn::tensor(cldnn::batch(v4[0]), cldnn::feature(v4[1]),
                                 cldnn::spatial(v4[3], v4[2]));
        };

        cldnn::tensor block_t = to_tensor(shapes[1]);
        cldnn::tensor pb_t = to_tensor(shapes[2]);
        cldnn::tensor pe_t = to_tensor(shapes[3]);

        // Compute output size
        auto& in_s = shapes[0];
        auto& block_s = shapes[1];
        auto& pb = shapes[2];
        auto& pe = shapes[3];
        std::vector<int64_t> out_s(in_s.size());
        int64_t batch_mult = 1;
        for (size_t i = 1; i < in_s.size(); ++i) batch_mult *= (i < block_s.size() ? block_s[i] : 1);
        out_s[0] = in_s[0] * batch_mult;
        for (size_t i = 1; i < in_s.size(); ++i) {
            out_s[i] = (in_s[i] + (i < pb.size() ? pb[i] : 0) + (i < pe.size() ? pe[i] : 0)) / (i < block_s.size() ? block_s[i] : 1);
        }
        cldnn::tensor out_t = to_tensor(out_s);

        cldnn::topology topology;
        topology.add(cldnn::input_layout("input", input_layout));
        topology.add(cldnn::space_to_batch("prim", cldnn::input_info("input"),
                     block_t, pb_t, pe_t, out_t));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("input", input_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto input_f32 = read_memory_to_f32(input_mem, *stream);
            auto ref_out = ref::space_to_batch(input_f32, shapes[0], shapes[1], shapes[2], shapes[3]);
            float atol, rtol;
            get_default_tolerance(dt, atol, rtol);
            acc_res = compare_f32(gpu_out, ref_out, atol, rtol);
            reported_acc_ = {true, acc_res.total_elements, acc_res.mismatches, acc_res.max_abs_diff, acc_res.max_rel_diff};
            if (!acc_res.pass) test_passed = false;
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

REGISTER_KERNEL(bench_space_to_batch)

}  // namespace bench_kernel
