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
#include <intel_gpu/primitives/embedding_bag.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_embedding_bag : public kernel_base {
public:
    std::string name() const override { return "embedding_bag"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: emb_table[V, D]:indices[N, ...]:per_sample_weights[N, ...]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("embedding_bag requires at least 2 shapes (emb_table:indices). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];
        auto eb_type = cldnn::embedding_bag::packed_sum;  // default

        // Compute output shape: [indices.shape[0], D]
        int64_t D = shapes[0].back();
        int64_t N = shapes[1][0];
        std::vector<int64_t> out_s = {N, D};
        while (out_s.size() < 4) out_s.insert(out_s.begin(), 1);
        cldnn::tensor out_t(cldnn::batch(out_s[0]), cldnn::feature(out_s[1]),
                            cldnn::spatial(out_s[3], out_s[2]));

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, cldnn::data_types::i32, cldnn::format::bfyx);
        auto emb_mem = engine.allocate_memory(lay0);
        auto idx_mem = engine.allocate_memory(lay1);
        fill_memory_random(emb_mem, *stream, dt);
        // Bounded indices: [0, V-1] where V = vocab_size = shapes[0][0]
        {
            int32_t V = static_cast<int32_t>(shapes[0][0]);
            cldnn::mem_lock<int32_t> idx(idx_mem, *stream);
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = static_cast<int32_t>(i % V);
        }

        cldnn::topology topology;
        topology.add(cldnn::input_layout("emb_table", lay0));
        topology.add(cldnn::input_layout("indices", lay1));
        topology.add(cldnn::embedding_bag("prim",
            {cldnn::input_info("emb_table"), cldnn::input_info("indices")},
            eb_type, out_t));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("emb_table", emb_mem);
        network.set_input_data("indices", idx_mem);

        auto wall_start = std::chrono::high_resolution_clock::now();
        bool test_passed = true;
        perf_timer timer;

        acc_result acc_res;
        if (config.is_acc()) {
            auto outputs = network.execute();
            auto gpu_out = read_network_output_f32(outputs, "prim", *stream);
            auto table_f32 = read_memory_to_f32(emb_mem, *stream);
            auto indices_f32 = read_memory_to_f32(idx_mem, *stream);
            auto ref_out = ref::embedding_bag_packed(table_f32, shapes[0], indices_f32, shapes[1]);
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

REGISTER_KERNEL(bench_embedding_bag)

}  // namespace bench_kernel
