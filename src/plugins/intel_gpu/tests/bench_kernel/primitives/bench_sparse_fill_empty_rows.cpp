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
#include <intel_gpu/primitives/sparse_fill_empty_rows.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_sparse_fill_empty_rows : public kernel_base {
public:
    std::string name() const override { return "sparse_fill_empty_rows"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: values[N] : dense_shape[2] : indices[N,2] : default_value[1]
        // (order matches shape inference: input[0]=values, input[1]=dense_shape, input[2]=indices, input[3]=default_value)
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 4) throw std::runtime_error("sparse_fill_empty_rows requires 4 shapes. Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        ov::PartialShape ps2(std::vector<ov::Dimension>(shapes[2].begin(), shapes[2].end()));
        ov::PartialShape ps3(std::vector<ov::Dimension>(shapes[3].begin(), shapes[3].end()));
        cldnn::layout lay0(ps0, cldnn::data_types::i64, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        cldnn::layout lay2(ps2, cldnn::data_types::i64, cldnn::format::bfyx);
        // default_value must be scalar (rank 0)
        cldnn::layout lay3(ov::PartialShape{}, dt, cldnn::format::bfyx);

        auto indices_mem = engine.allocate_memory(lay0);
        auto values_mem = engine.allocate_memory(lay1);
        auto dshape_mem = engine.allocate_memory(lay2);
        auto defval_mem = engine.allocate_memory(lay3);
        // Bounded indices to [0, dense_shape[dim]-1]
        {
            int64_t num_rows = 5;  // dense_shape[0]
            int64_t num_cols = 3;  // dense_shape[1]
            cldnn::mem_lock<int64_t> idx(indices_mem, *stream);
            for (size_t i = 0; i < idx.size(); i += 2) {
                idx[i] = static_cast<int64_t>(i / 2) % num_rows;      // row index
                idx[i + 1] = static_cast<int64_t>(i / 2) % num_cols;  // col index
            }
        }
        fill_memory_random(values_mem, *stream, dt);
        { cldnn::mem_lock<int64_t> ds(dshape_mem, *stream); ds[0] = 5; ds[1] = 3; }
        fill_memory_random(defval_mem, *stream, dt);

        // Shape inference expects: values[0], dense_shape[1], indices[2], default_value[3]
        // shapes: values[N] : dense_shape[2] : indices[N,2] : default_value[1]
        std::vector<cldnn::input_info> inputs = {
            cldnn::input_info("values"), cldnn::input_info("dense_shape"),
            cldnn::input_info("indices"), cldnn::input_info("default_value")
        };

        cldnn::topology topology;
        topology.add(cldnn::input_layout("values", lay1));
        topology.add(cldnn::input_layout("dense_shape", lay2));
        topology.add(cldnn::input_layout("indices", lay0));
        topology.add(cldnn::input_layout("default_value", lay3));
        topology.add(cldnn::sparse_fill_empty_rows("prim", inputs));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("indices", indices_mem);
        network.set_input_data("values", values_mem);
        network.set_input_data("dense_shape", dshape_mem);
        network.set_input_data("default_value", defval_mem);

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

REGISTER_KERNEL(bench_sparse_fill_empty_rows)

}  // namespace bench_kernel
