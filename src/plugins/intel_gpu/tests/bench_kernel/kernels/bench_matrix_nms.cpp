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
#include <intel_gpu/primitives/matrix_nms.hpp>

#include "kernel_base.hpp"
#include "kernel_registry.hpp"
#include "common/bench_config.hpp"
#include "common/bench_timer.hpp"
#include "common/bench_types.hpp"
#include "common/bench_utils.hpp"
#include "common/bench_reference.hpp"

namespace bench_kernel {

class bench_matrix_nms : public kernel_base {
public:
    std::string name() const override { return "matrix_nms"; }

    void run_single(cldnn::engine& engine, const bench_config& config) override {
        // shapes: boxes[N,M,4] : scores[N,C,M]
        auto shapes = parse_shapes(config.shapes_str);
        if (shapes.size() < 2) throw std::runtime_error("matrix_nms requires 2 shapes (boxes:scores). Got: " + config.shapes_str);

        auto dt = config.data_types.empty() ? cldnn::data_types::f16 : config.data_types[0];

        auto exec_config = make_exec_config(config, "prim");
        auto stream = engine.create_stream(exec_config);

        ov::PartialShape ps0(std::vector<ov::Dimension>(shapes[0].begin(), shapes[0].end()));
        ov::PartialShape ps1(std::vector<ov::Dimension>(shapes[1].begin(), shapes[1].end()));
        cldnn::layout lay0(ps0, dt, cldnn::format::bfyx);
        cldnn::layout lay1(ps1, dt, cldnn::format::bfyx);
        auto boxes_mem = engine.allocate_memory(lay0);
        auto scores_mem = engine.allocate_memory(lay1);
        fill_memory_random(boxes_mem, *stream, dt);
        fill_memory_random(scores_mem, *stream, dt);

        ov::op::v8::MatrixNms::Attributes attrs;
        attrs.nms_top_k = config.nms_top_k;
        attrs.score_threshold = config.nms_score_threshold;
        attrs.keep_top_k = -1;
        attrs.background_class = -1;

        cldnn::topology topology;
        topology.add(cldnn::input_layout("boxes", lay0));
        topology.add(cldnn::input_layout("scores", lay1));
        topology.add(cldnn::matrix_nms("prim",
            cldnn::input_info("boxes"), cldnn::input_info("scores"), attrs));

        cldnn::network network(engine, topology, exec_config);
        network.set_input_data("boxes", boxes_mem);
        network.set_input_data("scores", scores_mem);

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

REGISTER_KERNEL(bench_matrix_nms)

}  // namespace bench_kernel
