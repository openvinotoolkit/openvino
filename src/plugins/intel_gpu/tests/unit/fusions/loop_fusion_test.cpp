// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/loop.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct loop_params {
    size_t loop_trip_count;
    tensor in_shape;
    tensor loop_input_shape;
    tensor loop_eltwise_shape;
    std::vector<uint16_t> permute_order;
    data_types data_type;
    format default_format;
    data_types default_type;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};


program::ptr build_program(engine& engine,
                            topology& body_topology,
                            primitive_id initial_condition_id,
                            std::vector<loop::io_primitive_map> output_primitive_maps,
                            std::vector<loop::backedge_mapping> back_edges) {
    std::vector<primitive_id> output_names_vec;
    for (auto out_map : output_primitive_maps) {
        output_names_vec.push_back(out_map.internal_id.pid);
    }

    // setup outputs for backedges
    for (auto& back_edge : back_edges) {
        output_names_vec.push_back(back_edge.from);
    }

    // if execution_condition_id is specified, we need to add the id in build_option::outputs
    if (!initial_condition_id.empty()) {
        output_names_vec.push_back(initial_condition_id);
    }

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::custom_outputs(output_names_vec));

    return program::build_program(engine, body_topology, config, false, false, true);
}

class LoopFusingTest : public ::BaseFusingTest<loop_params> {
public:

    void execute(loop_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p, true);
    }

    layout get_input_layout(loop_params& p) {
        return layout{ p.data_type, p.default_format, p.in_shape, padding{} };
    }
};

class permute_eltwise_loop: public LoopFusingTest {};
TEST_P(permute_eltwise_loop, basic) {
    auto p = GetParam();
    auto num_iteration_mem = engine.allocate_memory({data_types::i64, format::bfyx, {1, 1, 1, 1}});
    auto trip_count_mem = engine.allocate_memory({data_types::i64, format::bfyx, {1, 1, 1, 1}});
    auto initial_condition_mem = engine.allocate_memory({data_types::i64, format::bfyx, {1, 1, 1, 1}});
    set_values(num_iteration_mem, {0});
    set_values(trip_count_mem, {p.loop_trip_count});
    set_values(initial_condition_mem, {1});
    topology body(
        input_layout("body_input", layout{p.data_type, format::bfyx, p.loop_eltwise_shape}),
        input_layout("body_eltwise_operand", layout({p.data_type, format::bfyx, p.loop_eltwise_shape})),
        eltwise("body_eltwise", input_info("body_input"), input_info("body_eltwise_operand"), eltwise_mode::sum)
    );

    std::vector<loop::io_primitive_map> input_primitive_maps {loop::io_primitive_map("eltwise", "body_input", 2),
                                                              loop::io_primitive_map("loop_eltwise_init_values", "body_eltwise_operand")};
    std::vector<loop::io_primitive_map> output_primitive_maps {loop::io_primitive_map("loop", "body_eltwise", 2)};
    std::vector<loop::backedge_mapping> back_edges {loop::backedge_mapping("body_eltwise", "body_eltwise_operand")};

    auto body_program = build_program(engine, body, "", output_primitive_maps, back_edges);

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{p.data_type, p.default_format, p.loop_input_shape})),
        permute("permute", input_info("input"), p.permute_order),
        eltwise("eltwise", { input_info("permute"), input_info("eltwise_data") }, eltwise_mode::sum, p.data_type),
        data("loop_eltwise_init_values", get_mem(layout{p.data_type, format::bfyx, p.loop_eltwise_shape}, 0.f)),
        data("trip_count", trip_count_mem),
        data("initial_condition", initial_condition_mem),
        mutable_data("num_iteration", num_iteration_mem),
        loop("loop", { input_info("num_iteration"), input_info("trip_count"), input_info("initial_condition"),
                input_info("eltwise"), input_info("loop_eltwise_init_values") }, body_program,
             "trip_count", "initial_condition", "num_iteration",
             input_primitive_maps, output_primitive_maps, back_edges, p.loop_trip_count),
        reorder("output", input_info("loop"), format::bfyx, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

#define CASE_LOOP_F32_1 3, { 1, 8, 3, 2 }, { 1, 2, 8, 3 }, { 1, 2, 8, 1 }, { 0, 2, 3, 1 }, data_types::f32, format::bfyx, data_types::f32
#define CASE_LOOP_F16_0 4, { 1, 12, 4, 2 }, { 1, 2, 12, 4 }, { 1, 2, 12, 1 }, { 0, 2, 3, 1 }, data_types::f16, format::bfyx, data_types::f16

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_eltwise_loop, ::testing::ValuesIn(std::vector<loop_params>{
    loop_params{ CASE_LOOP_F32_1, 3, 5 },
    loop_params{ CASE_LOOP_F16_0, 3, 5 },
}));
} // namespace
