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

class LoopFusingTest : public ::BaseFusingTest<loop_params> {
public:

    void execute(loop_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

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
        eltwise("body_eltwise", "body_input", "body_eltwise_operand", eltwise_mode::sum)
    );

    std::vector<loop::io_primitive_map> input_primitive_maps {loop::io_primitive_map("eltwise", "body_input", 2),
                                                              loop::io_primitive_map("loop_eltwise_init_values", "body_eltwise_operand")};
    std::vector<loop::io_primitive_map> output_primitive_maps {loop::io_primitive_map("loop", "body_eltwise", 2)};
    std::vector<loop::backedge_mapping> back_edges {loop::backedge_mapping("body_eltwise", "body_eltwise_operand")};

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{p.data_type, p.default_format, p.loop_input_shape})),
        permute("permute", "input", p.permute_order),
        eltwise("eltwise", {"permute", "eltwise_data"}, eltwise_mode::sum, p.data_type),
        data("loop_eltwise_init_values", get_mem(layout{p.data_type, format::bfyx, p.loop_eltwise_shape}, 0.f)),
        data("trip_count", trip_count_mem),
        data("initial_condition", initial_condition_mem),
        mutable_data("num_iteration", num_iteration_mem),
        loop("loop", {"eltwise", "loop_eltwise_init_values"}, body,
             "trip_count", "initial_condition", "num_iteration",
             input_primitive_maps, output_primitive_maps, back_edges, p.loop_trip_count),
        reorder("output", "loop", format::bfyx, p.default_type)
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
