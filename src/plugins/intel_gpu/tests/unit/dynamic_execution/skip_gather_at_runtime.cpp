// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "gather_inst.h"
#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace skip_gather_tests {
enum execution_status {
	optimized = 0,
	skipped   = 1,
	executed  = 2
};

struct gather_iter_params {
    ov::PartialShape input1_shape;
    ov::PartialShape input2_shape;
    execution_status expected_status;
};


struct skip_gather_params {
    std::vector<gather_iter_params> input_data;
    int axis;
};

class skip_gather_at_runtime_test : public testing::TestWithParam<skip_gather_params> {};

TEST_P(skip_gather_at_runtime_test, runtime_skip) {
    auto p = GetParam();
    auto& engine = get_test_engine();
    auto axis = p.axis;
    auto input1_rank = p.input_data[0].input1_shape.size();
    auto input1_layout_dynamic = layout {ov::PartialShape::dynamic(input1_rank), data_types::f16, format::get_default_format(input1_rank)};
    auto input2_rank = p.input_data[0].input2_shape.size();
    auto input2_layout_dynamic = layout {ov::PartialShape::dynamic(input2_rank), data_types::f16, format::get_default_format(input2_rank)};
    topology topology(input_layout("input1", input1_layout_dynamic),
                        input_layout("input2", input1_layout_dynamic),
                        reshape("squeeze", input_info("input2"), false, {-1}, {-1}, reshape::reshape_mode::base),
                        gather("gather",
                                input_info("input1"),
                                input_info("squeeze"),
                                axis,
                                p.input_data[0].input1_shape.size(),
                                ov::Shape{},
                                0,
                                true),
                        reorder("reorder", input_info("gather"), format::get_default_format(input1_rank), data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto gather_inst = network.get_primitive("gather");
    for (auto in_shape_data : p.input_data) {
        auto input1_static_layout = layout {in_shape_data.input1_shape, data_types::f16, format::get_default_format(input1_rank)};
        auto input1_mem = engine.allocate_memory(input1_static_layout);
        network.set_input_data("input1", input1_mem);

        auto input2_static_layout = layout {in_shape_data.input2_shape, data_types::f16, format::get_default_format(input2_rank)};
        auto intpu2_unit_static_layout = layout {ov::PartialShape{1}, data_types::f16, format::get_default_format(input2_rank)};
        auto input2_mem = (input2_static_layout.count() == 0)? engine.allocate_memory(intpu2_unit_static_layout) : engine.allocate_memory(input2_static_layout);
        if (input2_static_layout.count() == 0)
            input2_mem = engine.reinterpret_buffer(*input2_mem, input2_static_layout);
        network.set_input_data("input2", input2_mem);

        auto outputs = network.execute();
        if (in_shape_data.expected_status == execution_status::executed) {
            ASSERT_FALSE(engine.is_the_same_buffer(gather_inst->dep_memory(0),  gather_inst->output_memory(0)));
            ASSERT_FALSE(gather_inst->can_be_optimized());
        } else if (in_shape_data.expected_status == execution_status::optimized) {
            ASSERT_TRUE(engine.is_the_same_buffer(gather_inst->dep_memory(0),   gather_inst->output_memory(0)));
            ASSERT_TRUE(gather_inst->can_be_optimized());
        } else {
            ASSERT_TRUE(gather_inst->get_output_layout(0).count() == 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, skip_gather_at_runtime_test,
    testing::ValuesIn(std::vector<skip_gather_params> {
        {{{ov::PartialShape{1,1,8}, ov::PartialShape{1,1}, execution_status::optimized},{ov::PartialShape{1,1,8}, ov::PartialShape{1,0}, execution_status::skipped},  {ov::PartialShape{1,11,8}, ov::PartialShape{1,6}, execution_status::executed}}, 1},
        {{{ov::PartialShape{1,2,8}, ov::PartialShape{1,1}, execution_status::executed},{ov::PartialShape{1,1,8}, ov::PartialShape{1,0}, execution_status::skipped},  {ov::PartialShape{1,11,8}, ov::PartialShape{1,6}, execution_status::executed}}, 1},
        {{{ov::PartialShape{1,1,8}, ov::PartialShape{1,1}, execution_status::optimized},{ov::PartialShape{1,1,8}, ov::PartialShape{1,1}, execution_status::optimized},{ov::PartialShape{1,11,8}, ov::PartialShape{1,6}, execution_status::executed}}, 1}
    }));
}  // skip gather tests
