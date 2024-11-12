// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/primitives/scatter_nd_update.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "permute_inst.h"
#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace skip_scatter_update_tests {

enum scatter_update_type {
	ScatterUpdate           = 0,
	ScatterNDUpdate         = 1,
	ScatterElementsUpdate   = 2
};

struct skip_scatter_update_params {
    scatter_update_type scatter_type;
    bool scatter_update_01_skipped;
    bool scatter_update_02_skipped;
};

class skip_scatter_update_at_runtime_test : public testing::TestWithParam<skip_scatter_update_params> {};

TEST_P(skip_scatter_update_at_runtime_test, runtime_skip) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    auto input_layout_static    = layout{ov::PartialShape{1,16}, data_types::f16, format::bfyx};
    auto rank                   = input_layout_static.get_partial_shape().size();
    auto input_layout_dynamic   = layout {ov::PartialShape::dynamic(rank), data_types::f16, format::get_default_format(rank)};

    auto idx1_nonzero_layout    = layout{ov::PartialShape{1,16}, data_types::f16, format::bfyx};
    auto idx1_zero_layout       = layout{ov::PartialShape{0,16}, data_types::f16, format::bfyx};
    auto update1_nonzero_layout = layout{ov::PartialShape{1,16}, data_types::f16, format::bfyx};
    auto update1_zero_layout    = layout{ov::PartialShape{0,16}, data_types::f16, format::bfyx};

    auto idx2_nonzero_layout    = layout{ov::PartialShape{1,16}, data_types::f16, format::bfyx};
    auto idx2_zero_layout       = layout{ov::PartialShape{0,16}, data_types::f16, format::bfyx};
    auto update2_nonzero_layout = layout{ov::PartialShape{1,16}, data_types::f16, format::bfyx};
    auto update2_zero_layout    = layout{ov::PartialShape{0,16}, data_types::f16, format::bfyx};

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = nullptr;

    if (p.scatter_type == scatter_update_type::ScatterElementsUpdate) {
        topology topology(input_layout("input", input_layout_dynamic),
                            input_layout("idx1", input_layout_dynamic),
                            input_layout("idx2", input_layout_dynamic),
                            input_layout("update1", input_layout_dynamic),
                            input_layout("update2", input_layout_dynamic),
                            scatter_elements_update("scatter1", input_info("input"),    input_info("idx1"), input_info("update1"), 0),
                            scatter_elements_update("scatter2", input_info("scatter1"), input_info("idx2"), input_info("update2"), 0),
                            reorder("reorder", input_info("scatter2"), format::get_default_format(rank), data_types::f32));

        network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    } else if (p.scatter_type == scatter_update_type::ScatterUpdate) {
        topology topology(input_layout("input", input_layout_dynamic),
                            input_layout("idx1", input_layout_dynamic),
                            input_layout("idx2", input_layout_dynamic),
                            input_layout("update1", input_layout_dynamic),
                            input_layout("update2", input_layout_dynamic),
                            scatter_update("scatter1", input_info("input"),    input_info("idx1"), input_info("update1"), 0),
                            scatter_update("scatter2", input_info("scatter1"), input_info("idx2"), input_info("update2"), 0),
                            reorder("reorder", input_info("scatter2"), format::get_default_format(rank), data_types::f32));

        network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    } else if (p.scatter_type == scatter_update_type::ScatterNDUpdate) {
        input_layout_static     = layout{ov::PartialShape{12}, data_types::f16, format::bfyx};
        rank                    = input_layout_static.get_partial_shape().size();
        input_layout_dynamic    = layout {ov::PartialShape::dynamic(rank), data_types::f16, format::get_default_format(rank)};

        idx1_nonzero_layout     = layout{ov::PartialShape{12,1}, data_types::f16, format::bfyx};
        idx1_zero_layout        = layout{ov::PartialShape{0,1}, data_types::f16, format::bfyx};
        update1_nonzero_layout  = layout{ov::PartialShape{12}, data_types::f16, format::bfyx};
        update1_zero_layout     = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};
        rank                    = idx1_nonzero_layout.get_partial_shape().size();
        auto idx_layout_dynamic = layout {ov::PartialShape::dynamic(rank), data_types::f16, format::get_default_format(rank)};

        idx2_nonzero_layout     = layout{ov::PartialShape{12,1}, data_types::f16, format::bfyx};
        idx2_zero_layout        = layout{ov::PartialShape{0,1}, data_types::f16, format::bfyx};
        update2_nonzero_layout  = layout{ov::PartialShape{12}, data_types::f16, format::bfyx};
        update2_zero_layout     = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};



        topology topology(input_layout("input", input_layout_dynamic),
                            input_layout("idx1", idx_layout_dynamic),
                            input_layout("idx2", idx_layout_dynamic),
                            input_layout("update1", input_layout_dynamic),
                            input_layout("update2", input_layout_dynamic),
                            scatter_nd_update("scatter1", input_info("input"),    input_info("idx1"), input_info("update1"), 2),
                            scatter_nd_update("scatter2", input_info("scatter1"), input_info("idx2"), input_info("update2"), 2),
                            reorder("reorder", input_info("scatter2"), format::get_default_format(rank), data_types::f32));

        network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    }

    auto input_mem              = engine.allocate_memory(input_layout_static);

    auto idx1_layout_static     = p.scatter_update_01_skipped? idx1_zero_layout : idx1_nonzero_layout;
    auto update1_layout_static  = p.scatter_update_01_skipped? update1_zero_layout : update1_nonzero_layout;

    auto idx1_mem               = engine.allocate_memory(idx1_nonzero_layout);
    auto update1_mem            = engine.allocate_memory(update1_nonzero_layout);

    if (p.scatter_update_01_skipped) {
        idx1_mem    = engine.reinterpret_buffer(*idx1_mem, idx1_zero_layout);
        update1_mem = engine.reinterpret_buffer(*update1_mem, update1_zero_layout);
    }

    auto idx2_layout_static     = p.scatter_update_02_skipped? idx2_zero_layout : idx2_nonzero_layout;
    auto update2_layout_static  = p.scatter_update_02_skipped? update2_zero_layout : update2_nonzero_layout;

    auto idx2_mem               = engine.allocate_memory(idx2_nonzero_layout);
    auto update2_mem            = engine.allocate_memory(update2_nonzero_layout);
    if (p.scatter_update_02_skipped) {
        idx2_mem    = engine.reinterpret_buffer(*idx2_mem, idx2_zero_layout);
        update2_mem = engine.reinterpret_buffer(*update2_mem, update2_zero_layout);
    }
    network->set_input_data("input", input_mem);
    network->set_input_data("idx1", idx1_mem);
    network->set_input_data("idx2", idx2_mem);
    network->set_input_data("update1", update1_mem);
    network->set_input_data("update2", update2_mem);
    auto outputs = network->execute();
    outputs.begin()->second.get_memory();

    auto input_inst = network->get_primitive("input");
    auto scatter1_inst = network->get_primitive("scatter1");
    auto scatter2_inst = network->get_primitive("scatter2");

    ASSERT_EQ(scatter1_inst->can_be_optimized(), p.scatter_update_01_skipped);
    ASSERT_EQ(scatter2_inst->can_be_optimized(), p.scatter_update_02_skipped);

    if (scatter1_inst->can_be_optimized()) {
        ASSERT_TRUE(engine.is_the_same_buffer(scatter1_inst->dep_memory(0),     scatter1_inst->output_memory(0)));
    } else {
        ASSERT_FALSE(engine.is_the_same_buffer(scatter1_inst->dep_memory(0),    scatter1_inst->output_memory(0)));
    }

    if (scatter2_inst->can_be_optimized()) {
        ASSERT_TRUE(engine.is_the_same_buffer(scatter2_inst->dep_memory(0),     scatter2_inst->output_memory(0)));
    } else {
        ASSERT_FALSE(engine.is_the_same_buffer(scatter2_inst->dep_memory(0),    scatter2_inst->output_memory(0)));
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, skip_scatter_update_at_runtime_test,
    testing::ValuesIn(std::vector<skip_scatter_update_params> {
        { scatter_update_type::ScatterUpdate,           true,       true },
        { scatter_update_type::ScatterUpdate,           true,       false},
        { scatter_update_type::ScatterUpdate,           false,      true },
        { scatter_update_type::ScatterUpdate,           false,      false},

        { scatter_update_type::ScatterNDUpdate,         true,       true },
        { scatter_update_type::ScatterNDUpdate,         true,       false},
        { scatter_update_type::ScatterNDUpdate,         false,      true },
        { scatter_update_type::ScatterNDUpdate,         false,      false},

        { scatter_update_type::ScatterElementsUpdate,   true,       true },
        { scatter_update_type::ScatterElementsUpdate,   true,       false},
        { scatter_update_type::ScatterElementsUpdate,   false,      true },
        { scatter_update_type::ScatterElementsUpdate,   false,      false},

    }));
}  // skip permute tests
