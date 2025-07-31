// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"

#include "data_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"
#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(align_shape_for_numpy_broadcast, eltwise_broadcast) {
    // Topology :
    //   Input0 -> reorder -> (b_fs_yx_fsv16 / 5dims) -> Eltwise <- (bfyx / 3dims) <- Input1
    // Expected : Eltwise <- (bfzyx) <- Reshape <- (bfyz) <- Input1

    auto& engine = get_test_engine();

    topology topology;
    topology.add(input_layout("input", layout{ ov::PartialShape{ 1, 32, 2, 4, 8 }, data_types::f32, format::bfzyx }));
    topology.add(reorder("reorder_input", input_info("input"), format::b_fs_zyx_fsv16, data_types::f16));
    topology.add(input_layout("eltw_input", layout{ ov::PartialShape{ 2, 4, 8 }, data_types::f16, format::bfyx }));
    topology.add(eltwise("eltwise", input_info("eltw_input"), input_info("reorder_input"), eltwise_mode::sum, ov::op::AutoBroadcastType::NUMPY));
    topology.add(reorder("reorder_output", input_info("eltwise"), format::b_fs_zyx_fsv16, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    program::ptr prog = nullptr;
    OV_ASSERT_NO_THROW(prog = program::build_program(engine, topology, config));
    ASSERT_NE(prog, nullptr);

    auto prog_impl = prog.get();

    auto& eltwise_node = prog_impl->get_node("eltwise");

    ASSERT_EQ(eltwise_node.get_input_layouts()[0].format, format::b_fs_zyx_fsv16);
    ASSERT_EQ(eltwise_node.get_input_layouts()[1].format, format::b_fs_zyx_fsv16);
    ASSERT_EQ(eltwise_node.get_output_layout().format, format::b_fs_zyx_fsv16);
}
