// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "random_generator.hpp"

using namespace cldnn;
using namespace ::tests;

TEST(fuse_primitives_with_layout, fuse_when_layout_format_of_input_and_output_are_same) {
    // input1(b_fs_yx_fsv16)  input2(b_fs_yx_fsv16)
    //                    \   /
    //             eltwise(b_fs_yx_fsv16)
    //                      |
    //                quantize(bfyx)
    //                      |
    //                 reorder(bfyx)
    //
    // This test case validates the pattern : eltwise --> quantize
    // If eltwise and quantize node have different output layout format, do not fuse.
    //
    auto& engine = get_test_engine();

    auto in_layout1 = layout{ ov::PartialShape{2304, 64, 3, 3}, data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout2 = layout{ ov::PartialShape{2304, 1, 1, 1}, data_types::f16, format::b_fs_yx_fsv16 };
    auto qt_layout = layout{ ov::PartialShape{2304, 64, 3, 3}, data_types::f32, format::bfyx };
    auto data1 = engine.allocate_memory({ ov::PartialShape{2304, 1, 1, 1}, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input1", in_layout1),
        input_layout("input2", in_layout2),
        eltwise("multiply", input_info("input1"), input_info("input2"), eltwise_mode::prod),
        data("in_low", data1),
        data("in_high", data1),
        data("out_low", data1),
        data("out_high", data1),
        quantize("quantize", input_info("multiply"), input_info("in_low"), input_info("in_high"), input_info("out_low"), input_info("out_high"), 256, data_types::f32),
        reorder("reorder", input_info("quantize"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto program = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<compile_graph>(*program);
    for (auto node : program->get_processing_order()) {
        if (node->get_org_primitive_id() == "quantize") {
            node->set_output_layout(qt_layout, false);
        }
    }

    program_wrapper::apply_opt_pass<fuse_primitives_with_layout>(*program);

    ASSERT_TRUE(has_node(*program, "quantize"));
}
