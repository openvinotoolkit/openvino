// Copyright (C) 2018-2026 Intel Corporation
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

TEST(fuse_primitives_with_layout, not_fuse_quantize_with_runtime_ranges) {
    // input1(b_fs_yx_fsv16)  input2(b_fs_yx_fsv16)
    //                    \   /
    //             eltwise(b_fs_yx_fsv16)  in_lo  in_hi  out_lo  out_hi  (runtime inputs)
    //                      |                |      |      |       |
    //                  quantize(b_fs_yx_fsv16) ----------------------
    //                      |
    //                 reorder(bfyx)
    //
    // prepare_quantization builds the scale/shift form of a quantize only from constant ranges, and a fused quantize
    // is generated only in that form. A quantize with runtime ranges (e.g. from a dynamically quantized activation)
    // must stay a separate primitive, otherwise the fused code applies the input_low tensor as the input scale.
    auto& engine = get_test_engine();

    const auto data_layout = layout{ ov::PartialShape{1, 16, 2, 2}, data_types::f32, format::bfyx };
    const auto range_layout = layout{ ov::PartialShape{1, 1, 1, 1}, data_types::f32, format::bfyx };
    const int levels = 256;
    const float in_lo = 0.0f;
    const float in_hi = 255.0f;
    const float out_lo = 0.0f;
    const float out_hi = 25.5f;

    // k + 0.25 keeps every value away from a rounding tie, and the values cover both clamps of the range.
    std::vector<float> input1_data(data_layout.count());
    for (size_t i = 0; i < input1_data.size(); ++i) {
        input1_data[i] = static_cast<float>(static_cast<int>(i) * 5 - 20) + 0.25f;
    }

    auto input1 = engine.allocate_memory(data_layout);
    auto input2 = engine.allocate_memory(data_layout);
    set_values(input1, input1_data);
    set_values(input2, std::vector<float>(data_layout.count(), 1.0f));

    std::map<primitive_id, memory::ptr> ranges;
    for (const auto& [id, value] : std::vector<std::pair<primitive_id, float>>{{"in_lo", in_lo}, {"in_hi", in_hi}, {"out_lo", out_lo}, {"out_hi", out_hi}}) {
        ranges[id] = engine.allocate_memory(range_layout);
        set_values(ranges[id], {value});
    }

    topology topology(
        input_layout("input1", data_layout),
        input_layout("input2", data_layout),
        input_layout("in_lo", range_layout),
        input_layout("in_hi", range_layout),
        input_layout("out_lo", range_layout),
        input_layout("out_hi", range_layout),
        reorder("input1_fsv16", input_info("input1"), format::b_fs_yx_fsv16, data_types::f32),
        reorder("input2_fsv16", input_info("input2"), format::b_fs_yx_fsv16, data_types::f32),
        eltwise("multiply", input_info("input1_fsv16"), input_info("input2_fsv16"), eltwise_mode::prod),
        quantize("quantize", input_info("multiply"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"),
                 levels, data_types::f32),
        reorder("reorder", input_info("quantize"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    ASSERT_TRUE(has_node(*network.get_program(), "quantize"));

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    for (const auto& [id, mem] : ranges) {
        network.set_input_data(id, mem);
    }
    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < input1_data.size(); ++i) {
        const float x = input1_data[i];
        float expected = out_lo;
        if (x > in_hi) {
            expected = out_hi;
        } else if (x > in_lo) {
            expected = std::round((x - in_lo) / (in_hi - in_lo) * (levels - 1)) / (levels - 1) * (out_hi - out_lo) + out_lo;
        }
        ASSERT_NEAR(output_ptr[i], expected, 1e-4f) << "index " << i << ", input " << x;
    }
}
