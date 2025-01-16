// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "crop_inst.h"
#include "convolution_inst.h"
#include "gather_inst.h"
#include "gemm_inst.h"
#include "reshape_inst.h"
#include "fully_connected_inst.h"
#include "permute_inst.h"
#include "reorder_inst.h"
#include "shape_of_inst.h"
#include "gather_inst.h"
#include "strided_slice_inst.h"
#include "mvn_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"
#include "resample_inst.h"
#include "openvino/op/interpolate.hpp"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(prepare_buffer_fusing, optimize_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto pattern_layout = layout{ov::PartialShape::dynamic(4), data_types::i64, format::bfyx};
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("pattern", pattern_layout));
    topology.add(permute("permute1", input_info("input"), {0, 2, 3, 1}));
    topology.add(reshape("reshape", input_info("permute1"), input_info("pattern"), false, ov::PartialShape::dynamic(4)));
    topology.add(permute("permute2", input_info("reshape"), {0, 3, 2, 1}));
    topology.add(reorder("reorder", input_info("permute2"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 2, 4}, data_types::f16, format::bfyx });
    auto pattern_memory = engine.allocate_memory(layout{ ov::PartialShape{4}, data_types::i64, format::bfyx });
    set_values<ov::float16>(input_memory, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7, 4.8, 19.2, -10.1, 8.1, 10.2, 1.3, 1.44, 1.5});
    set_values<int64_t>(pattern_memory, {1, 4, 1, -1});

    net.set_input_data("input", input_memory);
    net.set_input_data("pattern", pattern_memory);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("reorder");
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);
    ASSERT_EQ(out_mem->count(), 16);
}

TEST(prepare_buffer_fusing, static_node_after_optimized_out_dyn_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape{1, 2, -1}, data_types::f32, format::bfyx };
    auto weights_layout = layout{ov::PartialShape{2, 4}, data_types::f32, format::bfyx};
    auto weights_memory = engine.allocate_memory(weights_layout);
    set_values<float>(weights_memory, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weights", weights_memory));
    topology.add(permute("permute1", input_info("input"), {0, 2, 1}));
    topology.add(reshape("reshape", input_info("permute1"), false, {2, 4}, ov::PartialShape{2, 4}));
    topology.add(fully_connected("fc", input_info("reshape"), "weights", "", 2));
    topology.add(reorder("reorder", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    prog->get_node("reorder").get_output_layout(true);
    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*prog);
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    OV_ASSERT_NO_THROW(prog->get_node("reshape"));
    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());
    program_wrapper::apply_opt_pass<build_implementations>(*prog);

    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 4}, data_types::f32, format::bfyx });
    set_values<float>(input_memory, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7});

    net.set_input_data("input", input_memory);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    OV_ASSERT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("reorder");
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);
    ov::PartialShape expected_shape = {2, 2};
    ASSERT_EQ(out_mem->count(), 4);
    ASSERT_EQ(out_mem->get_layout().get_partial_shape(), expected_shape);
}

TEST(prepare_buffer_fusing, in_place_concat_static) {
    auto& engine = get_test_engine();
    auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx }; // => {1, 4, 3, 2}
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 1}, data_types::f32, format::bfyx }; // => {1, 4, 1, 2}
    topology topology;
    topology.add(input_layout("input1", in_layout1));
    topology.add(input_layout("input2", in_layout2));
    topology.add(permute("permute1", input_info("input1"), {0, 3, 2, 1}));
    topology.add(permute("permute2", input_info("input2"), {3, 2, 0, 1}));
    topology.add(concatenation("concat", { input_info("permute1"), input_info("permute2") }, 2));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<float>(input_memory1,
                      {1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   11.0,   22.0,   33.0,   44.0,   55.0,   66.0,
                       111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 1111.0, 2222.0, 3333.0, 4444.0, 5555.0, 6666.0});
    set_values<float>(input_memory2, {1234.0, 2345.0, 3456.0, 4567.0, 5678.0, 6789.0, 9012.0, 9999.0});

    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    const auto& concat_node = net.get_primitive("concat")->get_node();
    auto concat_mem = net.get_primitive("concat")->output_memory_ptr();
    auto permute1_mem = net.get_primitive("permute1")->output_memory_ptr();
    auto permute2_mem = net.get_primitive("permute1")->output_memory_ptr();
    ASSERT_TRUE(concat_node.can_be_optimized());
    ASSERT_EQ(concat_mem, permute1_mem);
    ASSERT_EQ(concat_mem, permute2_mem);
    auto out_lay = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    std::vector<float> ref_output = {1.0,    2.0,    3.0,    4.0,    111.0,  222.0,  333.0,  444.0,  5.0,    6.0,   11.0,
                                     22.0,   555.0,  666.0,  1111.0, 2222.0, 33.0,   44.0,   55.0,   66.0,   3333.0, 4444.0,
                                     5555.0, 6666.0, 1234.0, 2345.0, 3456.0, 4567.0, 5678.0, 6789.0, 9012.0, 9999.0};

    for (size_t x = 0; x < out_lay.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}

TEST(prepare_buffer_fusing, in_place_concat_dynamic) {
    auto& engine = get_test_engine();
    auto in_layout1_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto in_layout2_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx };
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 1}, data_types::f32, format::bfyx };

    topology topology;
    topology.add(input_layout("input1", in_layout1_0));
    topology.add(input_layout("input2", in_layout2_0));
    topology.add(permute("permute1", input_info("input1"), {0, 3, 2, 1}));
    topology.add(permute("permute2", input_info("input2"), {3, 2, 0, 1}));

    topology.add(concatenation("concat", { input_info("permute1"), input_info("permute2") }, 2));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<float>(input_memory1,
                      {1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   11.0,   22.0,   33.0,   44.0,   55.0,   66.0,
                       111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 1111.0, 2222.0, 3333.0, 4444.0, 5555.0, 6666.0});
    set_values<float>(input_memory2, {1234.0, 2345.0, 3456.0, 4567.0, 5678.0, 6789.0, 9012.0, 9999.0});
    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);

    std::vector<float> ref_output = {1.0,    2.0,    3.0,    4.0,    111.0,  222.0,  333.0,  444.0,  5.0,    6.0,   11.0,
                                     22.0,   555.0,  666.0,  1111.0, 2222.0, 33.0,   44.0,   55.0,   66.0,   3333.0, 4444.0,
                                     5555.0, 6666.0, 1234.0, 2345.0, 3456.0, 4567.0, 5678.0, 6789.0, 9012.0, 9999.0};

    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    const auto& concat_node = net.get_primitive("concat")->get_node();
    auto concat_mem = net.get_primitive("concat")->output_memory_ptr();
    auto permute1_mem = net.get_primitive("permute1")->output_memory_ptr();
    auto permute2_mem = net.get_primitive("permute1")->output_memory_ptr();

    ASSERT_TRUE(concat_node.can_be_optimized());
    ASSERT_EQ(concat_mem.get(), permute1_mem.get());
    ASSERT_EQ(concat_mem.get(), permute2_mem.get());
    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}

TEST(prepare_buffer_fusing, in_place_concat_strided_slice_dyn) {
    auto& engine = get_test_engine();
    auto in_layout1_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto in_layout2_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto in_layout3_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };
    auto in_layout1 = layout{ ov::PartialShape{2, 2, 2, 2}, data_types::f32, format::bfyx };
    auto in_layout2 = layout{ ov::PartialShape{2, 2, 2, 2}, data_types::f32, format::bfyx };
    auto in_layout3 = layout{ ov::PartialShape{2, 2, 2, 2}, data_types::f32, format::bfyx };
    auto begin = engine.allocate_memory({ ov::PartialShape{4}, data_types::i64, format::bfyx });
    auto end = engine.allocate_memory({ ov::PartialShape{4}, data_types::i64, format::bfyx });
    auto strides = engine.allocate_memory({ ov::PartialShape{4}, data_types::i64, format::bfyx });
    set_values<int64_t>(begin, {0, 0, 0, 0});
    set_values<int64_t>(end, {2, 2, 2, 2 });
    set_values<int64_t>(strides, {1, 1, 1, 1});

    topology topology;
    topology.add(input_layout("input1", in_layout1_0));
    topology.add(input_layout("input2", in_layout2_0));
    topology.add(input_layout("input3", in_layout3_0));
    topology.add(data("input4", begin));
    topology.add(data("input5", end));
    topology.add(data("input6", strides));
    topology.add(reorder("reorder1", input_info("input1"), format::bfyx, data_types::f16));
    topology.add(reorder("reorder2", input_info("input2"), format::bfyx, data_types::f16));
    topology.add(reorder("reorder3", input_info("input3"), format::bfyx, data_types::f16));
    topology.add(eltwise("eltwise", { input_info("reorder1"), input_info("reorder2") }, eltwise_mode::prod));
    topology.add(strided_slice("strided_slice", input_info("reorder3"), input_info("input4"),
                               input_info("input5"), input_info("input6"), {}, {}, {}, {}, {}, {}));
    topology.add(concatenation("concat", { input_info("eltwise"), input_info("strided_slice") }, 0));
    topology.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    auto input_memory3 = engine.allocate_memory(in_layout3);
    set_values<float>(input_memory1, {
        1.f, 0.f, 5.f, 1.f, 2.f, 0.f, 6.f, 3.f,
        3.f, 0.5f, 7.f, 12.f, 4.f, -0.5f, 8.f, 7.5f
    });
    set_values<float>(input_memory2, {
        0.5f, 5.f, 15.f, 6.f, 0.5f, 2.f, 8.f, -0.5f,
        2.5f, 7.f, 17.f, 8.f, 2.5f, 4.f, 10.f, -2.5f
    });
    set_values<float>(input_memory3, {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
    });

    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);
    net.set_input_data("input3", input_memory3);

    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());

    const auto& concat_node = net.get_primitive("concat")->get_node();
    auto concat_mem = net.get_primitive("concat")->output_memory_ptr();
    auto eltwise_mem = net.get_primitive("eltwise")->output_memory_ptr();
    auto strided_slice_mem = net.get_primitive("strided_slice")->output_memory_ptr();

    ASSERT_TRUE(concat_node.can_be_optimized());
    ASSERT_EQ(concat_mem, eltwise_mem);
    ASSERT_EQ(concat_mem, strided_slice_mem);

    auto out_lay = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    std::vector<float> ref_output = {
        0.5f, 0.0f, 75.f, 6.0f, 1.0f, 0.0f, 48.f, -1.5f,
        7.5f, 3.5f, 119.f, 96.0f, 10.0f, -2.0f, 80.f, -18.75f,
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f
    };

    for (size_t x = 0; x < out_lay.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}

TEST(prepare_buffer_fusing, in_place_concat_dynamic_onednn_batch1) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;
    auto in_layout1_0 = layout{ ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout2_0 = layout{ ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout1 = layout{ ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout2 = layout{ ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16 };

    topology topology;
    topology.add(input_layout("input1", in_layout1));
    topology.add(input_layout("input2", in_layout2));
    topology.add(reorder("reorder1", input_info("input1"), format::bfyx, data_types::f16));
    topology.add(reorder("reorder2", input_info("input2"), format::bfyx, data_types::f16));

    topology.add(concatenation("concat", { input_info("reorder1"), input_info("reorder2") }, 1));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    auto& concat_node_p = prog->get_node("concat");
    ASSERT_TRUE(concat_node_p.can_be_optimized());
    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<ov::float16>(input_memory1,
                       {ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f),
                        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f)});
    set_values<ov::float16>(input_memory2,
                       {ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f),
                        ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f)});
    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);

    std::vector<ov::float16> ref_output = {
                        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f),
                        ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f),
                        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f),
                        ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f)};

    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(out_mem, get_test_stream());

    cldnn::mem_lock<ov::float16> input1_ptr(input_memory1, get_test_stream());
    cldnn::mem_lock<ov::float16> input2_ptr(input_memory2, get_test_stream());

    const auto& concat_inst = net.get_primitive("concat");
    const auto& concat_node_n = concat_inst->get_node();
    auto concat_mem = net.get_primitive("concat")->output_memory_ptr();
    auto reorder1_mem = net.get_primitive("reorder1")->output_memory_ptr();
    auto reorder2_mem = net.get_primitive("reorder2")->output_memory_ptr();

    ASSERT_EQ(concat_mem.get(), reorder1_mem.get());
    ASSERT_EQ(concat_mem.get(), reorder2_mem.get());
    ASSERT_TRUE(concat_inst->can_be_optimized());
    ASSERT_TRUE(concat_node_n.can_be_optimized());

    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}

TEST(prepare_buffer_fusing, in_place_concat_dynamic_onednn_batch2) {
    // Check no buffer fusing when onednn concat with b=2. It is not supported.
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;
    auto in_layout1_0 = layout{ ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout2_0 = layout{ ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout1 = layout{ ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16 };
    auto in_layout2 = layout{ ov::PartialShape{1, 16, 2, 1}, data_types::f16, format::b_fs_yx_fsv16 };

    topology topology;
    topology.add(input_layout("input1", in_layout1_0));
    topology.add(input_layout("input2", in_layout2_0));
    topology.add(reorder("reorder1", input_info("input1"), format::bfyx, data_types::f16));
    topology.add(reorder("reorder2", input_info("input2"), format::bfyx, data_types::f16));

    topology.add(concatenation("concat", { input_info("reorder1"), input_info("reorder2") }, 0));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplForcingMap forcing_map = {
        {"reorder1", ov::intel_gpu::ImplementationDesc{format::any, "", impl_types::onednn}},
        {"reorder2", ov::intel_gpu::ImplementationDesc{format::any, "", impl_types::onednn}}
    };
    config.set_property(ov::intel_gpu::force_implementations(forcing_map));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    auto& concat_node_p = prog->get_node("concat");
    ASSERT_TRUE(concat_node_p.can_be_optimized());
    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<ov::float16>(input_memory1,
                       {ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f),
                        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f)});
    set_values<ov::float16>(input_memory2,
                       {ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f),
                        ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f)});
    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);

    std::vector<ov::float16> ref_output = {
                        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f),
                        ov::float16(1.0f), ov::float16(2.0f), ov::float16(3.0f), ov::float16(4.0f), ov::float16(5.0f), ov::float16(6.0f), ov::float16(7.0f), ov::float16(8.0f),
                        ov::float16(11.0f), ov::float16(22.0f), ov::float16(33.0f), ov::float16(44.0f), ov::float16(55.0f), ov::float16(66.0f), ov::float16(77.0f), ov::float16(88.0f),
                        ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f),
                        ov::float16(111.0f), ov::float16(222.0f), ov::float16(333.0f), ov::float16(444.0f), ov::float16(555.0f), ov::float16(666.0f), ov::float16(777.0f), ov::float16(888.0f),
                        ov::float16(1111.0f), ov::float16(2222.0f), ov::float16(3333.0f), ov::float16(4444.0f), ov::float16(5555.0f), ov::float16(6666.0f), ov::float16(7777.0f), ov::float16(8888.0f)};

    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<ov::float16> output_ptr(out_mem, get_test_stream());

    cldnn::mem_lock<ov::float16> input1_ptr(input_memory1, get_test_stream());
    cldnn::mem_lock<ov::float16> input2_ptr(input_memory2, get_test_stream());

    const auto& concat_inst = net.get_primitive("concat");
    const auto& concat_node_n = concat_inst->get_node();
    auto concat_mem = net.get_primitive("concat")->output_memory_ptr();
    auto reorder1_mem = net.get_primitive("reorder1")->output_memory_ptr();
    auto reorder2_mem = net.get_primitive("reorder2")->output_memory_ptr();

    ASSERT_NE(concat_mem.get(), reorder1_mem.get());
    ASSERT_NE(concat_mem.get(), reorder2_mem.get());
    ASSERT_FALSE(concat_inst->can_be_optimized());
    ASSERT_TRUE(concat_node_n.can_be_optimized());

    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}

TEST(prepare_buffer_fusing, in_place_concat_dynamic__static_dim_dyn_pad) {
    auto& engine = get_test_engine();
    auto in_layout1_0 = layout{ ov::PartialShape{-1, 2, -1, -1}, data_types::f32, format::bfyx }; // => {-1, -1, -1, 2}
    auto in_layout2_0 = layout{ ov::PartialShape{1, 2, -1, -1}, data_types::f32, format::bfyx }; // => {-1, -1, 1, 2}
    auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx };
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 1}, data_types::f32, format::bfyx };

    topology topology;
    topology.add(input_layout("input1", in_layout1_0));
    topology.add(input_layout("input2", in_layout2_0));
    topology.add(permute("permute1", input_info("input1"), {0, 3, 2, 1}));
    topology.add(permute("permute2", input_info("input2"), {3, 2, 0, 1}));

    topology.add(concatenation("concat", { input_info("permute1"), input_info("permute2") }, 2));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<float>(input_memory1,
                      {1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   11.0,   22.0,   33.0,   44.0,   55.0,   66.0,
                       111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 1111.0, 2222.0, 3333.0, 4444.0, 5555.0, 6666.0});
    set_values<float>(input_memory2, {1234.0, 2345.0, 3456.0, 4567.0, 5678.0, 6789.0, 9012.0, 9999.0});
    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);

    std::vector<float> ref_output = {1.0,    2.0,    3.0,    4.0,    111.0,  222.0,  333.0,  444.0,  5.0,    6.0,   11.0,
                                     22.0,   555.0,  666.0,  1111.0, 2222.0, 33.0,   44.0,   55.0,   66.0,   3333.0, 4444.0,
                                     5555.0, 6666.0, 1234.0, 2345.0, 3456.0, 4567.0, 5678.0, 6789.0, 9012.0, 9999.0};

    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    const auto& concat_node = net.get_primitive("concat")->get_node();
    auto concat_mem = net.get_primitive("concat")->output_memory_ptr();
    auto permute1_mem = net.get_primitive("permute1")->output_memory_ptr();
    auto permute2_mem = net.get_primitive("permute1")->output_memory_ptr();

    ASSERT_TRUE(concat_node.can_be_optimized());
    ASSERT_EQ(concat_mem.get(), permute1_mem.get());
    ASSERT_EQ(concat_mem.get(), permute2_mem.get());
    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}

TEST(prepare_buffer_fusing, crop_b_axis) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 2, 2 } });


    set_values(input1, {
        1.f, 2.f,  3.f,  4.f,  1.f, 2.f,  3.f,  4.f,
        5.f, 6.f,  7.f,  8.f,  5.f, 6.f,  7.f,  11.f,
        9.f, 10.f, 11.f, 12.f, 9.f, 10.f, 11.f, 12.f
    });

    topology topology;
    topology.add(input_layout("Input", input1->get_layout()));
    topology.add(crop("crop", input_info("Input"), tensor{1, 2, 2, 2}, tensor(1, 0, 0, 0)));
    topology.add(reorder("reorder", input_info("crop"), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("Input", input1);

    auto outputs = network.execute();

    auto crop_prim = network.get_primitive("crop");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);

    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    std::vector<int8_t> expected_results = {
        5, 6, 7, 8, 5, 6, 7, 11
    };

    int crop_batch_num = 1;
    int crop_feature_num = 2;
    int crop_y_size = 2;
    int crop_x_size = 2;
    for (int b = 0; b < crop_batch_num; ++b) {
        for (int f = 0; f < crop_feature_num; ++f) {
            for (int y = 0; y < crop_y_size; ++y) {
                for (int x = 0; x < crop_x_size; ++x) {
                    int linear_id = x + 2 * (y + 2 * f);
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    ASSERT_EQ(output_ptr[output_linear_id], expected_results[linear_id]);
                }
            }
        }
    }
}

TEST(prepare_buffer_fusing, skip_in_place_concat_inside_shape_of_subgraph) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 32, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{1, 32, 32, 32}, data_types::f16, format::bfyx});
    auto data_0 = engine.allocate_memory({ ov::PartialShape{}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{}, data_types::f32, format::bfyx });
    auto data_2 = engine.allocate_memory({ ov::PartialShape{4}, data_types::i32, format::bfyx });

    const ov::op::AutoBroadcastSpec& broadcast_spec = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY);

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(data("data_2", data_2));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather0", input_info("shape_of"), input_info("data_0"), 0, 0, {}, 0, true));
    topology.add(reorder("reorder0", input_info("gather0"), format::any, data_types::f32,
                         std::vector<float>(), reorder_mean_mode::subtract, padding(), true));
    topology.add(eltwise("eltwise0", input_info("reorder0"), input_info("data_1"), eltwise_mode::prod, broadcast_spec));
    topology.add(reshape("reshape0", input_info("eltwise0"), false, {},
                         ov::PartialShape{1}, reshape::reshape_mode::unsqueeze));
    topology.add(gather("gather1", input_info("shape_of"), input_info("data_0"), 0, 0, {}, 0, true));
    topology.add(reorder("reorder1", input_info("gather1"), format::any, data_types::f32,
                         std::vector<float>(), reorder_mean_mode::subtract, padding(), true));
    topology.add(eltwise("eltwise1", input_info("reorder1"), input_info("data_1"), eltwise_mode::prod, broadcast_spec));
    topology.add(reshape("reshape1", input_info("eltwise1"), false, {},
                         ov::PartialShape{1}, reshape::reshape_mode::unsqueeze));
    topology.add(crop("crop", input_info("shape_of"), tensor({2,1,1,1,1,1,1,1,1}), tensor({0,0,0,0,1,1,1,1,1})));
    topology.add(concatenation("concat0", {input_info("reshape0"), input_info("reshape1")}, 0, data_types::f32));
    topology.add(reorder("reorder3", input_info("concat0"), format::any, data_types::i32,
                         std::vector<float>(), reorder_mean_mode::subtract, padding(), true));
    topology.add(concatenation("concat1", {input_info("reorder3"), input_info("crop")}, 0, data_types::i32));
    topology.add(eltwise("eltwise2", input_info("concat1"), input_info("data_2"), eltwise_mode::prod, broadcast_spec));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    network.execute();

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    auto& crop_node = prog->get_node("crop");
    auto impl_param = crop_node.get_kernel_impl_params();
    auto crop_mem = network.get_output_memory("crop");
    ASSERT_EQ(impl_param->get_output_layout(), crop_mem->get_layout());
    auto in_place = engine.is_the_same_buffer(*network.get_output_memory("crop"), *network.get_output_memory("concat1"));
    ASSERT_FALSE(in_place);
}

TEST(prepare_buffer_fusing, in_place_crop_static) {
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
    auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });

    set_values(input_mem, { -0.5f,  2.0f,  0.5f,  1.0f,
                             0.5f, -2.0f, -0.5f, -1.0f });
    set_values<uint8_t>(weights_mem, { 1,  2,  3,  4,
                                       5,  6,  7,  8,
                                       9, 10, 11, 12,
                                      13, 14, 15,  0,
                                      15, 14, 13, 12,
                                      11, 10,  9,  8,
                                       7,  6,  5,  4,
                                       3,  2,  1,  0});
    set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f });
    set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });
    set_values(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f });

    std::vector<float> out1 = { 13.f, 58.f, -51.f, -108.f, -11.f, -62.f, 57.f, 100.f };
    std::vector<float> out2 = { 18.5f, -18.f, 1.f, -4.f, -8.5f, 6.f, 13.f, 8.f };
    std::vector<float> out3 = { 13.f, 58.f, -51.f, -108.f,     18.5f, -18.f, 1.f, -4.f,
                                -11.f, -62.f, 57.f, 100.f,    -8.5f, 6.f, 13.f, 8.f };

    topology topology(
        input_layout("input", input_mem->get_layout()),
        data("weights", weights_mem),
        data("bias", bias_mem),
        data("scale", scale_mem),
        data("zp", zp_mem),
        fully_connected("fc", input_info("input"), "weights", "bias", "scale", "zp", data_types::f32, 3, 2),
        crop("crop1", input_info("fc"), tensor(1, 2, 1, 4), tensor(0, 0, 0, 0)),
        reorder("output1", input_info("crop1"), format::bfyx, data_types::f32),
        crop("crop2", input_info("fc"), tensor(1, 2, 1, 4), tensor(0, 0, 0, 4)),
        reorder("output2", input_info("crop2"), format::bfyx, data_types::f32),
        reorder("output3", input_info("fc"), format::bfyx, data_types::f32)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    auto crop_prim = network.get_primitive("crop1");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
    crop_prim = network.get_primitive("crop2");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);

    auto output = outputs.at("output1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    auto output_2 = outputs.at("output2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    auto output_3 = outputs.at("output3").get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_3, get_test_stream());

    for (size_t i = 0; i < out3.size(); i++)
        ASSERT_EQ(output_ptr_3[i], out3[i]);

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);
}

TEST(prepare_buffer_fusing, in_place_crop_dynamic) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{-1, -1, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
    auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });

    int64_t axis = 2;
    set_values(input_mem, { -0.5f,  2.0f,  0.5f,  1.0f,
                             0.5f, -2.0f, -0.5f, -1.0f });
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, { 2, 6 });
    set_values<uint8_t>(weights_mem, { 1,  2,  3,  4,
                                       5,  6,  7,  8,
                                       9, 10, 11, 12,
                                      13, 14, 15,  0,
                                      15, 14, 13, 12,
                                      11, 10,  9,  8,
                                       7,  6,  5,  4,
                                       3,  2,  1,  0});
    set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f });
    set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });
    set_values(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f });

    std::vector<float> out1 = { 13.f, 58.f, -11.f, -62.f };
    std::vector<float> out2 = { -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };
    std::vector<float> out3 = { 13.f, 58.f, -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, -11.f, -62.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        data("weights", weights_mem),
        data("bias", bias_mem),
        data("scale", scale_mem),
        data("zp", zp_mem),
        fully_connected("fc", input_info("input"), "weights", "bias", "scale", "zp", data_types::f32, 3, 2),
        crop("crop1", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reorder("output1", input_info("crop1"), format::bfyx, data_types::f32),
        crop("crop2", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape", input_info("crop2"), true, std::vector<int64_t>{0, 0, 3, 2}, ov::PartialShape{-1, -1, 3, 2}, cldnn::reshape::reshape_mode::base),
        reorder("output2", input_info("reshape"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        reorder("output3", input_info("fc"), format::bfyx, data_types::f32)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    auto output = outputs.at("output1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("output2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);

    auto output_3 = outputs.at("output3").get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_3, get_test_stream());

    for (size_t i = 0; i < out3.size(); i++)
        ASSERT_EQ(output_ptr_3[i], out3[i]);
}

TEST(prepare_buffer_fusing, in_place_crop_dynamic_reshape_unsqueeze) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{-1, -1, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
    auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });

    int64_t axis = 2;
    set_values(input_mem, { -0.5f,  2.0f,  0.5f,  1.0f,
                             0.5f, -2.0f, -0.5f, -1.0f });
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, { 2, 6 });
    set_values<uint8_t>(weights_mem, { 1,  2,  3,  4,
                                       5,  6,  7,  8,
                                       9, 10, 11, 12,
                                      13, 14, 15,  0,
                                      15, 14, 13, 12,
                                      11, 10,  9,  8,
                                       7,  6,  5,  4,
                                       3,  2,  1,  0});
    set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f });
    set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });
    set_values(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f });

    std::vector<float> out1 = { 13.f, 58.f, -11.f, -62.f };
    std::vector<float> out2 = { -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };
    std::vector<float> out3 = { 13.f, 58.f, -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, -11.f, -62.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        data("weights", weights_mem),
        data("bias", bias_mem),
        data("scale", scale_mem),
        data("zp", zp_mem),
        fully_connected("fc", input_info("input"), "weights", "bias", "scale", "zp", data_types::f32, 3, 2),
        crop("crop1", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reorder("output1", input_info("crop1"), format::bfyx, data_types::f32),
        crop("crop2", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape", input_info("crop2"), false, std::vector<int64_t>{1}, ov::PartialShape{-1, 1, -1, 6}, cldnn::reshape::reshape_mode::unsqueeze),
        reorder("output2", input_info("reshape"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        reorder("output3", input_info("fc"), format::bfyx, data_types::f32)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    auto output = outputs.at("output1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("output2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);

    auto output_3 = outputs.at("output3").get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_3, get_test_stream());

    for (size_t i = 0; i < out3.size(); i++)
        ASSERT_EQ(output_ptr_3[i], out3[i]);
}

TEST(prepare_buffer_fusing, in_place_crop_dynamic_reshape_squeeze_crop_axis) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    auto in_layout = layout{ ov::PartialShape{2, -1, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ {2, 2, 4}, data_types::f32, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::f32, format::bfyx });
    auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });
    auto eltwise_scale = engine.allocate_memory({ {1, 1}, data_types::f32, format::bfyx });

    int64_t axis = 0;
    auto input_data = rg.generate_random_1d<float>(input_mem->count(), 0, 1);
    auto weights_data = rg.generate_random_1d<float>(weights_mem->count(), 0, 1);
    auto bias_data = rg.generate_random_1d<float>(bias_mem->count(), 0, 1);

    set_values(input_mem, input_data);
    set_values(weights_mem, weights_data);
    set_values(bias_mem, bias_data);
    set_values(eltwise_scale, { 1.f });
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, { 1, 1 });

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        data("eltwise_data", eltwise_scale),
        data("weights", weights_mem),
        data("bias", bias_mem),
        fully_connected("fc", input_info("input"), "weights", "bias", data_types::f32, 3, 2),
        crop("crop1", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reorder("first_half", input_info("crop1"), format::bfyx, data_types::f32),
        crop("crop2", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape", input_info("crop2"), false, std::vector<int64_t>{0}, ov::PartialShape{-1, 8}, cldnn::reshape::reshape_mode::squeeze),
        eltwise("multiply", { input_info("reshape"), input_info("eltwise_data") }, eltwise_mode::prod),
        reorder("second_half", input_info("multiply"), format::bfyx, data_types::f32),
        reorder("full_output", input_info("fc"), format::bfyx, data_types::f32)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    ASSERT_FALSE(network.get_primitive("crop2")->can_be_optimized());

    auto outputs = network.execute();

    auto full_output_mem = outputs.at("full_output").get_memory();
    cldnn::mem_lock<float> full_output(full_output_mem, get_test_stream());

    auto first_half_mem = outputs.at("first_half").get_memory();
    cldnn::mem_lock<float> first_half(first_half_mem, get_test_stream());

    auto second_half_mem = outputs.at("second_half").get_memory();
    cldnn::mem_lock<float> second_half(second_half_mem, get_test_stream());

    for (size_t i = 0; i < full_output.size() / 2; i++)
        ASSERT_EQ(first_half[i], full_output[i]) << i;

    for (size_t i = 0; i < full_output.size() / 2; i++)
        ASSERT_EQ(second_half[i], full_output[full_output.size() / 2 + i]) << i;
}

TEST(prepare_buffer_fusing, in_place_crop_dynamic_split_lengths) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{-1, -1, -1}, data_types::f32, format::bfyx};
    auto in2_layout = layout{ ov::PartialShape{-1, -1}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
    auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto shapeof_mem = engine.allocate_memory({ {2, 6}, data_types::f32, format::bfyx });

    int64_t axis = 2;
    set_values(input_mem, { -0.5f,  2.0f,  0.5f,  1.0f,
                             0.5f, -2.0f, -0.5f, -1.0f });
    set_values<int64_t>(axis_mem, {axis});
    set_values(shapeof_mem, { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f});
    set_values<uint8_t>(weights_mem, { 1,  2,  3,  4,
                                       5,  6,  7,  8,
                                       9, 10, 11, 12,
                                      13, 14, 15,  0,
                                      15, 14, 13, 12,
                                      11, 10,  9,  8,
                                       7,  6,  5,  4,
                                       3,  2,  1,  0});
    set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f });
    set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });
    set_values(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f });

    std::vector<float> out1 = { 13.f, 58.f, -11.f, -62.f };
    std::vector<float> out2 = { -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };
    std::vector<float> out3 = { 13.f, 58.f, -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, -11.f, -62.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        input_layout("input_shapeof", in2_layout),
        data("axis", axis_mem),
        data("weights", weights_mem),
        data("bias", bias_mem),
        data("scale", scale_mem),
        data("zp", zp_mem),
        fully_connected("fc", input_info("input"), "weights", "bias", "scale", "zp", data_types::f32, 3, 2),
        shape_of("shapeof", input_info("input_shapeof"), cldnn::data_types::i64),
        crop("crop1", { input_info("fc"), input_info("axis"), input_info("shapeof") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reorder("output1", input_info("crop1"), format::bfyx, data_types::f32),
        crop("crop2", { input_info("fc"), input_info("axis"), input_info("shapeof") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape", input_info("crop2"), true, std::vector<int64_t>{0, 0, 3, 2}, ov::PartialShape{-1, -1, 3, 2}, cldnn::reshape::reshape_mode::base),
        reorder("output2", input_info("reshape"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        reorder("output3", input_info("fc"), format::bfyx, data_types::f32)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);
    network.set_input_data("input_shapeof", shapeof_mem);

    std::map<cldnn::primitive_id, cldnn::network_output> outputs;
    EXPECT_NO_THROW(outputs = network.execute());

    auto output = outputs.at("output1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("output2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        ASSERT_EQ(output_ptr_2[i], out2[i]);

    auto output_3 = outputs.at("output3").get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_3, get_test_stream());

    for (size_t i = 0; i < out3.size(); i++)
        ASSERT_EQ(output_ptr_3[i], out3[i]);
}

TEST(prepare_buffer_fusing, in_place_crop_dynamic_mvn) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{-1, -1, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({ {1, 2, 4}, data_types::f32, format::bfyx });
    auto weights_mem = engine.allocate_memory({ {8, 4}, data_types::u8, format::bfyx });
    auto bias_mem = engine.allocate_memory({ {1, 1, 8}, data_types::f32, format::bfyx });
    auto scale_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto zp_mem = engine.allocate_memory({ {8, 1}, data_types::f32, format::bfyx });
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });

    int64_t axis = 2;
    set_values(input_mem, { -0.5f,  2.0f,  0.5f,  1.0f,
                             0.5f, -2.0f, -0.5f, -1.0f });
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, { 2, 6 });
    set_values<uint8_t>(weights_mem, { 1,  2,  3,  4,
                                       5,  6,  7,  8,
                                       9, 10, 11, 12,
                                      13, 14, 15,  0,
                                      15, 14, 13, 12,
                                      11, 10,  9,  8,
                                       7,  6,  5,  4,
                                       3,  2,  1,  0});
    set_values(bias_mem, { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 2.0f });
    set_values(scale_mem, { 2.0f, 4.0f, -2.0f, -4.0f, 0.5f, -0.5f, 2.0f, 2.0f });
    set_values(zp_mem, { 1.0f, 2.0f, 2.0f, 1.0f, 4.0f, 1.0f, 6.0f, 2.0f });

    std::vector<float> out1 = { 13.f, 58.f, -11.f, -62.f };
    std::vector<float> out2 = { -24.0833f, -81.0833f, 45.4167f, 8.91667f, 27.9167f, 22.9167f, 27.75f, 70.75f, -37.75f, -23.25f, -16.25f, -21.25f };
    std::vector<float> out3 = { 13.f, 58.f, -51.f, -108.f, 18.5f, -18.f, 1.f, -4.f, -11.f, -62.f, 57.f, 100.f, -8.5f, 6.f, 13.f, 8.f };

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        data("weights", weights_mem),
        data("bias", bias_mem),
        data("scale", scale_mem),
        data("zp", zp_mem),
        fully_connected("fc", input_info("input"), "weights", "bias", "scale", "zp", data_types::f32, 3, 2),
        crop("crop1", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reorder("output1", input_info("crop1"), format::bfyx, data_types::f32),
        crop("crop2", { input_info("fc"), input_info("axis"), input_info("splits_length") }, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape", input_info("crop2"), true, std::vector<int64_t>{0, 0, 3, 2}, ov::PartialShape{-1, -1, 3, 2}, cldnn::reshape::reshape_mode::base),
        mvn("mvn", input_info("reshape"), false, 1e-10f, false, {2, 3}),
        reorder("output2", input_info("mvn"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        reorder("output3", input_info("fc"), format::bfyx, data_types::f32)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    auto output = outputs.at("output1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        ASSERT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("output2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++) {
        ASSERT_NEAR(output_ptr_2[i], out2[i], 0.0001);
    }

    auto output_3 = outputs.at("output3").get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_3, get_test_stream());

    for (size_t i = 0; i < out3.size(); i++)
        ASSERT_EQ(output_ptr_3[i], out3[i]);
}

// Testing for implicit crop along batch axis and outer padding optimzing.
// Outer padding opt includes opt out of reshape and reorder which has padded input only in batch axis
// This optimzing also includes offset(outer axis padded input) handling of oneDNN primitive.
TEST(prepare_buffer_fusing, test_implicit_crop_and_outerpadding) {
    auto& engine = get_test_engine();

    auto in_input = engine.allocate_memory({ data_types::i32, format::bfzyx, tensor{ 3, 6, 2, 2, 2 } }); // Dictionary
    // Indexes
    auto input_idx1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1 } });
    int64_t axis = 0;

    // Change dimension (optimized out)
    layout reorder_layout(data_types::f32, format::bfyx, { 1, 6, 4, 2 });

    tests::set_random_values<int32_t>(in_input);

    set_values(input_idx1, { 0 });

    topology topology;
    topology.add(input_layout("Input", in_input->get_layout()));
    topology.add(input_layout("Input_idx_1", input_idx1->get_layout()));
    topology.add(reorder("reorder_input", input_info("Input"), format::bfzyx, data_types::f32));
    topology.add(gather("gather1", input_info("reorder_input"), input_info("Input_idx_1"), axis, 5, ov::Shape{1, 6, 2, 2, 2}));
    topology.add(reorder("gather1_reorder", input_info("gather1"), reorder_layout));
    topology.add(reshape("reshape1", input_info("gather1_reorder"), tensor(6, 2, 2, 2)));
    topology.add(crop("crop", input_info("reorder_input"), tensor{1, 6, 2, 2, 2}, tensor(1, 0, 0, 0, 0)));
    topology.add(reorder("gather2_reorder", input_info("crop"), reorder_layout));
    topology.add(reshape("reshape2", input_info("gather2_reorder"), tensor(6, 2, 2, 2)));
    topology.add(gemm("gemm", { input_info("reshape1"), input_info("reshape2") }, data_types::f32, false, true));
    topology.add(reorder("reorder_output", input_info("gemm"), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);

    network.set_input_data("Input", in_input);
    network.set_input_data("Input_idx_1", input_idx1);

    auto outputs = network.execute();
    auto output = outputs.at("reorder_output").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);
    ref_network.set_input_data("Input", in_input);
    ref_network.set_input_data("Input_idx_1", input_idx1);

    auto ref_outputs = ref_network.execute();
    auto ref_output = ref_outputs.at("reorder_output").get_memory();
    cldnn::mem_lock<int8_t> ref_output_ptr(ref_output, get_test_stream());

    int crop_batch_num = 6;
    int crop_feature_num = 2;
    int crop_y_size = 2;
    int crop_x_size = 2;
    for (int b = 0; b < crop_batch_num; ++b) {
        for (int f = 0; f < crop_feature_num; ++f) {
            for (int y = 0; y < crop_y_size; ++y) {
                for (int x = 0; x < crop_x_size; ++x) {
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    ASSERT_EQ(output_ptr[output_linear_id], ref_output_ptr[output_linear_id]);
                }
            }
        }
    }

    auto crop_prim = network.get_primitive("gather1");
    ASSERT_EQ(crop_prim->can_be_optimized(), false);
    crop_prim = network.get_primitive("crop");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
    auto reorder_prim = network.get_primitive("gather1_reorder");
    ASSERT_EQ(reorder_prim->can_be_optimized(), true);
    reorder_prim = network.get_primitive("gather2_reorder");
    ASSERT_EQ(reorder_prim->can_be_optimized(), true);
    auto reshape_prim = network.get_primitive("reshape1");
    ASSERT_EQ(reshape_prim->can_be_optimized(), true);
}

// For conv, Check padded input and weight propagated by implicit crop are handled properly
TEST(prepare_buffer_fusing, test_implicit_crop_and_outerpadding_conv) {
    auto& engine = get_test_engine();
    const std::string no_bias = "";
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 5, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 3, 2 } });

    set_values(input, { 2.1f, 3.4f, 4.5f, 5.5f, 6.8f, 2.9f, 2.3f, 3.5f, 4.4f, 6.6f,
                        1.8f, 2.9f, 3.4f, 4.1f, 5.4f, 6.8f, 7.1f, 8.2f, 9.2f, 1.9f,
                        1.1f, 2.4f, 3.5f, 4.5f, 5.8f, 2.9f, 2.3f, 3.5f, 4.4f, 6.6f,
                        3.8f, 3.9f, 3.4f, 5.1f, 1.4f, 1.8f, 1.1f, 1.2f, 1.2f, 1.9f});
    set_values<float>(weights, { 2.f, 4.f, 2.f, 4.f, 2.f, 4.f,
                                 1.f, 2.f, 1.f, 2.f, 1.f, 2.f});
    VVF<float> output_vec = {
        { 20.0f, 27.0f, 38.0f },
        { 17.0f, 19.0f, 19.0f } };

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("to_int", input_info("input"), { data_types::i8, format::bfyx, { 2, 1, 5, 4 } }),
        crop("crop_input", input_info("to_int"), tensor{ 1, 1, 5, 4 }, tensor(1, 0, 0, 0)),
        data("weights", weights),
        reorder("to_weight", input_info("weights"), { data_types::i8, format::bfyx, { 2, 1, 3, 2 } }),
        crop("crop_weight", input_info("to_weight"), tensor{ 1, 1, 3, 2 }, tensor(1, 0, 0, 0)),
        convolution("conv", input_info("crop_input"), "crop_weight", no_bias, 1, {2, 1}, {1, 1}, {0, 0}, {0, 0}, false),
        reorder("output", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();
    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(y_size, 2);
    ASSERT_EQ(x_size, 3);
    ASSERT_EQ(f_size, 1);
    ASSERT_EQ(b_size, 1);
    for (int y = 0; y < y_size; ++y) {
        for (int x = 0; x < x_size; ++x) {
            ASSERT_EQ(output_vec[y][x], output_ptr[y * x_size + x]);
        }
    }

    if (!engine.get_device_info().supports_immad) {
        auto crop_prim = network.get_primitive("crop_input");
        ASSERT_EQ(crop_prim->can_be_optimized(), true);
    }
}

// For deconv, Check padded input and weight propagated by implicit crop are handled properly
TEST(prepare_buffer_fusing, test_implicit_crop_and_outerpadding_deconv) {
    auto& engine = get_test_engine();
    const std::string no_bias = "";
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 4, 1, 2, 2 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 1, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, { 16.f, 1.0f, 3.f, 4.5f, 9.f, 1.f, 3.f, 4.5f,
                        8.f,  0.5f, 6.f, 9.f,  1.f, 3.f, 2.f, 4.f});
    set_values<float>(weights, { -4.0f, 1.0f, 7.0f, 3.0f,
                                 -2.0f, 2.0f, 7.0f, -0.5f});
    set_values(biases, { 1.0f });
    std::vector<float> expected_output_vec = { -3.f, 4.5f, 13.f, -17.f, 0.5f, 22.f, 5.f, -7.f };

    topology topology(
        input_layout("input", input->get_layout()),
        reorder("to_int", input_info("input"), { data_types::f16, format::bfyx, { 4, 1, 2, 2 } }),
        crop("crop_input", input_info("to_int"), tensor{ 2, 1, 2, 2 }, tensor(2, 0, 0, 0)),
        data("weights", weights),
        data("biases", biases),
        reorder("to_weight", input_info("weights"), { data_types::f16, format::bfyx, { 2, 1, 2, 2 } }),
        crop("crop_weight", input_info("to_weight"), tensor{ 1, 1, 2, 2 }, tensor(1, 0, 0, 0)),
        deconvolution("deconv", input_info("crop_input"), { "crop_weight" }, { "biases" }, { 2, 2 }, { 1, 1 }),
        reorder("output", input_info("deconv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "output");

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
        ASSERT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);

    if (!engine.get_device_info().supports_immad) {
        auto crop_prim = network.get_primitive("crop_input");
        ASSERT_EQ(crop_prim->can_be_optimized(), true);
    }
}

TEST(prepare_buffer_fusing, test_checking_padding_supported) {
    auto& engine = get_test_engine();
    auto in_layout1 = layout{ ov::PartialShape{2, 36, 57, 57}, data_types::f16, format::fs_b_yx_fsv32};
    auto in_layout2 = layout{ ov::PartialShape{2, 72, 57, 57}, data_types::f16, format::fs_b_yx_fsv32};
    auto in_layout3 = layout{ ov::PartialShape{2, 144, 57, 57}, data_types::f16, format::fs_b_yx_fsv32};

    auto padding1 = padding({0,18,1,1}, {0,0,0,0});
    auto padding2 = padding({0,0,0,0}, {0,0,0,0});
    auto padding3 = padding({0,0,0,0}, {0,0,0,0});

    auto resample1 = resample("interp1", input_info("input1"), in_layout1.get_tensor(), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST);
    resample1.output_paddings = {padding1};
    auto resample2 = resample("interp2", input_info("input2"), in_layout2.get_tensor(), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST);
    resample2.output_paddings = {padding2};
    auto resample3 = resample("interp3", input_info("input3"), in_layout3.get_tensor(), 1, ov::op::v4::Interpolate::InterpolateMode::NEAREST);
    resample3.output_paddings = {padding3};

    topology topology(
        input_layout("input1", in_layout1),
        input_layout("input2", in_layout2),
        input_layout("input3", in_layout3),
        resample1,
        resample2,
        resample3,
        concatenation("concat", {input_info("interp1"), input_info("interp2"), input_info("interp3")}, 1),
        reorder("reorder", input_info("concat"), format::fs_b_yx_fsv32, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto program = program::build_program(engine, topology, config, false, true);
    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*program);
    ASSERT_NE(program, nullptr);

    auto& concat = program->get_node("concat");
    ASSERT_EQ(concat.can_be_optimized(), false);
}

TEST(prepare_buffer_fusing, skip_in_place_concat_padding_in_non_concat_axis_of_dynamic) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto in_layout = layout{ ov::PartialShape{ov::Dimension::dynamic(), 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                             data_types::f16, format::bfyx};

    auto begin = engine.allocate_memory({ ov::PartialShape{4}, data_types::i64, format::bfyx });
    auto end = engine.allocate_memory({ ov::PartialShape{4}, data_types::i64, format::bfyx });
    auto strides = engine.allocate_memory({ ov::PartialShape{4}, data_types::i64, format::bfyx });
    set_values<int64_t>(begin, {0, 0, 0, 0});
    set_values<int64_t>(end, {0, 0, 0, 9223372036854775807 });
    set_values<int64_t>(strides, {1, 1, 1, 2});

    auto concat_padding = padding({0,0,1,1}, {0,0,1,1});


    auto in_static_layout = layout{ ov::PartialShape{1, 3, 320, 640}, data_types::f16, format::bfyx};
    auto input1_mem = engine.allocate_memory(in_static_layout);
    auto input2_mem = engine.allocate_memory(in_static_layout);
    auto input3_mem = engine.allocate_memory(in_static_layout);
    auto input4_mem = engine.allocate_memory(in_static_layout);

    auto in1 = rg.generate_random_1d<ov::float16>(input1_mem->count(), 0, 1);
    auto in2 = rg.generate_random_1d<ov::float16>(input2_mem->count(), 0, 1);
    auto in3 = rg.generate_random_1d<ov::float16>(input3_mem->count(), 0, 1);
    auto in4 = rg.generate_random_1d<ov::float16>(input4_mem->count(), 0, 1);

    set_values<ov::float16>(input1_mem, in1);
    set_values<ov::float16>(input2_mem, in2);
    set_values<ov::float16>(input3_mem, in3);
    set_values<ov::float16>(input4_mem, in4);

    auto concat_prim = concatenation("concat", {input_info("strided_slice1"), input_info("strided_slice2"), input_info("strided_slice3"), input_info("strided_slice4")}, 1);
    concat_prim.output_paddings = {concat_padding};
    topology topology(
        input_layout("input1", in_layout),
        input_layout("input2", in_layout),
        input_layout("input3", in_layout),
        input_layout("input4", in_layout),
        data("begin", begin),
        data("end", end),
        data("strides", strides),
        strided_slice("strided_slice1", input_info("input1"), input_info("begin"),
                               input_info("end"), input_info("strides"), {1, 1, 1, 0}, {1, 1, 1, 0}, {}, {}, {}, {}),
        strided_slice("strided_slice2", input_info("input2"), input_info("begin"),
                               input_info("end"), input_info("strides"), {1, 1, 1, 0}, {1, 1, 1, 0}, {}, {}, {}, {}),
        strided_slice("strided_slice3", input_info("input3"), input_info("begin"),
                               input_info("end"), input_info("strides"), {1, 1, 1, 0}, {1, 1, 1, 0}, {}, {}, {}, {}),
        strided_slice("strided_slice4", input_info("input4"), input_info("begin"),
                               input_info("end"), input_info("strides"), {1, 1, 1, 0}, {1, 1, 1, 0}, {}, {}, {}, {}),
        concat_prim,
        reorder("reorder", input_info("concat"), format::fs_b_yx_fsv32, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto program = program::build_program(engine, topology, config, false, true);
    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*program);
    ASSERT_NE(program, nullptr);

    auto& concat = program->get_node("concat");
    ASSERT_EQ(concat.can_be_optimized(), false);

    network network(engine, topology, config);
    network.set_input_data("input1", input1_mem);
    network.set_input_data("input2", input2_mem);
    network.set_input_data("input3", input3_mem);
    network.set_input_data("input4", input4_mem);
    auto outputs = network.execute();

    const auto& concat_inst = network.get_primitive("concat");
    ASSERT_EQ(concat_inst->can_be_optimized(), false);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(prepare_buffer_fusing, in_place_onednn_concat_static) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
       return;

    auto in_layout1 = layout{ ov::PartialShape{1, 1, 4, 2}, data_types::f32, format::bfyx };
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 2}, data_types::f32, format::bfyx };

    topology topology;
    topology.add(input_layout("input1", in_layout1));
    topology.add(input_layout("input2", in_layout2));
    topology.add(reorder("reorder1", input_info("input1"), format::bfyx, data_types::f16));
    topology.add(reorder("reorder2", input_info("input2"), format::bfyx, data_types::f16));
    topology.add(concatenation("concat", { input_info("reorder1"), input_info("reorder2") }, 1));
    topology.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    network network(engine, topology, config);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);

    set_values<float>(input_memory1,
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    set_values<float>(input_memory2,
                       {11.f, 22.f, 33.f, 44.f, 55.f, 66.f, 77.f, 88.f,
                        111.f, 222.f, 333.f, 444.f, 555.f, 666.f, 777.f, 888.f});

    network.set_input_data("input1", input_memory1);
    network.set_input_data("input2", input_memory2);

    std::vector<float> ref_output = {
                        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                        11.f, 22.f, 33.f, 44.f, 55.f, 66.f, 77.f, 88.f,
                        111.f, 222.f, 333.f, 444.f, 555.f, 666.f, 777.f, 888.f};

    std::map<cldnn::primitive_id, cldnn::network_output> output;

    EXPECT_NO_THROW(output = network.execute());
    auto out_l = network.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());
    cldnn::mem_lock<float> input1_ptr(input_memory1, get_test_stream());
    cldnn::mem_lock<float> input2_ptr(input_memory2, get_test_stream());

    const auto& concat_node_n = network.get_primitive("concat")->get_node();
    auto concat_mem = network.get_primitive("concat")->output_memory_ptr();
    auto reorder1_mem = network.get_primitive("reorder1")->output_memory_ptr();
    auto reorder2_mem = network.get_primitive("reorder2")->output_memory_ptr();

    ASSERT_EQ(concat_mem.get(), reorder1_mem.get());
    ASSERT_EQ(concat_mem.get(), reorder2_mem.get());
    ASSERT_TRUE(concat_node_n.can_be_optimized());

    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }
}
#endif  // ENABLE_ONEDNN_FOR_GPU

TEST(prepare_buffer_fusing, inner_axis_data_offset_with_gemm_user) {
    tests::random_generator rg(GET_SUITE_NAME);

    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{1, 6, 16, 16}, data_types::f16, format::bfyx };
    auto crop_layout = layout{ ov::PartialShape{1, 6, 8, 16}, data_types::f16, format::bfyx };

    auto input_memory = engine.allocate_memory(in_layout);
    auto input_data = rg.generate_random_1d<float>(input_memory->count(), -1, 1);
    
    auto offsets1 = tensor{0, 0, 0, 0};
    auto offsets2 = tensor{0, 0, 8, 0};

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(crop("crop1", input_info("input"), crop_layout.get_tensor(), offsets1));
    topology.add(permute("permute", input_info("crop1"), {0, 1, 3, 2}));
    topology.add(crop("crop2", input_info("input"), crop_layout.get_tensor(), offsets2));
    topology.add(gemm("gemm", {input_info("permute"), input_info("crop2")}, data_types::f16, false, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);

    auto& crop_node = prog->get_node("crop2").as<crop>();
    ASSERT_FALSE(crop_node.can_be_optimized());
}
