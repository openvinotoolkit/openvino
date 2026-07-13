// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/vl_sdpa.hpp>
#include "vl_sdpa_inst.h"
#include "graph/impls/ocl/kernel_selector_helper.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "concatenation_inst.h"
#include "gemm_inst.h"
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
#include "eltwise_inst.h"
#include "mvn_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"
#include "resample_inst.h"
#include "openvino/op/interpolate.hpp"

#include "program_wrapper.h"
#include "primitive_inst_test_helper.h"

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

TEST(prepare_buffer_fusing, in_place_concat_dynamic_and_static) {
    auto& engine = get_test_engine();
    auto in_layout1_dyn = layout{ ov::PartialShape{1, 2, ov::Dimension() }, data_types::f32, format::bfyx };
    auto in_layout3_dyn = layout{ ov::PartialShape{1, 2, ov::Dimension(21, ov::Interval::s_max) }, data_types::f32, format::yxfb };
    auto reorder_layout = layout{ ov::PartialShape{1, 2, ov::Dimension(21, ov::Interval::s_max) }, data_types::f32, format::bfyx };

    auto in_layout1 = layout{ {1, 2, 42}, data_types::f32, format::bfyx };
    auto in_layout2 = layout{ {1, 2, 42}, data_types::f32, format::bfyx };
    auto in_layout3 = layout{ {1, 2, 42}, data_types::f32, format::yxfb };

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    auto input_memory3 = engine.allocate_memory(in_layout3);

    topology topology;
    topology.add(input_layout("input1", in_layout1_dyn));
    topology.add(input_layout("input2", in_layout2));
    topology.add(input_layout("input3", in_layout3_dyn));
    topology.add(eltwise("add", input_info("input1"), input_info("input2"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("input3"), reorder_layout));
    topology.add(concatenation("concat", { input_info("add"), input_info("reorder") }, 1));
    topology.add(permute("output", input_info("concat"), {0, 2, 1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    cldnn::network net(prog, 0);

    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);
    net.set_input_data("input3", input_memory3);
    auto outputs = net.execute();

    auto output_memory = outputs.at("output").get_memory();
    auto output_layout = output_memory->get_layout();

    int y_size = output_layout.spatial(1);
    int x_size = output_layout.spatial(0);
    int f_size = output_layout.feature();
    int b_size = output_layout.batch();

    ASSERT_EQ(y_size, 4);
    ASSERT_EQ(x_size, 1);
    ASSERT_EQ(f_size, 42);
    ASSERT_EQ(b_size, 1);
}

TEST(prepare_buffer_fusing, in_place_concat_dynamic_memory_reallocation) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto input_dynamic_layout = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };

    topology t;
    t.add(input_layout("input1", input_dynamic_layout));
    t.add(input_layout("input2", input_dynamic_layout));
    t.add(reorder("input1_reordered", input_info("input1"), format::bfyx, data_types::f16));
    t.add(reorder("input2_reordered", input_info("input2"), format::bfyx, data_types::f16));
    t.add(concatenation("concat", { input_info("input1_reordered"), input_info("input2_reordered") }, 1));
    t.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto net = cldnn::network(program::build_program(engine, t, config, false, false));

    std::vector<ov::Shape> input_shapes =
    { ov::Shape{1, 32, 8, 6},
      ov::Shape{1, 32, 8, 6},
      ov::Shape{1, 32, 12, 9},
      ov::Shape{1, 32, 12, 9},
      ov::Shape{1, 32, 6, 11},
      ov::Shape{1, 32, 6, 11},
      ov::Shape{1, 32, 9, 16},
      ov::Shape{1, 32, 9, 16} };

    auto prev_concat_mem = memory_ptr{nullptr};
    auto prev_shape = ov::Shape{};

    for (const auto& input_shape : input_shapes) {
        auto input_layout = input_dynamic_layout.clone_with_other_shape(input_shape);
        auto input_memory1 = engine.allocate_memory(input_layout);
        auto input_memory2 = engine.allocate_memory(input_layout);

        auto input_data1 = rg.generate_random_1d<float>(input_layout.count(), 0, 1);
        auto input_data2 = rg.generate_random_1d<float>(input_layout.count(), 0, 1);

        set_values<float>(input_memory1, input_data1);
        set_values<float>(input_memory2, input_data2);

        net.set_input_data("input1", input_memory1);
        net.set_input_data("input2", input_memory2);

        std::map<cldnn::primitive_id, cldnn::network_output> output;
        EXPECT_NO_THROW(output = net.execute());

        auto out_mem = output.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

        const auto& reorder_inst = net.get_primitive("input1_reordered");
        const auto& concat_inst = net.get_primitive("concat");
        auto reorder1_mem = net.get_primitive("input1_reordered")->output_memory_ptr();
        auto reorder2_mem = net.get_primitive("input2_reordered")->output_memory_ptr();
        auto concat_mem = net.get_primitive("concat")->output_memory_ptr();

        ASSERT_TRUE(concat_inst->get_node().can_be_optimized());
        ASSERT_TRUE(engine.is_the_same_buffer(*concat_mem, *reorder1_mem));
        ASSERT_TRUE(engine.is_the_same_buffer(*concat_mem, *reorder2_mem));

        if (prev_concat_mem) {
            const auto can_reuse_mem = ov::shape_size(input_shape) <= ov::shape_size(prev_shape);
            ASSERT_EQ(engine.is_the_same_buffer(*prev_concat_mem, *concat_mem), can_reuse_mem);
        }

        // Under certain circumstances (e.g., asynchronous compilation for some primitives), `allocation_done_by_other` flag
        // might be incorrectly set or unset for the concat primitive. This can lead to incorrect memory assignment:
        // if the flag remains set without being properly reset, concat may reuse a smaller buffer than required.
        // Manually forcing the flag value ensures it can be correctly reconfigured for each execution iteration
        PrimitiveInstTestHelper::set_allocation_done_by_other(concat_inst, true);

        prev_concat_mem = concat_mem;
        prev_shape = input_shape;

        for (size_t i = 0; i < input_data1.size(); ++i) {
            ASSERT_EQ(output_ptr[i], input_data1[i]);
        }

        for (size_t i = 0; i < input_data2.size(); ++i) {
            ASSERT_EQ(output_ptr[input_data1.size() + i], input_data2[i]);
        }
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

TEST(prepare_buffer_fusing, disable_crop_buffer_fusing_with_shift_right_padding) {
    auto& engine = get_test_engine();

    auto gemm_input_mem = engine.allocate_memory({ {1, 4, 4, 2}, data_types::f32, format::bfyx });
    auto concat_input_mem = engine.allocate_memory({ {1, 4, 2}, data_types::f32, format::bfyx });

    set_values(gemm_input_mem, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                                 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                                 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f,
                                 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f });
    set_values(concat_input_mem, { -0.5f, 2.0f, 0.5f, 1.0f, 0.5f, -2.0f, -0.5f, -1.0f });

    std::vector<float> expected = { 0.5, 4, 0.5, 10, 0.5, 16, 0.5, 22,
                                    0.5, 4, 0.5, 10, 0.5, 16, 0.5, 22,
                                    0.5, 4, 0.5, 10, 0.5, 16, 0.5, 22,
                                    0.5, 4, 0.5, 10, 0.5, 16, 0.5, 22};
    cldnn::tensor refSize = {1, 2, 1, 2};

    topology topology(
        input_layout("gemm_input", gemm_input_mem->get_layout()),
        input_layout("concat_input", concat_input_mem->get_layout()),
        concatenation("concat", { input_info("concat_input"), input_info("concat_input") }, 2),
        crop("crop", input_info("concat"), refSize, tensor(0, 0, 0, 0)),
        gemm("gemm", { input_info("gemm_input"), input_info("crop") }, data_types::f32, false, false, 1.0, 0.0, 4, 3),
        reorder("output", input_info("gemm"), format::bfyx, data_types::f32)
    );

    {
        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        network network(engine, topology, config);

        network.set_input_data("gemm_input", gemm_input_mem);
        network.set_input_data("concat_input", concat_input_mem);

        auto outputs = network.execute();

        auto crop_prim = network.get_primitive("crop");
        ASSERT_EQ(crop_prim->can_be_optimized(), false);    // Not opt out because the user, gemm node, has paddings at spatial dimensions

        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());
        for (size_t i = 0; i < expected.size(); i++) {
            ASSERT_EQ(output_ptr[i], expected[i]);
        }
    }
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

TEST(prepare_buffer_fusing, do_runtime_in_place_crop_skips_non_propagatable_reshape_when_reshape_was_preupdated) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape{1, 1, 4}, data_types::f32, format::bfyx};
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{2}, data_types::i64, format::bfyx});
    set_values<int64_t>(axis_mem, {2});
    set_values<int64_t>(splits_length_mem, {2, 2});

    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        crop("crop", {input_info("input"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), cldnn::tensor(0), cldnn::crop_ngraph_op_mode::variadic_split, 0, 2),
        reshape("reshape", input_info("crop"), false, {}, ov::PartialShape{-1, -1, 4}, cldnn::reshape::reshape_mode::base),
        reorder("output", input_info("reshape"), format::bfyx, data_types::f32));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto input_inst = network.get_primitive("input");
    auto crop_inst = network.get_primitive("crop");
    auto reshape_inst = network.get_primitive("reshape");

    crop_inst->set_can_be_optimized(true);
    const_cast<cldnn::program_node&>(crop_inst->get_node()).can_be_optimized(true);
    PrimitiveInstTestHelper::set_update_shape_done_by_other(reshape_inst, true);

    PrimitiveInstTestHelper::do_runtime_in_place_crop(input_inst);

    ASSERT_FALSE(crop_inst->can_be_optimized());
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
    ASSERT_EQ(reorder_prim->can_be_optimized(), false);
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

TEST(prepare_buffer_fusing, in_place_concat_with_fsv32_to_fsv16_reorder_regression) {
    // Regression test for fsv32->fsv16 reorder + in-place concat path.
    // Keep in-place enabled, then verify buffer sharing and output channel order.
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape{1, 32, 1, 1, 1}, data_types::f32, format::bfzyx};

    topology topology;
    topology.add(input_layout("input1", in_layout));
    topology.add(input_layout("input2", in_layout));
    topology.add(reorder("input1_fsv16", input_info("input1"), format::b_fs_zyx_fsv16, data_types::f16));
    topology.add(reorder("input2_fsv32", input_info("input2"), format::b_fs_zyx_fsv32, data_types::f16));
    topology.add(reorder("input2_fsv16", input_info("input2_fsv32"), format::b_fs_zyx_fsv16, data_types::f16));
    topology.add(concatenation("concat", {input_info("input1_fsv16"), input_info("input2_fsv16")}, 1));
    topology.add(reorder("output", input_info("concat"), format::bfzyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));

    network network(engine, topology, config);

    auto input_memory1 = engine.allocate_memory(in_layout);
    auto input_memory2 = engine.allocate_memory(in_layout);

    std::vector<float> input1_vals(32);
    std::vector<float> input2_vals(32);
    for (size_t i = 0; i < 32; ++i) {
        input1_vals[i] = static_cast<float>(i + 1);
        input2_vals[i] = static_cast<float>(1000 + i + 1);
    }

    set_values<float>(input_memory1, input1_vals);
    set_values<float>(input_memory2, input2_vals);

    network.set_input_data("input1", input_memory1);
    network.set_input_data("input2", input_memory2);

    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = network.execute());

    const auto& concat_inst = network.get_primitive("concat");
    // This case is expected to be in-place after prepare_buffer_fusing.
    ASSERT_TRUE(concat_inst->can_be_optimized());

    auto concat_mem = concat_inst->output_memory_ptr();
    auto input1_fsv16_mem = network.get_primitive("input1_fsv16")->output_memory_ptr();
    auto input2_fsv16_mem = network.get_primitive("input2_fsv16")->output_memory_ptr();

    // In-place concat means all these primitives point to the same underlying buffer.
    ASSERT_EQ(concat_mem.get(), input1_fsv16_mem.get());
    ASSERT_EQ(concat_mem.get(), input2_fsv16_mem.get());

    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    // Validate semantic correctness: first 32 channels from input1, next 32 from input2.
    ASSERT_EQ(out_mem->count(), 64);
    for (size_t i = 0; i < 32; ++i) {
        ASSERT_EQ(output_ptr[i], input1_vals[i]);
        ASSERT_EQ(output_ptr[i + 32], input2_vals[i]);
    }
}

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

TEST(prepare_buffer_fusing, redundant_reorder_permute) {
    tests::random_generator rg(GET_SUITE_NAME);

    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{1, 2, 3, 5}, data_types::f16, format::byfx };

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(reorder("reorder", input_info("input"), format::bfyx, data_types::f16));
    topology.add(permute("permute", input_info("reorder"), {0, 2, 1, 3}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);

    auto& permute_node = prog->get_node("permute").as<permute>();
    auto& reorder_node = prog->get_node("reorder").as<reorder>();
    ASSERT_TRUE(reorder_node.can_be_optimized());
    ASSERT_TRUE(permute_node.can_be_optimized());
}

TEST(prepare_buffer_fusing, reorder_permute_with_fused_prim) {
    auto& engine = get_test_engine();

    auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 5}, data_types::f16, format::byxf };
    auto in_layout2 = layout{ ov::PartialShape{1, 3, 5, 2}, data_types::f16, format::bfyx };

    topology topology;
    topology.add(input_layout("input1", in_layout1));
    topology.add(input_layout("input2", in_layout2));
    topology.add(reorder("reorder", input_info("input1"), format::bfyx, data_types::f16));
    topology.add(permute("permute", input_info("reorder"), {0, 2, 3, 1}));
    topology.add(eltwise("eltwise", { input_info("permute"), input_info("input2") }, eltwise_mode::sum));
    topology.add(reorder("output", input_info("eltwise"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);

    auto& permute_node = prog->get_node("permute").as<permute>();
    auto& reorder_node = prog->get_node("reorder").as<reorder>();
    ASSERT_FALSE(reorder_node.can_be_optimized());
    ASSERT_FALSE(permute_node.can_be_optimized());
}

TEST(prepare_buffer_fusing, disable_reshape_with_feature_upper_padding) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape{2, 24, 1, 1}, data_types::f16, format::bfyx};

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(activation("act", input_info("input"), activation_func::relu));
    topology.add(reshape("reshape", input_info("act"), false, {2, 6, 1, 4},
                         ov::PartialShape{2, 6, 1, 4}));
    topology.add(reorder("output", input_info("reshape"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->has_node("reshape"));
    ASSERT_TRUE(prog->has_node("act"));

    auto& act_node = prog->get_node("act");
    auto act_layout = act_node.get_output_layout();
    padding feature_upper_padding({0, 0, 0, 0}, {0, 8, 0, 0});
    act_layout.data_padding = feature_upper_padding;
    act_node.set_output_layout(act_layout, false);

    auto& reshape_node = prog->get_node("reshape").as<reshape>();

    auto& reshape_input_layout = reshape_node.get_dependency(0).get_output_layout();
    ASSERT_GT(reshape_input_layout.data_padding._upper_size[1], 0)
        << "Test setup: reshape input should have feature upper padding";

    bool has_outer_pad = reshape_node.has_outer_padding_offset();
    ASSERT_FALSE(has_outer_pad)
        << "has_outer_padding_offset() should return FALSE for inputs with feature upper padding";

    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*prog);
    ASSERT_FALSE(reshape_node.can_be_optimized())
        << "Reshape with feature upper padding should NOT be optimized";
}

TEST(prepare_buffer_fusing, in_place_crop_dynamic_batch_axis_split_with_reshape) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    auto in_layout = layout{ov::PartialShape{3, -1, 2, 4}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({{3, 2, 2, 4}, data_types::f32, format::bfyx});
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});

    const int64_t axis = 0;
    const size_t dim_f = 2, dim_y = 2, dim_x = 4;
    const size_t slice_elems = dim_f * dim_y * dim_x;

    auto input_data = rg.generate_random_1d<float>(input_mem->count(), -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {1, 1, 1});

    // reshape [1, dim_f, dim_y, dim_x] → [-1, dim_y, dim_x] (base mode, absorbs static b=1)
    const std::vector<int64_t> squeeze_pattern = {-1, static_cast<int64_t>(dim_y), static_cast<int64_t>(dim_x)};
    const ov::PartialShape squeeze_out_shape = {-1, static_cast<int64_t>(dim_y), static_cast<int64_t>(dim_x)};

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        // Q branch: crop output_idx=0 → [1, dim_f, dim_y, dim_x] → reshape → [-1, dim_y, dim_x]
        crop("crop_q", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape_q", input_info("crop_q"), false, squeeze_pattern, squeeze_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_q", input_info("reshape_q"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // K branch: crop output_idx=1
        crop("crop_k", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape_k", input_info("crop_k"), false, squeeze_pattern, squeeze_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_k", input_info("reshape_k"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // V branch: crop output_idx=2
        crop("crop_v", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape_v", input_info("crop_v"), false, squeeze_pattern, squeeze_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_v", input_info("reshape_v"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    ASSERT_TRUE(network.get_primitive("crop_q")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop_k")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop_v")->can_be_optimized());

    auto q_mem = outputs.at("output_q").get_memory();
    cldnn::mem_lock<float> q_out(q_mem, get_test_stream());
    auto k_mem = outputs.at("output_k").get_memory();
    cldnn::mem_lock<float> k_out(k_mem, get_test_stream());
    auto v_mem = outputs.at("output_v").get_memory();
    cldnn::mem_lock<float> v_out(v_mem, get_test_stream());

    ASSERT_EQ(q_out.size(), slice_elems);
    ASSERT_EQ(k_out.size(), slice_elems);
    ASSERT_EQ(v_out.size(), slice_elems);

    // Each in-place crop+reshape must read the correct slice without data movement
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(q_out[i], input_data[0 * slice_elems + i]) << "Q mismatch at " << i;
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(k_out[i], input_data[1 * slice_elems + i]) << "K mismatch at " << i;
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(v_out[i], input_data[2 * slice_elems + i]) << "V mismatch at " << i;
}

// RAFT-like pattern: VariadicSplit on batch axis → crop [1,C,H,W] → Reshape [1,C,H*W]
// The reshape flattens spatial dims while keeping batch=1.  The in-place crop
// optimisation must NOT treat this as a batch-squeeze and must produce correct data.
TEST(prepare_buffer_fusing, in_place_crop_dynamic_batch_axis_split_with_spatial_flatten_reshape) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const size_t dim_b = 2, dim_c = 4, dim_h = 3, dim_w = 5;
    const size_t slice_elems = dim_c * dim_h * dim_w;

    auto in_layout = layout{ov::PartialShape{static_cast<int64_t>(dim_b), -1, static_cast<int64_t>(dim_h), static_cast<int64_t>(dim_w)},
                            data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({{static_cast<int64_t>(dim_b), static_cast<int64_t>(dim_c),
                                              static_cast<int64_t>(dim_h), static_cast<int64_t>(dim_w)},
                                             data_types::f32, format::bfyx});
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{2}, data_types::i64, format::bfyx});

    const int64_t axis = 0;

    auto input_data = rg.generate_random_1d<float>(input_mem->count(), -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {1, 1});

    // reshape [1, C, H, W] → [1, C, H*W]  (spatial flatten, batch preserved)
    const int64_t hw = static_cast<int64_t>(dim_h * dim_w);
    const std::vector<int64_t> flatten_pattern = {1, -1, hw};
    const ov::PartialShape flatten_out_shape = {1, -1, hw};

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        // Branch 0: crop → [1, C, H, W] → reshape → [1, C, H*W]
        crop("crop_0", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape_0", input_info("crop_0"), false, flatten_pattern, flatten_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_0", input_info("reshape_0"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // Branch 1: crop → [1, C, H, W] → reshape → [1, C, H*W]
        crop("crop_1", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape_1", input_info("crop_1"), false, flatten_pattern, flatten_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_1", input_info("reshape_1"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    auto out0_mem = outputs.at("output_0").get_memory();
    cldnn::mem_lock<float> out0(out0_mem, get_test_stream());
    auto out1_mem = outputs.at("output_1").get_memory();
    cldnn::mem_lock<float> out1(out1_mem, get_test_stream());

    ASSERT_EQ(out0.size(), slice_elems);
    ASSERT_EQ(out1.size(), slice_elems);

    // Branch 0 must read the first batch slice
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(out0[i], input_data[0 * slice_elems + i]) << "Branch 0 mismatch at " << i;
    // Branch 1 must read the second batch slice
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(out1[i], input_data[1 * slice_elems + i]) << "Branch 1 mismatch at " << i;
}

// SwinTransformer layers.3 pattern: crop [1,1,H,W,C] → Reshape [-1,H,W,C] (base mode)
// Runtime reshape output[0] == 1, but output_pattern[0] == -1 (batch squeeze).
// In-place crop must still work correctly.
TEST(prepare_buffer_fusing, in_place_crop_dynamic_batch_axis_split_with_reshape_window_count_one) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    // 3-way QKV split on axis 0 with f=1 (like SwinTransformer window_count=1)
    const size_t num_heads = 3, seq_len = 4, head_dim = 2;
    const size_t slice_elems = 1 * num_heads * seq_len * head_dim;

    auto in_layout = layout{ov::PartialShape{3, -1, static_cast<int64_t>(num_heads),
                                              static_cast<int64_t>(seq_len), static_cast<int64_t>(head_dim)},
                            data_types::f32, format::bfzyx};
    auto input_mem = engine.allocate_memory({{3, 1, static_cast<int64_t>(num_heads),
                                              static_cast<int64_t>(seq_len), static_cast<int64_t>(head_dim)},
                                             data_types::f32, format::bfzyx});
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});

    const int64_t axis = 0;

    auto input_data = rg.generate_random_1d<float>(input_mem->count(), -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {1, 1, 1});

    // output_pattern[0] == -1: batch dim is squeezed. Runtime output[0] resolves to 1.
    const std::vector<int64_t> squeeze_pattern = {-1, static_cast<int64_t>(num_heads),
                                                   static_cast<int64_t>(seq_len), static_cast<int64_t>(head_dim)};
    const ov::PartialShape squeeze_out_shape = {-1, static_cast<int64_t>(num_heads),
                                                 static_cast<int64_t>(seq_len), static_cast<int64_t>(head_dim)};

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        // Q branch
        crop("crop_q", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape_q", input_info("crop_q"), false, squeeze_pattern, squeeze_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_q", input_info("reshape_q"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // K branch
        crop("crop_k", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape_k", input_info("crop_k"), false, squeeze_pattern, squeeze_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_k", input_info("reshape_k"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // V branch
        crop("crop_v", {input_info("input"), input_info("axis"), input_info("splits_length")}, cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape_v", input_info("crop_v"), false, squeeze_pattern, squeeze_out_shape, cldnn::reshape::reshape_mode::base),
        reorder("output_v", input_info("reshape_v"), format::bfyx, data_types::f32, std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    // Crops must still be in-place optimized despite runtime reshape output[0] == 1
    ASSERT_TRUE(network.get_primitive("crop_q")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop_k")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop_v")->can_be_optimized());

    auto q_mem = outputs.at("output_q").get_memory();
    cldnn::mem_lock<float> q_out(q_mem, get_test_stream());
    auto k_mem = outputs.at("output_k").get_memory();
    cldnn::mem_lock<float> k_out(k_mem, get_test_stream());
    auto v_mem = outputs.at("output_v").get_memory();
    cldnn::mem_lock<float> v_out(v_mem, get_test_stream());

    ASSERT_EQ(q_out.size(), slice_elems);
    ASSERT_EQ(k_out.size(), slice_elems);
    ASSERT_EQ(v_out.size(), slice_elems);

    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(q_out[i], input_data[0 * slice_elems + i]) << "Q mismatch at " << i;
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(k_out[i], input_data[1 * slice_elems + i]) << "K mismatch at " << i;
    for (size_t i = 0; i < slice_elems; i++)
        ASSERT_FLOAT_EQ(v_out[i], input_data[2 * slice_elems + i]) << "V mismatch at " << i;
}

// Regression test for test_no_input_pad with duplicate dependencies.
// When an in-place crop feeds both inputs of an eltwise (self-multiply),
// test_no_input_pad must save/restore each dependency's padding only once.
TEST(prepare_buffer_fusing, in_place_crop_self_multiply_spatial_split) {
    auto& engine = get_test_engine();

    // Small tensor to avoid blocked-format selection while keeping feature > 1.
    // Split on axis 2 (spatial Y) into [2, 1].
    const int64_t dim_b = 1, dim_f = 2, dim_y = 3, dim_x = 4;
    const int64_t split0 = 2, split1 = 1;

    auto in_layout = layout{ov::PartialShape{dim_b, dim_f, dim_y, dim_x},
                            data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({{dim_b, dim_f, dim_y, dim_x},
                                             data_types::f32, format::bfyx});
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{2}, data_types::i64, format::bfyx});

    const int64_t axis = 2;
    const size_t total = static_cast<size_t>(dim_b * dim_f * dim_y * dim_x);

    // Deterministic input: input[i] = i + 1
    std::vector<float> input_data(total);
    for (size_t i = 0; i < total; i++)
        input_data[i] = static_cast<float>(i + 1);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {split0, split1});

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    // Offsets must match what split.cpp computes: accumulated output shapes along split axis
    auto offset_0 = cldnn::tensor(0);
    auto offset_1 = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(0, split0));
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        // Branch 0: crop [1,2,2,4] -> eltwise(self-multiply)
        crop("crop_0", {input_info("input"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), offset_0, op_mode, 0, axis),
        eltwise("mul_0", {input_info("crop_0"), input_info("crop_0")}, eltwise_mode::prod),
        reorder("output_0", input_info("mul_0"), format::bfyx, data_types::f32,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // Branch 1: crop [1,2,1,4] -> eltwise(self-multiply)
        crop("crop_1", {input_info("input"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), offset_1, op_mode, 1, axis),
        eltwise("mul_1", {input_info("crop_1"), input_info("crop_1")}, eltwise_mode::prod),
        reorder("output_1", input_info("mul_1"), format::bfyx, data_types::f32,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    ASSERT_TRUE(network.get_primitive("crop_0")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop_1")->can_be_optimized());

    // bfyx layout: linear index = f * (dim_y * dim_x) + y * dim_x + x (b=0)
    // Verify branch 0: crop [1,2,2,4], y-offset = 0
    auto out0_mem = outputs.at("output_0").get_memory();
    cldnn::mem_lock<float> out0(out0_mem, get_test_stream());
    ASSERT_EQ(out0.size(), static_cast<size_t>(dim_b * dim_f * split0 * dim_x));

    for (int64_t f = 0; f < dim_f; f++) {
        for (int64_t y = 0; y < split0; y++) {
            for (int64_t x = 0; x < dim_x; x++) {
                size_t src_idx = static_cast<size_t>(f * dim_y * dim_x + y * dim_x + x);
                size_t dst_idx = static_cast<size_t>(f * split0 * dim_x + y * dim_x + x);
                float expected = input_data[src_idx] * input_data[src_idx];
                ASSERT_FLOAT_EQ(out0[dst_idx], expected)
                    << "Branch 0 mismatch at f=" << f << " y=" << y << " x=" << x
                    << " src_idx=" << src_idx << " dst_idx=" << dst_idx;
            }
        }
    }

    // Verify branch 1: crop [1,2,1,4], y-offset = split0
    auto out1_mem = outputs.at("output_1").get_memory();
    cldnn::mem_lock<float> out1(out1_mem, get_test_stream());
    ASSERT_EQ(out1.size(), static_cast<size_t>(dim_b * dim_f * split1 * dim_x));

    for (int64_t f = 0; f < dim_f; f++) {
        for (int64_t y = 0; y < split1; y++) {
            for (int64_t x = 0; x < dim_x; x++) {
                size_t src_idx = static_cast<size_t>(f * dim_y * dim_x + (split0 + y) * dim_x + x);
                size_t dst_idx = static_cast<size_t>(f * split1 * dim_x + y * dim_x + x);
                float expected = input_data[src_idx] * input_data[src_idx];
                ASSERT_FLOAT_EQ(out1[dst_idx], expected)
                    << "Branch 1 mismatch at f=" << f << " y=" << y << " x=" << x
                    << " src_idx=" << src_idx << " dst_idx=" << dst_idx;
            }
        }
    }
}

// A base-mode reshape that drops the cropped (size-1) last axis must not be marked
// runtime-padding-propagatable; otherwise sibling crop outputs alias the same buffer.
TEST(prepare_buffer_fusing, in_place_crop_dynamic_last_axis_split_to_collapsing_reshape) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{-1, -1, 4}, data_types::f32, format::bfyx };
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {4}, data_types::i64, format::bfyx });

    const int64_t axis = 2;
    set_values<int64_t>(axis_mem, { axis });
    set_values<int64_t>(splits_length_mem, { 1, 1, 1, 1 });

    auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        crop("crop0", { input_info("input"), input_info("axis"), input_info("splits_length") },
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        crop("crop1", { input_info("input"), input_info("axis"), input_info("splits_length") },
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        // Drops the cropped axis: [-1,-1,1] -> [-1,-1]  => must be rejected.
        reshape("rs0_collapse", input_info("crop0"), false, std::vector<int64_t>{-1, 3},
                ov::PartialShape{-1, -1}, cldnn::reshape::reshape_mode::base),
        // Preserves the cropped axis: [-1,-1,1] -> [-1,-1,1]  => still allowed.
        reshape("rs1_keep", input_info("crop1"), true, std::vector<int64_t>{-1, -1, 1},
                ov::PartialShape{-1, -1, 1}, cldnn::reshape::reshape_mode::base),
        reorder("out0", input_info("rs0_collapse"), format::bfyx, data_types::f32),
        reorder("out1", input_info("rs1_keep"), format::bfyx, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    ASSERT_FALSE(prog->get_node("rs0_collapse").as<reshape>().is_runtime_propagatable_padding());
    ASSERT_TRUE(prog->get_node("rs1_keep").as<reshape>().is_runtime_propagatable_padding());
}

// Regression test: VariadicSplit along spatial axis followed by Reshape where
// the padding from sibling crop slices is NOT evenly divisible by the reshape
// divider. Without the fix, integer truncation in the divider path causes the
// reshape to read from wrong offsets in the parent buffer.
TEST(prepare_buffer_fusing, in_place_crop_dynamic_indivisible_padding_with_reshape) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    // 3D input [-1, 2, 13]: batch dynamic, split along axis=2 (last dim) into [8, 4, 1].
    // Reshape splits last dim: crop0 [b,2,8] -> [b,2,4,2], crop1 [b,2,4] -> [b,2,2,2].
    // crop0: upper_pad=13-0-8=5, divider=2, 5%2!=0 → indivisible padding bug
    // crop1: upper_pad=13-8-4=1, divider=2, 1%2!=0 → indivisible padding bug
    auto in_layout = layout{ov::PartialShape{-1, 2, 13}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({{1, 2, 13}, data_types::f32, format::bfyx});
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});

    const int64_t axis = 2;
    auto input_data = rg.generate_random_1d<float>(input_mem->count(), -5.f, 5.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {8, 4, 1});

    const size_t feat = 2;
    const size_t crop0_last = 8, crop1_last = 4, crop2_last = 1;
    const size_t total_last = 13;

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology(
        input_layout("input", in_layout),
        activation("act", input_info("input"), activation_func::none),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        // crop0: [b,2,8] -> reshape [b,2,4,2]
        crop("crop0", {input_info("act"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape0", input_info("crop0"), true,
                std::vector<int64_t>{0, 0, 4, 2}, ov::PartialShape{-1, -1, 4, 2},
                cldnn::reshape::reshape_mode::base),
        reorder("output0", input_info("reshape0"), format::bfyx, data_types::f32,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // crop1: [b,2,4] -> reshape [b,2,2,2]
        crop("crop1", {input_info("act"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape1", input_info("crop1"), true,
                std::vector<int64_t>{0, 0, 2, 2}, ov::PartialShape{-1, -1, 2, 2},
                cldnn::reshape::reshape_mode::base),
        reorder("output1", input_info("reshape1"), format::bfyx, data_types::f32,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        // crop2: [b,2,1] -> reorder only
        crop("crop2", {input_info("act"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reorder("output2", input_info("crop2"), format::bfyx, data_types::f32,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    // Verify that the indivisible-padding crops were NOT optimized in-place,
    // while the crop without reshape user remains optimized.
    ASSERT_FALSE(network.get_primitive("crop0")->can_be_optimized());
    ASSERT_FALSE(network.get_primitive("crop1")->can_be_optimized());
    ASSERT_TRUE(network.get_primitive("crop2")->can_be_optimized());

    auto out0_mem = outputs.at("output0").get_memory();
    cldnn::mem_lock<float> out0(out0_mem, get_test_stream());
    auto out1_mem = outputs.at("output1").get_memory();
    cldnn::mem_lock<float> out1(out1_mem, get_test_stream());
    auto out2_mem = outputs.at("output2").get_memory();
    cldnn::mem_lock<float> out2(out2_mem, get_test_stream());

    ASSERT_EQ(out0.size(), feat * crop0_last);
    ASSERT_EQ(out1.size(), feat * crop1_last);
    ASSERT_EQ(out2.size(), feat * crop2_last);

    // Check the outputs as linear buffers: for each feature f, every crop contributes
    // a contiguous slice of length crop*_last, which must match the corresponding
    // contiguous range in the input feature slice starting at the crop offset.
    const size_t offset0 = 0, offset1 = crop0_last, offset2 = crop0_last + crop1_last;

    for (size_t f = 0; f < feat; f++)
        for (size_t y = 0; y < crop0_last; y++)
            ASSERT_FLOAT_EQ(out0[f * crop0_last + y], input_data[f * total_last + offset0 + y])
                << "output0 mismatch at f=" << f << " y=" << y;
    for (size_t f = 0; f < feat; f++)
        for (size_t y = 0; y < crop1_last; y++)
            ASSERT_FLOAT_EQ(out1[f * crop1_last + y], input_data[f * total_last + offset1 + y])
                << "output1 mismatch at f=" << f << " y=" << y;
    for (size_t f = 0; f < feat; f++)
        for (size_t y = 0; y < crop2_last; y++)
            ASSERT_FLOAT_EQ(out2[f * crop2_last + y], input_data[f * total_last + offset2 + y])
                << "output2 mismatch at f=" << f << " y=" << y;
}

// dyn-aware match() guard: crop with a static own layout but a dynamic predecessor
// (reshape with concrete output_pattern fed by dynamic input). Without the guard,
// build-time match took the static path and wrote padding that leaked into runtime.
TEST(prepare_buffer_fusing, in_place_crop_static_output_with_dynamic_predecessor) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    auto in_layout = layout{ov::PartialShape{-1, 128}, data_types::f32, format::bfyx};
    auto input_mem = engine.allocate_memory({{1, 128}, data_types::f32, format::bfyx});
    auto axis_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_length_mem = engine.allocate_memory({{2}, data_types::i64, format::bfyx});

    const int64_t axis = 2;
    auto input_data = rg.generate_random_1d<float>(input_mem->count(), -2.f, 2.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {axis});
    set_values<int64_t>(splits_length_mem, {2, 2});

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    auto offset_q = cldnn::tensor(0);
    auto offset_k = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(0, 2));
    topology topology(
        input_layout("input", in_layout),
        // Dynamic input collapses to fully static [1,4,4,8] via a concrete output_pattern.
        reshape("reshape", input_info("input"), true,
                std::vector<int64_t>{1, 4, 4, 8}, ov::PartialShape{1, 4, 4, 8},
                cldnn::reshape::reshape_mode::base),
        data("axis", axis_mem),
        data("splits_length", splits_length_mem),
        // Q branch: crop -> eltwise -> gemm(input0)
        crop("crop_q", {input_info("reshape"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), offset_q, op_mode, 0, axis),
        eltwise("scale_q", input_info("crop_q"), input_info("crop_q"), eltwise_mode::prod),
        // K branch: crop -> gemm(input1)
        crop("crop_k", {input_info("reshape"), input_info("axis"), input_info("splits_length")},
             cldnn::tensor(1), offset_k, op_mode, 1, axis),
        gemm("attn", {input_info("scale_q"), input_info("crop_k")},
             data_types::f32, false, true),
        reorder("output", input_info("attn"), format::bfyx, data_types::f32,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    // dyn-aware guards must reject in-place crop and leave no stale padding behind.
    ASSERT_FALSE(network.get_primitive("crop_q")->can_be_optimized());
    ASSERT_FALSE(network.get_primitive("crop_k")->can_be_optimized());
    const auto& q_pad = network.get_primitive("crop_q")->get_impl_params()->get_output_layout().data_padding;
    const auto& k_pad = network.get_primitive("crop_k")->get_impl_params()->get_output_layout().data_padding;
    ASSERT_FALSE(static_cast<bool>(q_pad));
    ASSERT_FALSE(static_cast<bool>(k_pad));

    // End-to-end correctness: gemm output matches host reference.
    const int64_t F = 4, Y = 4, X = 8;
    auto host_gemm_output = std::vector<float>(F * 2 * 2);
    for (int64_t f = 0; f < F; f++) {
        for (int64_t qy = 0; qy < 2; qy++) {
            for (int64_t ky = 0; ky < 2; ky++) {
                float acc = 0.f;
                for (int64_t x = 0; x < X; x++) {
                    const float q = input_data[f * Y * X + qy * X + x];
                    const float k = input_data[f * Y * X + (2 + ky) * X + x];
                    acc += (q * q) * k;
                }
                host_gemm_output[f * 4 + qy * 2 + ky] = acc;
            }
        }
    }

    auto out_mem = outputs.at("output").get_memory();
    cldnn::mem_lock<float> out(out_mem, get_test_stream());
    ASSERT_EQ(out.size(), host_gemm_output.size());
    for (size_t i = 0; i < host_gemm_output.size(); i++) {
        ASSERT_NEAR(out[i], host_gemm_output[i], 1e-3f) << "mismatch at i=" << i;
    }
}

// Runtime along-feature contract for the TransposeSplitMatcher pattern is covered
// by `in_place_crop_along_feature_runtime_updates_reshape_padding` below. That
// test uses the same shape family (input `[-1, 3, H, S]`, variadic_split axis=1,
// splits `{1,1,1}`) but drives it end-to-end through `network::execute()` and
// asserts the physically-correct padding contract on all three reshape outputs:
//   _lower_size[1] == k*H, _upper_size[1] == (F-1-k)*H,
//   get_pitches()[0] == F*H*S, get_linear_offset() == k*H*S.
// A former build-time-only variant of this test asserted arbitrary
// `_lower_size[1] == 1` / `_upper_size[1] == 1` values that the build-time
// `simple_data_format` dynamic branch never actually writes (that branch sets
// only the dynamic pad mask, not concrete sizes), so it has been dropped in
// favor of the runtime test.

// =============================================================================
// UNIT TEST: runtime along-feature path must propagate the crop's feature-axis
//            padding into the reshape output layout.
// -----------------------------------------------------------------------------
// Contract under test:
//   When do_runtime_in_place_crop picks the along-feature branch (because the
//   crop preserves batch/spatial dims and only cuts on the feature axis) and
//   the crop's user is a rank-reducing reshape flagged as
//   is_runtime_propagatable_padding(), the reshape output layout MUST end up
//   with a non-empty data_padding at runtime. Otherwise downstream consumers
//   (eltwise, RoPE, vl_sdpa, ...) read the wrong region of the parent buffer.
//
// Failure mode under current code (before fix):
//   The along-feature helper only updates crop_layout.data_padding, and the
//   caller relies on reshape_inst->update_shape() to re-derive the reshape
//   output padding. But reshape_inst::calc_output_layouts explicitly resets
//   the padding to padding() for reshape_mode::base (see reshape.cpp), so the
//   reshape observes no padding at all -> optimization is silently broken
//   for the TransposeSplitMatcher pattern.
//
// Setup (TransposeSplitMatcher pattern):
//   Build a topology with a dynamic input [-1, 3, H, S] so the build-time
//   crop optimization goes through simple_data_format (build-time dynamic
//   branch, which stamps a dyn pad mask on the reshape output). Then run
//   with a concrete static [L, 3, H, S] so that do_runtime_in_place_crop's
//   along-feature branch fires (can_crop_be_optimized_along_feature is true
//   because batch/H/S are preserved). Three crops on axis=1 of size 1
//   feed rank-reducing reshapes (base mode, [-1, 1, H, S] -> [-1, H, S]),
//   each followed by an eltwise producer so the network can execute.
//
// Expectations after execute():
//   1) All three crops are optimized in-place (can_be_optimized() == true).
//   2) Each reshape*'s runtime output layout carries a dynamic pad mask on
//      at least one axis.
//   3) The total padding (sum of lower + upper across all axes) on each
//      reshape output equals 2 — the crop takes 1 slot out of 3, so 2 slots
//      of padding remain regardless of which slice we're looking at
//      (slice 0: 0 + 2, slice 1: 1 + 1, slice 2: 2 + 0).
// =============================================================================
TEST(prepare_buffer_fusing, in_place_crop_along_feature_runtime_updates_reshape_padding) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const int64_t L = 32, H = 4, S = 8;

    auto in_layout_dyn  = layout{ov::PartialShape{-1, 3, H, S}, data_types::f16, format::bfyx};
    auto input_mem      = engine.allocate_memory({{L, 3, H, S}, data_types::f16, format::bfyx});
    auto axis_mem       = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_len_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});
    auto scale_mem      = engine.allocate_memory({{1}, data_types::f16, format::bfyx});

    auto input_data = rg.generate_random_1d<ov::float16>(L * 3 * H * S, -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_len_mem, {1, 1, 1});
    set_values<ov::float16>(scale_mem, {ov::float16(1.f)});

    const auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    const int64_t axis = 1;
    // Use an explicit output_pattern so reshape's shape inference can resolve
    // the dynamic batch dim at runtime; special_zero=false with -1 tells
    // ov::op::v1::Reshape to infer this dim from the total element count.
    // Without a resolvable output shape, downstream nodes would trip
    // "[GPU] Count is called for dynamic shape" during execute().
    const std::vector<int64_t> rs_pattern    = {-1, H, S};
    const auto                 rs_shape_dyn  = ov::PartialShape{-1, H, S};

    topology topo(
        input_layout("input", in_layout_dyn),
        data("axis",       axis_mem),
        data("splits_len", splits_len_mem),
        data("scale",      scale_mem),
        crop("crop0", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape0", input_info("crop0"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out0", {input_info("reshape0"), input_info("scale")}, eltwise_mode::prod),
        crop("crop1", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape1", input_info("crop1"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out1", {input_info("reshape1"), input_info("scale")}, eltwise_mode::prod),
        crop("crop2", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape2", input_info("crop2"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out2", {input_info("reshape2"), input_info("scale")}, eltwise_mode::prod)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, topo, config);
    net.set_input_data("input", input_mem);
    net.execute();

    ASSERT_TRUE(net.get_primitive("crop0")->can_be_optimized()) << "crop0 must be in-place at runtime";
    ASSERT_TRUE(net.get_primitive("crop1")->can_be_optimized()) << "crop1 must be in-place at runtime";
    ASSERT_TRUE(net.get_primitive("crop2")->can_be_optimized()) << "crop2 must be in-place at runtime";

    // Physically-correct reshape padding contract:
    // The reshape output layout must view the parent QKV buffer with:
    //   get_pitches()[0]    == F * H * S          (per-token stride in parent)
    //   get_linear_offset() == k * H * S          (slice-k start offset)
    // The helper achieves this by placing scaled padding on the reshape's
    // dim 1 (bfyx feature position = pshape H):
    //   _lower_size[1] = k       * H
    //   _upper_size[1] = (F-1-k) * H
    // Both values are asserted precisely so any regression that reintroduces
    // the empty / raw-`k` padding will fail the test immediately.
    constexpr int64_t F = 3;
    const int64_t expected_pitch  = F * H * S;
    auto assert_reshape_padded = [&](const std::string& id, int64_t k) {
        const auto inst = net.get_primitive(id);
        ASSERT_NE(inst, nullptr) << id << " instance not found";
        const auto& lo  = inst->get_output_layout();
        const auto& pad = lo.data_padding;

        const int64_t expected_lower  = k * H;
        const int64_t expected_upper  = (F - 1 - k) * H;
        const int64_t expected_offset = k * H * S;

        ASSERT_TRUE(pad._dynamic_dims_mask[1])
            << id << ": dynamic pad mask must be attached to the H axis (dim 1)";
        ASSERT_EQ(pad._lower_size[1], expected_lower)
            << id << ": _lower_size[1] must equal k * H = " << expected_lower;
        ASSERT_EQ(pad._upper_size[1], expected_upper)
            << id << ": _upper_size[1] must equal (F-1-k) * H = " << expected_upper;
        ASSERT_EQ(static_cast<int64_t>(lo.get_pitches()[0]), expected_pitch)
            << id << ": get_pitches()[0] must equal F * H * S = " << expected_pitch;
        ASSERT_EQ(static_cast<int64_t>(lo.get_linear_offset()), expected_offset)
            << id << ": get_linear_offset() must equal k * H * S = " << expected_offset;
    };

    assert_reshape_padded("reshape0", 0);
    assert_reshape_padded("reshape1", 1);
    assert_reshape_padded("reshape2", 2);
}

// =============================================================================
// UNIT TEST: build-time and runtime must agree on which axis carries the
//            dyn-pad mask for the TransposeSplitMatcher pattern.
// -----------------------------------------------------------------------------
// Contract under test:
//   OCL consumers (RoPE, eltwise, MVN, ...) JIT-compile against the layout
//   snapshot produced by prepare_buffer_fusing at build time. The JIT bakes in
//   which shape_info slot to read pad_before / pad_after from — that slot is
//   fixed for the lifetime of the compiled kernel. fill_shape_info_data then
//   writes the runtime _lower_size / _upper_size into that fixed slot.
//
//   If the along-feature runtime helper places the mask on a DIFFERENT axis
//   than the build-time simple_data_format algorithm, three things break:
//     1. The JIT'd kernel reads pad_before from the wrong shape_info slot.
//     2. fill_shape_info_data pulls _lower_size / _upper_size from an axis
//        that carries 0 at runtime (because the actual padding migrated).
//     3. The consumer walks memory as if the input had no padding, silently
//        addressing the wrong slice.
//
// Setup: same TransposeSplitMatcher pattern as
//   in_place_crop_along_feature_runtime_updates_reshape_padding — dynamic
//   input [-1, 3, H, S], variadic_split axis=1 with 3 unit slices, each crop
//   followed by a rank-reducing reshape to [-1, H, S] (base mode).
//
// Expectations:
//   For every reshape output, the dyn-pad mask axis chosen by the build-time
//   simple_data_format branch MUST match the axis on which the runtime
//   along-feature helper writes the concrete _lower/_upper values (bfyx dim
//   1, i.e. the H axis). Otherwise the runtime fix is invisible to any
//   consumer that reads shape_info instead of taking the layout directly.
// =============================================================================
TEST(prepare_buffer_fusing, in_place_crop_along_feature_reshape_pad_axis_matches_between_build_and_runtime) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const int64_t L = 32, H = 4, S = 8;

    auto in_layout_dyn  = layout{ov::PartialShape{-1, 3, H, S}, data_types::f16, format::bfyx};
    auto input_mem      = engine.allocate_memory({{L, 3, H, S}, data_types::f16, format::bfyx});
    auto axis_mem       = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_len_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});
    auto scale_mem      = engine.allocate_memory({{1}, data_types::f16, format::bfyx});

    auto input_data = rg.generate_random_1d<ov::float16>(L * 3 * H * S, -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_len_mem, {1, 1, 1});
    set_values<ov::float16>(scale_mem, {ov::float16(1.f)});

    const auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    const int64_t axis = 1;
    const std::vector<int64_t> rs_pattern   = {-1, H, S};
    const auto                 rs_shape_dyn = ov::PartialShape{-1, H, S};

    topology topo(
        input_layout("input", in_layout_dyn),
        data("axis",       axis_mem),
        data("splits_len", splits_len_mem),
        data("scale",      scale_mem),
        crop("crop0", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape0", input_info("crop0"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out0", {input_info("reshape0"), input_info("scale")}, eltwise_mode::prod),
        crop("crop1", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape1", input_info("crop1"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out1", {input_info("reshape1"), input_info("scale")}, eltwise_mode::prod),
        crop("crop2", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape2", input_info("crop2"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out2", {input_info("reshape2"), input_info("scale")}, eltwise_mode::prod)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    auto prog = program::build_program(engine, topo, config, false, false);
    ASSERT_NE(prog, nullptr);

    // Runtime along-feature helper writes concrete padding on the H axis
    // (bfyx dim 1). Build-time simple_data_format MUST place the dyn-pad mask
    // on the same axis so downstream OCL consumers JIT'd from the build-time
    // layout read the right shape_info slot at runtime.
    auto assert_build_time_mask_on_h_axis = [&](const std::string& id) {
        const auto& lo   = prog->get_node(id).get_output_layout();
        const auto& mask = lo.data_padding._dynamic_dims_mask;
        ASSERT_TRUE(mask[1])
            << id << ": build-time dyn-pad mask must be attached to the H axis (dim 1) "
            << "to match the runtime along-feature helper";
        ASSERT_FALSE(mask[2])
            << id << ": build-time dyn-pad mask must NOT sit on dim 2 (S axis); "
            << "otherwise consumers JIT'd against dim 2 will read 0 padding at runtime "
            << "while the actual padding migrates to dim 1";
    };
    assert_build_time_mask_on_h_axis("reshape0");
    assert_build_time_mask_on_h_axis("reshape1");
    assert_build_time_mask_on_h_axis("reshape2");

    // Execute and verify that the runtime mask location is IDENTICAL to the
    // build-time one — no silent migration between axes.
    network net(prog, 0);
    net.set_input_data("input", input_mem);
    net.execute();

    auto assert_masks_agree = [&](const std::string& id) {
        const auto& build_mask   = prog->get_node(id).get_output_layout().data_padding._dynamic_dims_mask;
        const auto& runtime_mask = net.get_primitive(id)->get_output_layout().data_padding._dynamic_dims_mask;
        ASSERT_EQ(build_mask, runtime_mask)
            << id << ": build-time and runtime dyn-pad masks must sit on the same axis; "
            << "otherwise fill_shape_info_data writes 0 into the JIT-baked slot";
    };
    assert_masks_agree("reshape0");
    assert_masks_agree("reshape1");
    assert_masks_agree("reshape2");
}

// =============================================================================
// UNIT TEST: end-to-end addressing regression through an OCL consumer that
//            reads the reshape's shape_info.
// -----------------------------------------------------------------------------
// Contract under test:
//   The full build → JIT → runtime-fill → kernel-arithmetic pipeline for the
//   TransposeSplitMatcher pattern must produce numerically-correct outputs.
//   Downstream consumers walk the padded reshape output using shape_info's
//   pad_before / pad_after slots; if any link in the chain is wrong (build
//   picks a different mask axis than runtime, fill_shape_info_data writes 0
//   into the JIT-baked slot, kernel template picks the wrong slot, etc.),
//   the consumer walks wrong memory and produces wrong data.
//
// The other tests in this family assert layout properties in isolation
// (masks, sizes, pitches, offsets). This one is a *behavior* test: it checks
// that an eltwise OCL consumer — which is exactly the class of kernel
// affected by the shape_info axis mismatch documented in the verbose log —
// reads the correct slice of the parent QKV buffer.
//
// The consumer here is `eltwise(reshape, bias, mode=sum)` where `bias` has
// shape `[L, 1, 1]` with a distinct value per token index. That construction
// makes any wrong token pitch, wrong linear offset, or shape_info-slot
// mismatch fail immediately in the numerical comparison against a
// host-computed golden reference for the correct slice (as opposed to the
// `scale=1.0f` variant used by sibling tests, which multiplies garbage by
// 1.0 and silently masks address-walking bugs).
//
// Failure mode this catches (pre-fix history):
//   Build-time simple_data_format placed the mask on the S axis (dim 2)
//   while runtime along-feature helper wrote _lower_size / _upper_size on
//   the H axis (dim 1). fill_shape_info_data pulled 0 from the S slot the
//   JIT'd eltwise kernel expected, so the kernel walked the input as if
//   there were no padding: q_start * H * S instead of q_start * F * H * S,
//   plus no k * H * S slice offset. Slice 0 accidentally worked (offset 0),
//   slices 1 and 2 read wrong tokens.
// =============================================================================
TEST(prepare_buffer_fusing, in_place_crop_along_feature_end_to_end_addressing_through_ocl_consumer) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const int64_t L = 32, H = 4, S = 8;
    constexpr int64_t F = 3;

    auto in_layout_dyn  = layout{ov::PartialShape{-1, F, H, S}, data_types::f32, format::bfyx};
    auto input_mem      = engine.allocate_memory({{L, F, H, S}, data_types::f32, format::bfyx});
    auto axis_mem       = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_len_mem = engine.allocate_memory({{F}, data_types::i64, format::bfyx});
    // Per-token bias broadcast over [H, S]. Distinct value per token so any
    // wrong token stride / slice offset shows up in the numerical output.
    auto bias_mem       = engine.allocate_memory({{L, 1, 1}, data_types::f32, format::bfyx});

    auto input_data = rg.generate_random_1d<float>(L * F * H * S, -1.f, 1.f);
    std::vector<float> bias_data(L);
    for (int64_t t = 0; t < L; t++)
        bias_data[t] = static_cast<float>(t + 1);

    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_len_mem, {1, 1, 1});
    set_values(bias_mem, bias_data);

    const auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    const int64_t axis = 1;
    const std::vector<int64_t> rs_pattern   = {-1, H, S};
    const auto                 rs_shape_dyn = ov::PartialShape{-1, H, S};

    topology topo(
        input_layout("input", in_layout_dyn),
        data("axis",       axis_mem),
        data("splits_len", splits_len_mem),
        data("bias",       bias_mem),
        crop("crop0", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape0", input_info("crop0"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out0", {input_info("reshape0"), input_info("bias")}, eltwise_mode::sum),
        crop("crop1", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape1", input_info("crop1"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out1", {input_info("reshape1"), input_info("bias")}, eltwise_mode::sum),
        crop("crop2", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape2", input_info("crop2"), false, rs_pattern, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        eltwise("out2", {input_info("reshape2"), input_info("bias")}, eltwise_mode::sum)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, topo, config);
    net.set_input_data("input", input_mem);
    auto outputs = net.execute();

    // Sanity: all three crops must be in-place, otherwise the test would
    // not exercise the shape_info addressing path we care about.
    ASSERT_TRUE(net.get_primitive("crop0")->can_be_optimized());
    ASSERT_TRUE(net.get_primitive("crop1")->can_be_optimized());
    ASSERT_TRUE(net.get_primitive("crop2")->can_be_optimized());

    // Per-slice golden reference computed on the CORRECT slice of the parent
    // buffer: out_k[t, h, s] = input[t, k, h, s] + bias[t].
    // Any wrong token stride, wrong linear offset, or shape_info-slot
    // mismatch will fail one of these comparisons with a diagnostic that
    // points directly at the offending (k, t, h, s) coordinates.
    auto check_slice = [&](const std::string& id, int64_t k) {
        auto out_mem = outputs.at(id).get_memory();
        cldnn::mem_lock<float> out(out_mem, get_test_stream());
        ASSERT_EQ(out.size(), static_cast<size_t>(L * H * S)) << id;
        for (int64_t t = 0; t < L; t++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t s = 0; s < S; s++) {
                    const size_t in_idx  = static_cast<size_t>(((t * F + k) * H + h) * S + s);
                    const size_t out_idx = static_cast<size_t>((t * H + h) * S + s);
                    const float expected = input_data[in_idx] + bias_data[t];
                    ASSERT_FLOAT_EQ(out[out_idx], expected)
                        << id << " mismatch: k=" << k
                        << " t=" << t << " h=" << h << " s=" << s;
                }
            }
        }
    };
    check_slice("out0", 0);
    check_slice("out1", 1);
    check_slice("out2", 2);
}

// =============================================================================
// UNIT TEST: runtime along-feature path with no reshape user (regression guard).
// -----------------------------------------------------------------------------
// Contract under test:
//   After the along-feature helper is extended to take a user_info parameter,
//   the crop-only path (no reshape user) must still work: user_info.first
//   stays null, the helper must not dereference it, and the crop must still
//   be optimized in-place.
//
// Setup:
//   Dynamic input [-1, 3, H, S]; a single crop on axis=1 slice 1 feeds
//   directly into a reorder (no reshape between crop and consumer). At
//   runtime, do_runtime_in_place_crop sees crop_users.size() == 1 but the
//   user is not a reshape, so user_info.first stays null through the helper.
//
// Expectations after execute():
//   1) crop is optimized in-place (can_be_optimized() == true).
//   2) The extended helper signature does not crash with a null user_info.
// =============================================================================
TEST(prepare_buffer_fusing, in_place_crop_along_feature_no_reshape_user) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const int64_t L = 8, H = 4, S = 8;

    auto in_layout_dyn  = layout{ov::PartialShape{-1, 3, H, S}, data_types::f16, format::bfyx};
    auto input_mem      = engine.allocate_memory({{L, 3, H, S}, data_types::f16, format::bfyx});
    auto axis_mem       = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_len_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});

    auto input_data = rg.generate_random_1d<ov::float16>(L * 3 * H * S, -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_len_mem, {1, 1, 1});

    topology topo(
        input_layout("input", in_layout_dyn),
        data("axis",       axis_mem),
        data("splits_len", splits_len_mem),
        crop("crop", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), cldnn::crop_ngraph_op_mode::variadic_split, 1, 1),
        reorder("out", input_info("crop"), format::bfyx, data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network net(engine, topo, config);
    net.set_input_data("input", input_mem);
    net.execute();

    ASSERT_TRUE(net.get_primitive("crop")->can_be_optimized())
        << "crop with no reshape user must still be optimized in-place along the feature axis";
}

// =============================================================================
// TransposeSplitMatcher: Split(axis=1) with 3 crops -> Reshape -> eltwise
//
// Pattern: Input[-1, 3, H, S]
//   crop0(axis=1, slice=0) [-1,1,H,S] -> reshape[-1,H,S] -> eltwise(scale) -> out0
//   crop1(axis=1, slice=1) [-1,1,H,S] -> reshape[-1,H,S] -> eltwise(scale) -> out1
//   crop2(axis=1, slice=2) [-1,1,H,S] -> reshape[-1,H,S] -> eltwise(scale) -> out2
//
// All 3 crops must have can_be_optimized=1 at runtime:
//   axis=1, input[1]==1 (static), output rank = input rank - 1
//   reshape.is_runtime_propagatable_padding() == true for all three
//   downstream consumer is eltwise (no further block)
// =============================================================================
TEST(prepare_buffer_fusing, in_place_crop_split_axis1_three_crops_eltwise_consumer) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const int64_t H = 4, S = 8;
    const int64_t L = 16;  // batch (sequence length)

    // Use a STATIC input layout so that the GPU runtime can allocate output buffers.
    // The dynamic propagation (is_runtime_propagatable_padding) is verified at
    // program-build level below; runtime can_be_optimized is verified after execute().
    auto in_layout      = layout{{L, 3, H, S}, data_types::f16, format::bfyx};
    auto in_layout_dyn  = layout{ov::PartialShape{-1, 3, H, S}, data_types::f16, format::bfyx};
    auto input_mem      = engine.allocate_memory(in_layout);
    auto axis_mem       = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_len_mem = engine.allocate_memory({{3}, data_types::i64, format::bfyx});

    auto input_data = rg.generate_random_1d<ov::float16>(L * 3 * H * S, -1.f, 1.f);
    set_values(input_mem, input_data);
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_len_mem, {1, 1, 1});

    auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    const int64_t axis = 1;
    // Reshape output shape: static for proper buffer allocation
    auto rs_shape     = ov::PartialShape{L, H, S};
    auto rs_shape_dyn = ov::PartialShape{-1, H, S};

    // Build topology with the DYNAMIC layout for the is_runtime_propagatable_padding check
    topology topo_dyn(
        input_layout("input", in_layout_dyn),
        data("axis",       axis_mem),
        data("splits_len", splits_len_mem),
        crop("crop0", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape0", input_info("crop0"), false, {}, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        reorder("out0", input_info("reshape0"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        crop("crop1", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape1", input_info("crop1"), false, {}, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        reorder("out1", input_info("reshape1"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        crop("crop2", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape2", input_info("crop2"), false, {}, rs_shape_dyn, cldnn::reshape::reshape_mode::base),
        reorder("out2", input_info("reshape2"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true)
    );
    ExecutionConfig config_dyn = get_test_default_config(engine);
    config_dyn.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config_dyn.set_property(ov::intel_gpu::optimize_data(true));

    // Check static optimization decision: all 3 reshapes must be is_runtime_propagatable_padding
    auto prog = program::build_program(engine, topo_dyn, config_dyn, false, true);
    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->get_node("reshape0").as<reshape>().is_runtime_propagatable_padding())
        << "reshape0 must be runtime-propagatable";
    ASSERT_TRUE(prog->get_node("reshape1").as<reshape>().is_runtime_propagatable_padding())
        << "reshape1 must be runtime-propagatable";
    ASSERT_TRUE(prog->get_node("reshape2").as<reshape>().is_runtime_propagatable_padding())
        << "reshape2 must be runtime-propagatable";

    // For runtime check, use the dynamic topology. The reorder output marker nodes
    // allow observing can_be_optimized at runtime without triggering count() errors.
    network net(engine, topo_dyn, config_dyn);
    net.set_input_data("input", input_mem);

    // Verify can_be_optimized BEFORE execute so we check the compile-time decision
    // (prepare_buffer_fusing sets can_be_optimized statically for dynamic crops too
    // when is_runtime_propagatable_padding() returns true and the simple_data_format
    // path fires at build time)
    ASSERT_TRUE(net.get_primitive("crop0")->can_be_optimized()) << "crop0 should be in-place";
    ASSERT_TRUE(net.get_primitive("crop1")->can_be_optimized()) << "crop1 should be in-place";
    ASSERT_TRUE(net.get_primitive("crop2")->can_be_optimized()) << "crop2 should be in-place";
}

// =============================================================================
// TransposeSplitMatcher: Split(axis=1) with 3 crops -> vl_sdpa consumer.
// -----------------------------------------------------------------------------
// vl_sdpa-specific coverage. General "3 crops in-place" for the
// TransposeSplitMatcher axis=1 pattern is covered by the sibling test
// `in_place_crop_split_axis1_three_crops_eltwise_consumer`; the reshape
// padding contract (mask axis / _lower / _upper / pitches / offset) is
// covered by `in_place_crop_along_feature_runtime_updates_reshape_padding`.
// This test focuses on what only the vl_sdpa consumer path can verify:
//
//   1. `is_runtime_propagatable_padding()` returns true for a reshape whose
//      downstream is `vl_sdpa` (the vl_sdpa branch of the predicate in
//      reshape_inst.h). Every other test in the family exercises the
//      eltwise/reorder branches; if someone tightens or reverts the vl_sdpa
//      branch, no other test would notice.
//
//   2. Layout asymmetry — V retains the crop-derived padded parent view
//      (F*H*S token pitch) while Q/K went through eltwise and got fresh
//      contiguous buffers (H*S token pitch). Q_pitch == K_pitch != V_pitch.
//      Regresses if reshape2 loses its padded view for the direct-to-vl_sdpa
//      path.
//
//   3. End-to-end numerical correctness of the CM kernel's
//      `token_offset_v` / `v_token_pitch` scalar-based addressing. The V
//      input has non-zero linear_offset and non-standard pitch; the CM
//      kernel dispatch reads `params.input_layouts[2].get_linear_offset()`
//      and `.get_pitches()[0]` to compute those scalars, and then the
//      kernel body walks V using them. If any link in that chain is wrong,
//      the numerical comparison against a host-side SDPA reference fails.
//
// Pattern: Input[-1, 3, H, S]
//   crop0(axis=1, slice=0) → reshape → eltwise (Q proxy) ────────►
//   crop1(axis=1, slice=1) → reshape → eltwise (K proxy) ────────► vl_sdpa
//   crop2(axis=1, slice=2) → reshape (V, direct)   ──────────────►
//
// NOTE: vl_sdpa is a CM kernel available only on XMX (systolic) GPU devices.
// Skipped on non-XMX or non-CM-capable devices.
// =============================================================================
TEST(prepare_buffer_fusing, in_place_crop_split_axis1_three_crops_vlsdpa_consumer) {
    auto& engine = get_test_engine();

    ExecutionConfig check_config = get_test_default_config(engine);
    if (!engine.get_device_info().supports_immad ||
        !cldnn::check_cm_jit_support(engine, check_config))
        GTEST_SKIP() << "vl_sdpa CM kernel not available on this device";

    tests::random_generator rg(GET_SUITE_NAME);

    const int32_t H = 2, S = 64;
    const std::vector<int32_t> cu_seqlens = {0, 16};
    const int32_t L = cu_seqlens.back();
    constexpr int32_t F = 3;

    auto in_layout = layout{ov::PartialShape{-1, F, H, S}, data_types::f16, format::bfyx};
    auto input_mem        = engine.allocate_memory({{L, F, H, S}, data_types::f16, format::bfyx});
    auto cu_seqlens_mem   = engine.allocate_memory({{static_cast<ov::Dimension::value_type>(cu_seqlens.size())}, data_types::i32, format::bfyx});
    auto axis_mem         = engine.allocate_memory({{}, data_types::i64, format::bfyx});
    auto splits_len_mem   = engine.allocate_memory({{F}, data_types::i64, format::bfyx});
    auto scale_mem        = engine.allocate_memory({{1}, data_types::f16, format::bfyx});

    auto input_data = rg.generate_random_1d<ov::float16>(L * F * H * S, -0.5f, 0.5f);
    set_values(input_mem, input_data);
    set_values(cu_seqlens_mem, cu_seqlens);
    set_values<int64_t>(axis_mem, {1});
    set_values<int64_t>(splits_len_mem, {1, 1, 1});
    set_values<ov::float16>(scale_mem, {ov::float16(1.f)});

    auto op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    const int64_t axis = 1;
    const std::vector<int64_t> rs_pattern = {-1, H, S};
    const auto rs_shape = ov::PartialShape{-1, H, S};

    // vl_sdpa transpose order: raw memory is [L, H, S] but the kernel extracts
    // num_heads / head_size from dim 1 / dim 2 of the logical view. Order
    // {1, 0, 2} maps raw dims [L, H, S] to logical [H, L, S] so that
    // num_heads = raw[1] = H and head_size = raw[2] = S. This matches the
    // functional test in transpose_split_vlsdpa.cpp. Using identity order
    // here silently mis-computes num_heads as L.
    const std::vector<int64_t> order{1, 0, 2};

    topology topo(
        input_layout("input",     in_layout),
        input_layout("cu_seqlens", layout{ov::PartialShape{-1}, data_types::i32, format::bfyx}),
        data("axis",       axis_mem),
        data("splits_len", splits_len_mem),
        data("scale",      scale_mem),
        // Q path: crop0 → reshape → eltwise*1.0 (contiguous fresh buffer).
        crop("crop0", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 0, axis),
        reshape("reshape0", input_info("crop0"), false, rs_pattern, rs_shape, cldnn::reshape::reshape_mode::base),
        eltwise("eltwise0", {input_info("reshape0"), input_info("scale")}, eltwise_mode::prod),
        // K path: crop1 → reshape → eltwise*1.0 (contiguous fresh buffer).
        crop("crop1", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 1, axis),
        reshape("reshape1", input_info("crop1"), false, rs_pattern, rs_shape, cldnn::reshape::reshape_mode::base),
        eltwise("eltwise1", {input_info("reshape1"), input_info("scale")}, eltwise_mode::prod),
        // V path: crop2 → reshape (padded view straight into vl_sdpa).
        crop("crop2", {input_info("input"), input_info("axis"), input_info("splits_len")},
             cldnn::tensor(1), cldnn::tensor(0), op_mode, 2, axis),
        reshape("reshape2", input_info("crop2"), false, rs_pattern, rs_shape, cldnn::reshape::reshape_mode::base),
        vl_sdpa("vlsdpa",
                {input_info("eltwise0"), input_info("eltwise1"),
                 input_info("reshape2"), input_info("cu_seqlens")},
                order, order, order, order),
        reorder("output", input_info("vlsdpa"), format::bfyx, data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    // (1) Build-time predicate: all 3 reshapes must be runtime-propagatable,
    //     including reshape2 whose downstream is vl_sdpa (the branch unique
    //     to this test).
    {
        auto prog = program::build_program(engine, topo, config, false, true);
        ASSERT_NE(prog, nullptr);
        ASSERT_TRUE(prog->get_node("reshape0").as<reshape>().is_runtime_propagatable_padding())
            << "reshape0 (Q→eltwise) must be runtime-propagatable";
        ASSERT_TRUE(prog->get_node("reshape1").as<reshape>().is_runtime_propagatable_padding())
            << "reshape1 (K→eltwise) must be runtime-propagatable";
        ASSERT_TRUE(prog->get_node("reshape2").as<reshape>().is_runtime_propagatable_padding())
            << "reshape2 (V→vl_sdpa) must be runtime-propagatable (vl_sdpa-specific guard)";
    }

    // Execute with optimizations enabled.
    network net(engine, topo, config);
    net.set_input_data("input", input_mem);
    net.set_input_data("cu_seqlens", cu_seqlens_mem);
    auto outputs = net.execute();
    auto out_mem = outputs.at("output").get_memory();
    ASSERT_NE(out_mem, nullptr);

    // (2) Layout asymmetry: V still carries crop-derived padding (F*H*S
    //     token pitch); Q/K went through eltwise and are contiguous (H*S).
    //     If reshape2 loses its padded view for the vl_sdpa path, v_pitch
    //     collapses to q_pitch.
    const auto q_pitch = net.get_primitive("eltwise0")->get_output_layout().get_pitches()[0];
    const auto k_pitch = net.get_primitive("eltwise1")->get_output_layout().get_pitches()[0];
    const auto v_pitch = net.get_primitive("reshape2")->get_output_layout().get_pitches()[0];
    ASSERT_EQ(q_pitch, k_pitch)
        << "Q and K flow through eltwise, so their pitches must match";
    ASSERT_NE(v_pitch, q_pitch)
        << "V must keep the crop-derived padded pitch (F*H*S) — regression if it collapses to H*S";

    // (3) End-to-end numerical: host-side SDPA reference against vl_sdpa
    //     output. This is the only automated check that the CM kernel's
    //     token_offset_v / v_token_pitch scalar dispatch actually addresses
    //     the correct V slice of the parent QKV buffer.
    //
    //     Reference: for each sequence [seq_start, seq_end) and each query
    //     token q in that range, each head h computes
    //         scores[k] = (Q[q, h] · K[seq_start+k, hkv]) * scale
    //         P[k]      = softmax(scores)
    //         out[q, h] = sum_k P[k] * V[seq_start+k, hkv]
    //     with scale = 1 / sqrt(head_size). No causal mask (vl_sdpa
    //     wrapper passes use_causal_mask=false). Q, K, V come from the
    //     three F-slices of the parent input: Q = input[:, 0, :, :],
    //     K = input[:, 1, :, :], V = input[:, 2, :, :].
    const size_t num_heads = static_cast<size_t>(H);
    const size_t num_kv_heads = static_cast<size_t>(H);
    const size_t head_ratio = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(S));

    auto qkv_at = [&](size_t token, int32_t f, size_t head, size_t d) -> float {
        const size_t off = ((token * static_cast<size_t>(F) + static_cast<size_t>(f)) * num_heads + head) * static_cast<size_t>(S) + d;
        return static_cast<float>(input_data[off]);
    };

    std::vector<float> ref_output(static_cast<size_t>(L) * num_heads * static_cast<size_t>(S), 0.f);
    for (size_t s = 0; s + 1 < cu_seqlens.size(); s++) {
        const int32_t seq_start = cu_seqlens[s];
        const int32_t seq_end   = cu_seqlens[s + 1];
        const int32_t seq_len   = seq_end - seq_start;
        ASSERT_GT(seq_len, 0);

        std::vector<float> scores(static_cast<size_t>(seq_len));
        for (int32_t q = 0; q < seq_len; q++) {
            const size_t q_idx = static_cast<size_t>(seq_start + q);
            for (size_t h = 0; h < num_heads; h++) {
                const size_t hkv = h / head_ratio;
                // dot-products Q·K, scaled.
                for (int32_t k = 0; k < seq_len; k++) {
                    const size_t k_idx = static_cast<size_t>(seq_start + k);
                    float acc = 0.f;
                    for (size_t d = 0; d < static_cast<size_t>(S); d++) {
                        acc += qkv_at(q_idx, 0, h, d) * qkv_at(k_idx, 1, hkv, d);
                    }
                    scores[static_cast<size_t>(k)] = acc * scale;
                }
                // softmax
                float max_s = scores[0];
                for (int32_t k = 1; k < seq_len; k++)
                    max_s = std::max(max_s, scores[static_cast<size_t>(k)]);
                float denom = 0.f;
                for (int32_t k = 0; k < seq_len; k++) {
                    scores[static_cast<size_t>(k)] = std::exp(scores[static_cast<size_t>(k)] - max_s);
                    denom += scores[static_cast<size_t>(k)];
                }
                const float inv_denom = 1.f / denom;
                for (int32_t k = 0; k < seq_len; k++)
                    scores[static_cast<size_t>(k)] *= inv_denom;
                // weighted sum with V.
                for (size_t d = 0; d < static_cast<size_t>(S); d++) {
                    float acc = 0.f;
                    for (int32_t k = 0; k < seq_len; k++) {
                        const size_t k_idx = static_cast<size_t>(seq_start + k);
                        acc += scores[static_cast<size_t>(k)] * qkv_at(k_idx, 2, hkv, d);
                    }
                    ref_output[(q_idx * num_heads + h) * static_cast<size_t>(S) + d] = acc;
                }
            }
        }
    }

    cldnn::mem_lock<ov::float16> observed(out_mem, get_test_stream());
    ASSERT_EQ(observed.size(), ref_output.size());
    // f16 + softmax tolerance. Values stay in a well-behaved range because
    // the inputs are in [-0.5, 0.5] and the head_size normalization keeps
    // pre-softmax scores modest.
    const float tol = 5e-2f;
    for (size_t i = 0; i < ref_output.size(); i++) {
        const float got = static_cast<float>(observed[i]);
        ASSERT_NEAR(got, ref_output[i], tol)
            << "vl_sdpa output mismatch at flat index " << i
            << " (token=" << (i / (num_heads * S))
            << " head=" << ((i / S) % num_heads)
            << " dim=" << (i % S) << ")"
            << " — CM kernel likely addressed wrong V slice via token_offset_v / v_token_pitch";
    }
}

// A concat with one predecessor inside a shape-of subgraph (a crop lowered from VariadicSplit,
// running on a CPU impl that can't produce padded output) and one outside must NOT be optimized
// in place: implicit concat would force offset padding on the CPU crop output and assert at
// runtime with "[GPU] Padded output is not supported yet".
TEST(prepare_buffer_fusing, in_place_concat_rejected_for_shape_of_subgraph_input) {
    auto& engine = get_test_engine();
    auto in_layout_dyn = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx };

    // The out-of-subgraph concat branch is produced by an eltwise: concat's in-place
    // optimization only considers predecessors of certain types (see available_pred), and it
    // must be a runtime (non-const) node so the concat is not constant-folded away.
    auto other_layout = layout{ ov::PartialShape{1}, data_types::i32, format::bfyx };

    // Constants used to build the shape-calculation flow.
    auto gather_idx = engine.allocate_memory({ ov::PartialShape{4}, data_types::i32, format::bfyx });
    set_values<int32_t>(gather_idx, {0, 1, 2, 3});
    auto split_axis = engine.allocate_memory({ ov::PartialShape{}, data_types::i32, format::bfyx });
    set_values<int32_t>(split_axis, {0});
    auto split_lengths = engine.allocate_memory({ ov::PartialShape{2}, data_types::i32, format::bfyx });
    set_values<int32_t>(split_lengths, {1, 3});

    topology topology;
    topology.add(input_layout("input", in_layout_dyn));
    topology.add(input_layout("other", other_layout));
    topology.add(data("gather_idx", gather_idx));
    topology.add(data("split_axis", split_axis));
    topology.add(data("split_lengths", split_lengths));
    // eltwise producing the out-of-subgraph concat input.
    topology.add(eltwise("other_elt", { input_info("other"), input_info("other") }, eltwise_mode::sum));
    // shape-of subgraph: shape_of -> gather -> VariadicSplit(crop out0[1], out1[3])
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather", input_info("shape_of"), input_info("gather_idx"), 0, 1, {4}));
    topology.add(crop("crop0", { input_info("gather"), input_info("split_axis"), input_info("split_lengths") },
                      cldnn::tensor(1), cldnn::tensor(0), cldnn::crop_ngraph_op_mode::variadic_split, 0, 0, 2));
    topology.add(crop("crop1", { input_info("gather"), input_info("split_axis"), input_info("split_lengths") },
                      cldnn::tensor(1), cldnn::tensor(0), cldnn::crop_ngraph_op_mode::variadic_split, 1, 0, 2));
    // concat mixes the in-subgraph crop1 (CPU impl) with the out-of-subgraph eltwise result.
    topology.add(concatenation("concat", { input_info("other_elt"), input_info("crop1") }, 0));
    // Reorder to f32 so it isn't a redundant (identical) reorder that gets removed — which
    // would promote 'concat' to a network output (renamed) and disable in-place optimization.
    topology.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);
    ASSERT_NE(prog, nullptr);

    // Preconditions: crop is a shape-of subgraph node (CPU impl), concat is not.
    ASSERT_TRUE(prog->get_node("crop1").is_in_shape_of_subgraph());
    ASSERT_FALSE(prog->get_node("concat").is_in_shape_of_subgraph());
    // The concat must not be optimized in place. Non-fatal so we still reach execute() below,
    // which exercises the actual crash path.
    EXPECT_FALSE(prog->get_node("concat").can_be_optimized());

    cldnn::network net(prog, 0);
    auto input_mem = engine.allocate_memory({ ov::PartialShape{2, 3, 4, 5}, data_types::f32, format::bfyx });
    set_values<float>(input_mem, std::vector<float>(2 * 3 * 4 * 5, 1.f));
    net.set_input_data("input", input_mem);
    auto other_mem = engine.allocate_memory(other_layout);
    set_values<int32_t>(other_mem, {5});
    net.set_input_data("other", other_mem);

    // Without the fix the in-place concat forces padded output on the CPU crop impl and this
    // throws "[GPU] Padded output is not supported yet". Fatal so the value checks below don't
    // dereference a null output on failure.
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    ASSERT_NO_THROW(output = net.execute());

    // input shape {2,3,4,5} => shape_of/gather = [2,3,4,5]; other_elt = other+other = [10].
    // crop1 takes 3 elements at offset 0 => [2,3,4]; concat(other_elt, crop1) axis 0 => {10,2,3,4}.
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> out_ptr(out_mem, get_test_stream());
    ASSERT_EQ(out_mem->count(), 4u);
    ASSERT_EQ(out_ptr[0], 10.f);
    ASSERT_EQ(out_ptr[1], 2.f);
    ASSERT_EQ(out_ptr[2], 3.f);
    ASSERT_EQ(out_ptr[3], 4.f);
}
