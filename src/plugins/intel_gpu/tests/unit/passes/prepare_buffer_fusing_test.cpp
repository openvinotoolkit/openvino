// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "reshape_inst.h"
#include "fully_connected_inst.h"
#include "permute_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "pass_manager.h"
#include "to_string_utils.h"

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
    set_values<float>(input_memory, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7, 4.8, 19.2, -10.1, 8.1, 10.2, 1.3, 1.44, 1.5});
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
    topology.add(fully_connected("fc", input_info("reshape"), "weights", "", {}, 2));
    topology.add(reorder("reorder", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    prog->get_node("reorder").get_output_layout(true);
    program_wrapper::apply_opt_pass<prepare_buffer_fusing>(*prog);
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    ASSERT_NO_THROW(prog->get_node("reshape"));
    ASSERT_TRUE(prog->get_node("reshape").can_be_optimized());
    program_wrapper::apply_opt_pass<build_implementations>(*prog);

    ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory = engine.allocate_memory(layout{ ov::PartialShape{1, 2, 4}, data_types::f32, format::bfyx });
    set_values<float>(input_memory, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7});

    net.set_input_data("input", input_memory);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    ASSERT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("reorder");
    auto out_mem = output.at("reorder").get_memory();

    ASSERT_NE(out_mem, nullptr);
    ov::PartialShape expected_shape = {2, 2};
    ASSERT_EQ(out_mem->count(), 4);
    ASSERT_EQ(out_mem->get_layout().get_partial_shape(), expected_shape);
}

TEST(prepare_buffer_fusing, propagate_data_padding) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ ov::PartialShape{1, 4, 3, 3}, data_types::f32, format::bfyx };

    std::vector<std::pair<primitive_id, tensor>> offsets;
    std::vector<input_info> inputs;
    for (int i = 0; i < 2; i++) {
        auto id = "crop_" + std::to_string(i);
        inputs.push_back(input_info("split:" + id));
        offsets.push_back({ id, {0, (i * 2), 0, 0} });
    }

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(split("split", input_info("input"), offsets));
    topology.add(reorder("crop_0_reorder", inputs[0], format::bfzyx, data_types::f32));
    topology.add(reorder("crop_1_reorder", inputs[1], format::bfzyx, data_types::f32));
    topology.add(concatenation("concat", {input_info("crop_0_reorder"), input_info("crop_1_reorder")}, 1));
    topology.add(reorder("output", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network net(engine, topology, config);

    auto in_mem = engine.allocate_memory(in_layout);
    tests::set_random_values<float>(in_mem);

    net.set_input_data("input", in_mem);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    ASSERT_NO_THROW(output = net.execute());

    auto out_mem = output.at("output").get_memory();

    ASSERT_NE(out_mem, nullptr);
    cldnn::mem_lock<int64_t> output_ptr(out_mem, get_test_stream());
    cldnn::mem_lock<int64_t> input_ptr(in_mem, get_test_stream());

    ASSERT_EQ(input_ptr.size(), output_ptr.size());
    for (size_t i = 0; i < input_ptr.size(); ++i)
    {
        ASSERT_EQ(output_ptr[i], input_ptr[i]);
    }
}

TEST(prepare_buffer_fusing, in_place_concat_static) {
    auto& engine = get_test_engine();
    auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx }; // => {1, 3, 4, 2}
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 1}, data_types::f32, format::bfyx }; // => {1, 1, 4, 2}
    topology topology;
    topology.add(input_layout("input1", in_layout1));
    topology.add(input_layout("input2", in_layout2));
    topology.add(permute("permute1", input_info("input1"), {0, 2, 3, 1}));
    topology.add(permute("permute2", input_info("input2"), {0, 3, 2, 1}));
    topology.add(concatenation("concat", { input_info("permute1"), input_info("permute2") }, 1));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    //config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    ASSERT_NE(prog, nullptr);
    //ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<float>(input_memory1,
        {
            0.0,   1.1,   2.2,   3.3,   4.4,     5.5,    6.6,    7.7,
            8.8,   9.9,   10.10, 11.11, 22.22,   33.33,  44.44,  55.55,
            66.66, 77.77, 88.88, 99.99, 100.100, 101.101,102.102,103.103
            }
    );
    set_values<float>(input_memory2, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7});

    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    const auto& concat_node = net.get_primitive("concat")->get_node();
    auto out_l = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();

    ASSERT_TRUE(concat_node.can_be_optimized());
//    ASSERT_EQ(out_mem->count(), 16);
}

TEST(prepare_buffer_fusing, in_place_concat_dynamic) {
    auto& engine = get_test_engine();
    auto in_layout1_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx }; // => {1, 3, 4, 2}
    auto in_layout2_0 = layout{ ov::PartialShape::dynamic(4), data_types::f32, format::bfyx }; // => {1, 3, 4, 2}
#if 0
   auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx }; // {1, 2, 3, 4} => {1, 3, 4, 2}
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 1}, data_types::f32, format::bfyx }; // {1, 2, 4, 1} => {1, 1, 4, 2}
#endif
    auto in_layout1 = layout{ ov::PartialShape{1, 2, 3, 4}, data_types::f32, format::bfyx }; // {1, 2, 3, 4} =>  {1, 2, 3, 4}
    auto in_layout2 = layout{ ov::PartialShape{1, 2, 4, 1}, data_types::f32, format::bfyx }; // {1, 2, 4, 1} =>  {1, 2, 1, 4]

    topology topology;
    topology.add(input_layout("input1", in_layout1_0));
    topology.add(input_layout("input2", in_layout2_0));
//    topology.add(permute("permute1", input_info("input1"), {0, 2, 3, 1}));
//    topology.add(permute("permute2", input_info("input2"), {0, 3, 2, 1}));
    topology.add(permute("permute1", input_info("input1"), {0, 1, 2, 3}));
    topology.add(permute("permute2", input_info("input2"), {0, 1, 3, 2}));

//    topology.add(concatenation("concat", { input_info("permute1"), input_info("permute2") }, 1));
    topology.add(concatenation("concat", { input_info("permute1"), input_info("permute2") }, 2));
    topology.add(permute("output", input_info("concat"), {0, 2, 3, 1}));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    ASSERT_NE(prog, nullptr);
    //ASSERT_TRUE(has_node_with_type<reshape>(*prog));

    cldnn::network net(prog, 0);

    auto input_memory1 = engine.allocate_memory(in_layout1);
    auto input_memory2 = engine.allocate_memory(in_layout2);
    set_values<float>(input_memory1,
        {
            0.0,   1.1,   2.2,   3.3,   4.4,     5.5,    6.6,    7.7,
            8.8,   9.9,   10.10, 11.11, 22.22,   33.33,  44.44,  55.55,
            66.66, 77.77, 88.88, 99.99, 100.100, 101.101,102.102,103.103
            }
    );
    set_values<float>(input_memory2, {0.1, 1.1, 2.2, 3.0, 4.0, -5.0, 0.1, 0.7});
#if 0 // faxis
    std::vector<float> ref_output = {
        0.0, 4.4, 8.8,   0.1, 22.22, 66.66, 100.100, 4.0, 1.1, 5.5, 9.9,   1.1, 33.33, 77.77, 101.101, -5.0,
        2.2, 6.6, 10.10, 2.2, 44.44, 88.88, 102.102, 0.1, 3.3, 7.7, 11.11, 3.0, 55.55, 99.99, 103.103, 0.7
    };
#endif
    std::vector<float> ref_output = {
        0.0, 22.22,   1.1, 33.33,   2.2,   44.44,   3.3,   55.55,   4.4, 66.66, 5.5, 77.77, 6.6, 88.88, 7.7, 99.99,
        8.8,      100.100, 9.9, 101.101, 10.10, 102.102, 11.11, 103.103, 0.1, 4.0,   1.1, -5.0,  2.2, 0.1,   3.0, 0.7};
    net.set_input_data("input1", input_memory1);
    net.set_input_data("input2", input_memory2);
    std::map<cldnn::primitive_id, cldnn::network_output> output;
    EXPECT_NO_THROW(output = net.execute());
    auto out_l = net.get_output_layout("output");
    auto out_mem = output.at("output").get_memory();
    cldnn::mem_lock<float> output_ptr(out_mem, get_test_stream());

    ASSERT_NE(out_mem, nullptr);
    ASSERT_EQ(out_mem->count(), 32);
    for (size_t x = 0; x < out_l.count(); ++x) {
        ASSERT_EQ(ref_output[x], output_ptr[x]);
    }

}