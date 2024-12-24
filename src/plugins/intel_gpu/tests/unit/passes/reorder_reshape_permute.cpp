// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "permute_inst.h"
#include "program_wrapper.h"
#include "reshape_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

TEST(opt_reorder_reshape_permute, no_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({3, 2, 1, 1}), data_types::f16, format::bfyx});

    set_values<ov::float16>(input, {2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.f,  2.f,  3.f,  1.f,  2.f,  4.f,
                                    5.f,  1.f,  1.f,  2.f,  1.f,  2.f,  2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f,
                                    3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.f,  4.f,  1.f,  1.f,  2.f,  1.f,
                                    1.f,  2.f,  0.f,  2.f,  5.f,  2.f,  4.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f});

    set_values<ov::float16>(weight, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(
        convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reorder("reorder_inter", input_info("convolution"), format::bfyx, data_types::f16));
    topology.add(permute("permute_inter", input_info("reorder_inter"), {0, 2, 3, 1}));
    topology.add(softmax("softmax", input_info("permute_inter"), 1));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);

    network net(prog);

    net.set_input_data("input", input);
    auto output = net.execute();

    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);
    // reorder node is removed in primitive fusing
    // later permute is optimized after convolution in selected preferred formats, e.g conv + permute
    auto optimzed_nodes = net.get_program()->get_optimized();
    auto it =
        std::find_if(std::begin(optimzed_nodes), std::end(optimzed_nodes), [&](cldnn::program::optimized_info& oi) {
            return oi.first == "reorder_inter";
        });
    ASSERT_NE(it, optimzed_nodes.end());
    auto permute_inst = net.get_primitive("permute_inter");
    ASSERT_TRUE(permute_inst->can_be_optimized());
    auto out_mem = output.at("softmax").get_memory();
    mem_lock<ov::float16> lock(out_mem, get_test_stream());

    ref_network.set_input_data("input", input);
    auto ref_output = ref_network.execute();
    auto ref_out_mem = ref_output.at("softmax").get_memory();
    mem_lock<ov::float16> lock_ref(ref_out_mem, get_test_stream());
    for (size_t i = 0; i < out_mem->count(); i++) {
        float actual = lock[i];
        ASSERT_EQ(actual, lock_ref[i]);
    }
}

TEST(opt_reorder_reshape_permute, no_reorder_no_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({3, 2, 1, 1}), data_types::f16, format::bfyx});
    set_values<ov::float16>(input, {2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.f,  2.f,  3.f,  1.f,  2.f,  4.f,
                                    5.f,  1.f,  1.f,  2.f,  1.f,  2.f,  2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f,
                                    3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.f,  4.f,  1.f,  1.f,  2.f,  1.f,
                                    1.f,  2.f,  0.f,  2.f,  5.f,  2.f,  4.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f});

    set_values<ov::float16>(weight, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(
        convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(permute("permute_inter", input_info("convolution"), {0, 2, 3, 1}));
    topology.add(softmax("softmax", input_info("permute_inter"), 1));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);

    network net(prog);

    net.set_input_data("input", input);
    auto output = net.execute();

    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);
    // select preferred formats, conv + permute
    auto permute_inst = net.get_primitive("permute_inter");
    ASSERT_TRUE(permute_inst->can_be_optimized());
    auto out_mem = output.at("softmax").get_memory();
    mem_lock<ov::float16> lock(out_mem, get_test_stream());

    ref_network.set_input_data("input", input);
    auto ref_output = ref_network.execute();
    auto ref_out_mem = ref_output.at("softmax").get_memory();
    mem_lock<ov::float16> lock_ref(ref_out_mem, get_test_stream());
    for (size_t i = 0; i < out_mem->count(); i++) {
        float actual = lock[i];
        ASSERT_EQ(actual, lock_ref[i]);
    }
}

TEST(opt_reorder_reshape_permute, cutomized_net_yolov6_alike) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({3, 2, 1, 1}), data_types::f16, format::bfyx});
    set_values<ov::float16>(input, {2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.f,  2.f,  3.f,  1.f,  2.f,  4.f,
                                    5.f,  1.f,  1.f,  2.f,  1.f,  2.f,  2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f,
                                    3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.f,  4.f,  1.f,  1.f,  2.f,  1.f,
                                    1.f,  2.f,  0.f,  2.f,  5.f,  2.f,  4.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f});

    set_values<ov::float16>(weight, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(
        convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reorder("reorder_inter", input_info("convolution"), format::bfyx, data_types::f16));
    topology.add(
        reshape("reshape_inter", input_info("reorder_inter"), false, {1, 3, 24, 1}, ov::PartialShape{1, 3, 24, 1}));
    topology.add(permute("permute_inter", input_info("reshape_inter"), {0, 2, 1}));
    topology.add(softmax("softmax", input_info("permute_inter"), 1));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);
    network net(prog);

    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);

    net.set_input_data("input", input);
    auto output = net.execute();
    auto optimzed_nodes = net.get_program()->get_optimized();
    auto it =
        std::find_if(std::begin(optimzed_nodes), std::end(optimzed_nodes), [&](cldnn::program::optimized_info& oi) {
            return oi.first == "reorder_inter";
        });
    ASSERT_NE(it, optimzed_nodes.end());
    auto permute_inst = net.get_primitive("permute_inter");
    ASSERT_TRUE(permute_inst->can_be_optimized());
    auto reshape_inst = net.get_primitive("reshape_inter");
    ASSERT_TRUE(reshape_inst->can_be_optimized());

    auto& processing_order = prog->get_processing_order();

    auto reshape_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("reshape_inter"));
    size_t reshape_dist = std::distance(processing_order.begin(), reshape_node);

    auto permute_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("permute_inter"));
    size_t permute_dist = std::distance(processing_order.begin(), permute_node);
    ASSERT_TRUE(reshape_dist > permute_dist);
    auto out_mem = output.at("softmax").get_memory();
    mem_lock<ov::float16> lock(out_mem, get_test_stream());

    ref_network.set_input_data("input", input);
    auto ref_output = ref_network.execute();

    auto ref_out_mem = ref_output.at("softmax").get_memory();
    mem_lock<ov::float16> lock_ref(ref_out_mem, get_test_stream());
    for (size_t i = 0; i < out_mem->count(); i++) {
        float actual = lock[i];
        ASSERT_EQ(actual, lock_ref[i]);
    }
}

TEST(opt_reorder_reshape_permute, cutomized_net_yolov6_alike_4d) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({3, 2, 1, 1}), data_types::f16, format::bfyx});
    set_values<ov::float16>(input, {2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.f,  2.f,  3.f,  1.f,  2.f,  4.f,
                                    5.f,  1.f,  1.f,  2.f,  1.f,  2.f,  2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f,
                                    3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.f,  4.f,  1.f,  1.f,  2.f,  1.f,
                                    1.f,  2.f,  0.f,  2.f,  5.f,  2.f,  4.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f});

    set_values<ov::float16>(weight, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(
        convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reorder("reorder_inter", input_info("convolution"), format::bfyx, data_types::f16));
    topology.add(
        reshape("reshape_inter", input_info("reorder_inter"), false, {1, 3, 24, 1}, ov::PartialShape{1, 3, 24, 1}));
    topology.add(permute("permute_inter", input_info("reshape_inter"), {0, 2, 1, 3}));
    topology.add(softmax("softmax", input_info("permute_inter"), 1));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);
    network net(prog);

    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);

    net.set_input_data("input", input);
    auto output = net.execute();
    auto optimzed_nodes = net.get_program()->get_optimized();
    auto it =
        std::find_if(std::begin(optimzed_nodes), std::end(optimzed_nodes), [&](cldnn::program::optimized_info& oi) {
            return oi.first == "reorder_inter";
        });
    ASSERT_NE(it, optimzed_nodes.end());
    auto permute_inst = net.get_primitive("permute_inter");
    ASSERT_TRUE(permute_inst->can_be_optimized());
    auto reshape_inst = net.get_primitive("reshape_inter");
    ASSERT_TRUE(reshape_inst->can_be_optimized());

    auto& processing_order = prog->get_processing_order();

    auto reshape_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("reshape_inter"));
    size_t reshape_dist = std::distance(processing_order.begin(), reshape_node);

    auto permute_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("permute_inter"));
    size_t permute_dist = std::distance(processing_order.begin(), permute_node);
    ASSERT_TRUE(reshape_dist > permute_dist);
    auto out_mem = output.at("softmax").get_memory();
    mem_lock<ov::float16> lock(out_mem, get_test_stream());

    ref_network.set_input_data("input", input);
    auto ref_output = ref_network.execute();

    auto ref_out_mem = ref_output.at("softmax").get_memory();
    mem_lock<ov::float16> lock_ref(ref_out_mem, get_test_stream());
    for (size_t i = 0; i < out_mem->count(); i++) {
        float actual = lock[i];
        ASSERT_EQ(actual, lock_ref[i]);
    }
}

TEST(opt_reorder_reshape_permute, not_sinking_reshape) {
    auto& engine = get_test_engine();
    auto in_layout = layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 2, 4, 6}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({3, 2, 1, 1}), data_types::f16, format::bfyx});
    set_values<ov::float16>(input, {2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.f,  2.f,  3.f,  1.f,  2.f,  4.f,
                                    5.f,  1.f,  1.f,  2.f,  1.f,  2.f,  2.0f, 3.0f, 1.0f, 4.0f, 1.0f, 4.0f,
                                    3.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f, 2.f,  4.f,  1.f,  1.f,  2.f,  1.f,
                                    1.f,  2.f,  0.f,  2.f,  5.f,  2.f,  4.0f, 3.0f, 1.0f, 0.0f, 3.0f, 2.0f});

    set_values<ov::float16>(weight, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(
        convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(reorder("reorder_inter", input_info("convolution"), format::bfyx, data_types::f16));
    topology.add(
        reshape("reshape_inter", input_info("reorder_inter"), false, {1, 18, 4, 1}, ov::PartialShape{1, 18, 4, 1}));
    topology.add(permute("permute_inter", input_info("reshape_inter"), {0, 2, 1}));
    topology.add(softmax("softmax", input_info("permute_inter"), 1));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(false));
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto prog = program::build_program(engine, topology, config);
    network net(prog);

    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);

    net.set_input_data("input", input);
    auto output = net.execute();
    auto optimzed_nodes = net.get_program()->get_optimized();
    auto it =
        std::find_if(std::begin(optimzed_nodes), std::end(optimzed_nodes), [&](cldnn::program::optimized_info& oi) {
            return oi.first == "reorder_inter";
        });
    ASSERT_NE(it, optimzed_nodes.end());
    auto permute_inst = net.get_primitive("permute_inter");
    ASSERT_FALSE(permute_inst->can_be_optimized());
    auto reshape_inst = net.get_primitive("reshape_inter");
    ASSERT_FALSE(reshape_inst->can_be_optimized());

    auto& processing_order = prog->get_processing_order();

    auto reshape_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("reshape_inter"));
    size_t reshape_dist = std::distance(processing_order.begin(), reshape_node);

    auto permute_node = std::find(processing_order.begin(), processing_order.end(), &prog->get_node("permute_inter"));
    size_t permute_dist = std::distance(processing_order.begin(), permute_node);
    ASSERT_TRUE(reshape_dist < permute_dist);
    auto out_mem = output.at("softmax").get_memory();
    mem_lock<ov::float16> lock(out_mem, get_test_stream());

    ref_network.set_input_data("input", input);
    auto ref_output = ref_network.execute();

    auto ref_out_mem = ref_output.at("softmax").get_memory();
    mem_lock<ov::float16> lock_ref(ref_out_mem, get_test_stream());
    for (size_t i = 0; i < out_mem->count(); i++) {
        float actual = lock[i];
        std::cout << actual << ", " << std::endl;
        ASSERT_EQ(actual, lock_ref[i]);
    }
}
