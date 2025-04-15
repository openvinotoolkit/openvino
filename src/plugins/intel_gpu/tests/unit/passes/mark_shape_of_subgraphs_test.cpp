// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "gather_inst.h"
#include "eltwise_inst.h"
#include "concatenation_inst.h"
#include "scatter_update_inst.h"
#include "select_inst.h"
#include "strided_slice_inst.h"
#include "broadcast_inst.h"
#include "paged_attention_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

static bool check_subgraph(const program_node& node, const program_node& last_node, std::map<std::string, size_t> custom_dependant_nodes_count = {}) {
    size_t expected_dependant_nodes = 1;
    if (custom_dependant_nodes_count.find(node.id()) != custom_dependant_nodes_count.end())
        expected_dependant_nodes = custom_dependant_nodes_count[node.id()];

    // Skip some custom nodes if they are not intended to be included into shape_of subgraph
    if (expected_dependant_nodes == 0)
        return true;

    if (!node.is_in_shape_of_subgraph() || node.get_dependant_shape_of_nodes().size() != expected_dependant_nodes)
        return false;

    // Check if there are no any extra nodes added to subgraph
    if (&node == &last_node) {
        for (auto user : node.get_users())
            if (user->is_in_shape_of_subgraph())
                return false;
        return true;
    }

    for (auto user : node.get_users())
        if (!check_subgraph(*user, last_node, custom_dependant_nodes_count))
            return false;

    return true;
}

TEST(mark_shape_of_subgraphs, simple_chain) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    set_values(data_0, {0});
    set_values(data_1, {2});
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather", input_info("shape_of"), input_info("data_0"), 0, 0, {}));
    topology.add(eltwise("eltwise", input_info("gather"), input_info("data_1"), eltwise_mode::sum));
    topology.add(concatenation("concat", {input_info("eltwise"), input_info("data_1")}, 0));
    topology.add(broadcast("broadcast", input_info("input"), input_info("concat"), {}, ov::op::BroadcastType::BIDIRECTIONAL));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("concat")));

    auto input_mem = engine.allocate_memory({ov::PartialShape{1, 1}, data_types::f32, format::bfyx});
    set_values(input_mem, {10.f});
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());
    ASSERT_EQ(6, output_prim->get_layout().count());
    for (size_t i = 0; i < output_prim->get_layout().count(); ++i) {
        ASSERT_EQ(10.0f, output_ptr[i]);
    }
}

TEST(mark_shape_of_subgraphs, simple_chain_w_reshape_inside_subgraph) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{2}, data_types::i32, format::bfyx });
    set_values<int32_t>(data_1, {1, 1});

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather", input_info("shape_of"), input_info("data_0"), 0, 1, {1}));
    topology.add(reshape("reshape", input_info("gather"), input_info("data_1"), false, ov::PartialShape{2}));
    topology.add(broadcast("broadcast", input_info("input"), input_info("reshape"), {}, ov::op::BroadcastType::BIDIRECTIONAL));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("reshape")));
}

TEST(mark_shape_of_subgraphs, parallel_shape_of_subgraphs) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{}, data_types::i32, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(shape_of("shape_of_0", input_info("input"), data_types::i32));
    topology.add(shape_of("shape_of_1", input_info("input"), data_types::i32));
    topology.add(gather("gather_0", input_info("shape_of_0"), input_info("data_0"), 0, 0, {}));
    topology.add(gather("gather_1", input_info("shape_of_1"), input_info("data_0"), 0, 0, {}));
    topology.add(eltwise("eltwise", input_info("gather_0"), input_info("gather_1"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("input"), input_info("eltwise"), false, ov::PartialShape()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_0"), prog->get_node("eltwise"), {{"eltwise", 2}}));
    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_1"), prog->get_node("eltwise"), {{"eltwise", 2}}));
}

TEST(mark_shape_of_subgraphs, parallel_shape_of_subgraphs_cascade) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{1, 4, 8, 16}, data_types::i32, format::bfyx });
    auto data_2 = engine.allocate_memory({ ov::PartialShape{1}, data_types::f16, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(data("data_2", data_2));
    topology.add(shape_of("shape_of_0", input_info("input"), data_types::i32));
    topology.add(gather("gather_0", input_info("shape_of_0"), input_info("data_0"), 0, 1, {1}));
    topology.add(shape_of("shape_of_1", input_info("input"), data_types::i32));
    topology.add(gather("gather_1", input_info("shape_of_1"), input_info("data_0"), 0, 1, {1}));
    topology.add(scatter_update("scatter_update_0", input_info("gather_0"), input_info("data_0"), input_info("data_0"), 0));
    topology.add(scatter_update("scatter_update_1", input_info("gather_1"), input_info("data_0"), input_info("data_0"), 0));
    topology.add(strided_slice("strided_slice_1",
                               input_info("data_1"),
                               input_info("scatter_update_0"),
                               input_info("scatter_update_1"),
                               input_info("data_0"), {}, {}, {}, {}, {}, {}));
    topology.add(shape_of("shape_of_2", input_info("input"), data_types::i32));
    topology.add(gather("gather_2", input_info("shape_of_2"), input_info("data_0"), 0, 0, {}));
    topology.add(scatter_update("scatter_update_2", input_info("gather_2"), input_info("data_0"), input_info("data_0"), 0));
    topology.add(strided_slice("strided_slice_2",
                               input_info("data_1"),
                               input_info("strided_slice_1"),
                               input_info("scatter_update_2"),
                               input_info("data_0"), {}, {}, {}, {}, {}, {}));
    topology.add(reshape("reshape", input_info("data_2"), input_info("strided_slice_2"), true, {}));

    topology.add(cldnn::select("select", input_info("data_0"), input_info("input"), input_info("reshape")));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    std::map<std::string, size_t> custom_dependant_nodes_count = {{"strided_slice_1", 2}, {"strided_slice_2", 3}, {"reshape", 3}};

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_0"), prog->get_node("reshape"), custom_dependant_nodes_count));
    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_1"), prog->get_node("reshape"), custom_dependant_nodes_count));
    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_2"), prog->get_node("reshape"), custom_dependant_nodes_count));
}

TEST(mark_shape_of_subgraphs, simple_chain_w_inserted_reorder) {
    // This test covers marking of newely added nodes during graph optimization passes
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather", input_info("shape_of"), input_info("data_0"), 0, 1, {1}));
    topology.add(reshape("reshape", input_info("gather"), true, {}, {}));
    topology.add(reorder("reorder", input_info("reshape"), format::bfyx, data_types::f16));
    topology.add(eltwise("eltwise", input_info("reorder"), input_info("data_0"), eltwise_mode::prod));
    topology.add(reshape("reshape_2", input_info("input"), input_info("eltwise"), false, ov::PartialShape()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("eltwise")));
}

TEST(mark_shape_of_subgraphs, concat_with_empty_tensor_inputs) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), 4}, data_types::f32, format::bfyx};
    auto input_layout_empty = layout{ov::PartialShape{}, data_types::f32, format::bfyx};

    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    set_values(data_0, {0});

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(input_layout("input_empty", input_layout_empty));
    topology.add(data("data_0", data_0));
    topology.add(shape_of("shape_of_01", input_info("input"), data_types::i32));
    topology.add(gather("gather01", input_info("shape_of_01"), input_info("data_0"), 0, 1, {1}));
    topology.add(shape_of("shape_of_02", input_info("input_empty"), data_types::i32));
    topology.add(shape_of("shape_of_03", input_info("input_empty"), data_types::i32));
    topology.add(concatenation("concat", {input_info("gather01"), input_info("shape_of_02"), input_info("shape_of_03")}, 0));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_01"), prog->get_node("concat"), {{"concat", 3}}));
    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_02"), prog->get_node("concat"), {{"concat", 3}}));
    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of_03"), prog->get_node("concat"), {{"concat", 3}}));

    auto input_mem = engine.allocate_memory({ov::PartialShape{5, 4}, data_types::f32, format::bfyx});
    set_values(input_mem, {10.f});
    network.set_input_data("input", input_mem);

    auto input_empty_mem = engine.allocate_memory(input_layout_empty);
    network.set_input_data("input_empty", input_empty_mem);

    auto outputs = network.execute();
    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<int32_t> output_ptr (output_prim, get_test_stream());
    ASSERT_EQ(1, output_prim->get_layout().count());
    for (size_t i = 0; i < output_prim->get_layout().count(); ++i) {
        ASSERT_EQ(5, output_ptr[i]);
    }
    set_values(input_mem, {20.f});
    network.set_input_data("input", input_mem);
    auto outputs2 = network.execute();
    auto output_prim2 = outputs.begin()->second.get_memory();

    cldnn::mem_lock<int32_t> output_ptr2 (output_prim2, get_test_stream());
    ASSERT_EQ(1, output_prim2->get_layout().count());
    for (size_t i = 0; i < output_prim2->get_layout().count(); ++i) {
        ASSERT_EQ(5, output_ptr2[i]);
    }
}

TEST(mark_shape_of_subgraphs, gather_compressed_no_mark) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    auto decompression_scale = engine.allocate_memory({ ov::PartialShape{1}, data_types::f32, format::bfyx });
    auto decompression_zero_point = engine.allocate_memory({ ov::PartialShape{1}, data_types::f32, format::bfyx });
    set_values(data_0, {0});
    set_values(data_1, {2});
    set_values(decompression_scale, {2});
    set_values(decompression_zero_point, {2});

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(data("decompression_scale", decompression_scale));
    topology.add(data("decompression_zero_point", decompression_zero_point));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather_compressed", input_info("shape_of"), input_info("data_0"), 0,
                        input_info("decompression_scale"), input_info("decompression_zero_point"), ov::element::f32, 0, {}));
    topology.add(eltwise("eltwise", input_info("gather_compressed"), input_info("data_1"), eltwise_mode::sum));
    topology.add(concatenation("concat", {input_info("eltwise"), input_info("data_1")}, 0));
    topology.add(broadcast("broadcast", input_info("input"), input_info("concat"), {}, ov::op::BroadcastType::BIDIRECTIONAL));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_FALSE(check_subgraph(prog->get_node("shape_of"), prog->get_node("gather_compressed")));
    ASSERT_FALSE(check_subgraph(prog->get_node("shape_of"), prog->get_node("concat")));
}

TEST(mark_shape_of_subgraphs, broadcast_not_existed_after_shapeof) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), 4, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{4}, data_types::i32, format::bfyx });
    set_values(data_0, {1, 4, 1, 1});
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, {1152, 4, 1, 1} });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("weights", weights));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(reshape("reshape", input_info("shape_of"), input_info("data_0"), false, {}));
    topology.add(convolution("convolution", input_info("reshape"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("convolution")));
}

TEST(mark_shape_of_subgraphs, broadcast_w_data_and_direct_shapeof_no_mark) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), 4, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    set_values(data_0, {0});
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, {1152, 4, 2, 2} });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(broadcast("broadcast", input_info("data_0"), input_info("shape_of"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    topology.add(data("weights", weights));
    topology.add(convolution("convolution", input_info("broadcast"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_FALSE(check_subgraph(prog->get_node("shape_of"), prog->get_node("convolution")));
    ASSERT_FALSE(check_subgraph(prog->get_node("shape_of"), prog->get_node("broadcast")));
}

TEST(mark_shape_of_subgraphs, broadcast_w_data_and_indirect_shapeof) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), 4, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto data_0 = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    set_values(data_0, {0});

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather", input_info("shape_of"), input_info("data_0"), 0, 0, {}));
    topology.add(broadcast("broadcast", input_info("data_0"), input_info("gather"), {}, ov::op::BroadcastType::BIDIRECTIONAL));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("broadcast")));
}

TEST(mark_shape_of_subgraphs, broadcast_w_direct_shapeof_and_data) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), 4, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto target_shape = engine.allocate_memory({ ov::PartialShape{4}, data_types::i32, format::bfyx });
    set_values(target_shape, {4, 4, 1, 1});

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("target_shape", target_shape));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(broadcast("broadcast", input_info("shape_of"), input_info("target_shape"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    topology.add(reshape("reshape", input_info("input"), input_info("broadcast"), false, ov::PartialShape{4, 4, 1, 1}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("broadcast")));
}

TEST(mark_shape_of_subgraphs, paged_attention_max_context_len_input) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), 4, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f32, format::bfyx};
    auto target_shape = engine.allocate_memory({ ov::PartialShape{4}, data_types::i32, format::bfyx });
    set_values(target_shape, {4, 4, 1, 1});

    auto subtract_one = engine.allocate_memory({ ov::PartialShape{1}, data_types::i32, format::bfyx });
    set_values(target_shape, {-1});

    auto query_layout = layout{ov::PartialShape{ov::Dimension::dynamic(), 128},
                               data_types::f32,
                               format::bfyx};
    auto key_layout = query_layout;
    auto value_layout = query_layout;
    auto key_cache_layout = layout{ov::PartialShape{ov::Dimension::dynamic(), 2, 64, 16},
                                   data_types::f32,
                                   format::bfyx};
    auto dynamic_i32_layout = layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx};
    auto value_cache_layout = key_cache_layout;
    auto past_lens_layout = dynamic_i32_layout;
    auto subsequence_begins_layout = dynamic_i32_layout;
    auto block_indices_layout = dynamic_i32_layout;
    auto block_indices_begins_layout = dynamic_i32_layout;
    auto scale_layout = layout{ov::PartialShape{1}, data_types::f32, format::bfyx};
    auto sliding_window_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto alibi_layout = layout{ov::PartialShape{}, data_types::f32, format::bfyx};
    auto max_context_len_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};;

    std::vector<input_info> pa_inputs = {input_info("query"),
                                         input_info("key"),
                                         input_info("value"),
                                         input_info("key_cache"),
                                         input_info("value_cache"),
                                         input_info("past_lens"),
                                         input_info("subsequence_begins"),
                                         input_info("block_indices"),
                                         input_info("block_indices_begins"),
                                         input_info("scale"),
                                         input_info("sliding_window"),
                                         input_info("alibi"),
                                         input_info("max_context_len")};

    auto pa_prim = paged_attention("paged_attention", pa_inputs);
    pa_prim.head_size = 64;
    pa_prim.kv_heads_num = 2;
    pa_prim.heads_num = 2;
    pa_prim.scale_val = 1.f;
    pa_prim.has_alibi = false;
    pa_prim.num_outputs = 1;
    pa_prim.has_rotated_blocks = false;

    topology topology;
    topology.add(input_layout("query", query_layout));
    topology.add(input_layout("key", key_layout));
    topology.add(input_layout("value", value_layout));
    topology.add(input_layout("key_cache", key_cache_layout));
    topology.add(input_layout("value_cache", value_cache_layout));
    topology.add(input_layout("past_lens", past_lens_layout));
    topology.add(input_layout("subsequence_begins", subsequence_begins_layout));
    topology.add(input_layout("block_indices", block_indices_layout));
    topology.add(input_layout("block_indices_begins", block_indices_begins_layout));
    topology.add(input_layout("scale", scale_layout));
    topology.add(input_layout("sliding_window", sliding_window_layout));
    topology.add(input_layout("alibi", alibi_layout));
    topology.add(input_layout("max_context_len", max_context_len_layout));
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("target_shape", target_shape));
    topology.add(data("subtract_one", subtract_one));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(broadcast("broadcast", input_info("shape_of"), input_info("target_shape"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    topology.add(eltwise("subtract_one_max_context_len", input_info("max_context_len"), input_info("subtract_one"), eltwise_mode::sum));
    topology.add(eltwise("updated_broadcast", input_info("broadcast"), input_info("subtract_one_max_context_len"), eltwise_mode::sum));
    topology.add(reshape("reshape", input_info("input"), input_info("updated_broadcast"), false, ov::PartialShape::dynamic(4)));
    topology.add(pa_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    ASSERT_TRUE(check_subgraph(prog->get_node("shape_of"), prog->get_node("updated_broadcast"), {{"updated_broadcast", 2}}));
    ASSERT_TRUE(check_subgraph(prog->get_node("max_context_len"), prog->get_node("updated_broadcast"), {{"updated_broadcast", 2}, {"paged_attention", 0}}));
}
