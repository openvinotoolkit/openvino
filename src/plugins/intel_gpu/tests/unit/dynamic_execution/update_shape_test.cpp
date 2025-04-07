// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/gather.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace update_shape_tests {
TEST(update_shape_test, ocl_impl_in_shapeof_subgraph) {
    auto& engine = get_test_engine();

    layout const1_gather_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto const1_gather = engine.allocate_memory(const1_gather_layout);
    set_values<int32_t>(const1_gather, {1});

    layout const_broadcast_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto const_broadcast = engine.allocate_memory(const_broadcast_layout);
    set_values<int32_t>(const_broadcast, {1});

    layout input_l= layout{ov::PartialShape{1, 128}, data_types::i32, format::bfyx};
    auto input_mem = engine.allocate_memory(input_l);
    set_values<int32_t>(input_mem, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    1, 2, 3, 4, 5, 6, 7, 8,});

    auto input_l_dynamic = layout{ov::PartialShape::dynamic(2), data_types::i32, format::bfyx};
    topology topology(input_layout("input", input_l_dynamic),
                      data("const1_gather", const1_gather),
                      data("const_broadcast", const_broadcast),
                      shape_of("shape_of", input_info("input"), data_types::i32),
                      gather("gather", input_info("shape_of"), input_info("const1_gather"), 0, 1, ov::Shape({1})),
                      broadcast("broadcast1", input_info("const_broadcast"), input_info("gather"), {}, ov::op::BroadcastType::NUMPY),
                      count_nonzero("count_nonzero", input_info("broadcast1")),
                      gather_nonzero("gather_nonzero", input_info("broadcast1"), input_info("count_nonzero")),
                      broadcast("broadcast2", input_info("gather_nonzero"), input_info("shape_of"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    std::map<primitive_id, network_output> outputs;
    OV_ASSERT_NO_THROW(outputs = network.execute());
}

TEST(update_shape_test, max_context_len_shapeof_subgraph) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto input_data_layout = layout{ov::PartialShape{1, -1}, data_types::f16, format::bfyx};

    auto qkv_mem_layout = layout{ov::PartialShape{1, 128}, data_types::f16, format::bfyx};
    auto qkv_mem = engine.allocate_memory(qkv_mem_layout);
    auto qkv_rnd = rg.generate_random_1d<ov::float16>(qkv_mem_layout.count(), 0, 10);
    set_values(qkv_mem, qkv_rnd);

    auto key_cache_mem_layout = layout{ov::PartialShape{1, 2, 64, 16}, data_types::f16, format::bfyx};
    auto value_cache_mem_layout = layout{ov::PartialShape{1, 2, 16, 64}, data_types::f16, format::bfyx};
    auto key_cache_mem = engine.allocate_memory(key_cache_mem_layout);
    auto value_cache_mem = engine.allocate_memory(value_cache_mem_layout);
    auto cache_rnd = rg.generate_random_1d<ov::float16>(key_cache_mem_layout.count(), 0, 10);
    set_values(key_cache_mem, cache_rnd);
    set_values(value_cache_mem, cache_rnd);

    auto past_lens_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto past_lens_mem = engine.allocate_memory(past_lens_mem_layout);
    set_values(value_cache_mem, {8});

    auto subsequence_begins_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto subsequence_begins_mem = engine.allocate_memory(subsequence_begins_mem_layout);
    set_values(subsequence_begins_mem, {0, 1});

    auto block_indices_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_indices_mem = engine.allocate_memory(block_indices_mem_layout);
    set_values(block_indices_mem, {0});

    auto block_indices_begins_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_mem_layout);
    set_values(block_indices_begins_mem, {0, 1});

    auto scale_mem_layout = layout{ov::PartialShape{1}, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_mem_layout);
    set_values<ov::float16>(scale_mem, {1});

    auto sliding_window_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto sliding_window_mem = engine.allocate_memory(sliding_window_mem_layout);
    set_values(sliding_window_mem, {0});

    auto alibi_mem_layout = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};
    auto alibi_mem = engine.allocate_memory(alibi_mem_layout);

    auto const_one_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto const_one_mem = engine.allocate_memory(const_one_layout);
    set_values(const_one_mem, {1});

    auto input_data_mem_layout = layout{ov::PartialShape{1, 9}, data_types::f16, format::bfyx};
    auto input_data_mem = engine.allocate_memory(input_data_mem_layout);
    auto input_data_rnd = rg.generate_random_1d<ov::float16>(input_data_mem_layout.count(), 0, 10);
    set_values(input_data_mem, input_data_rnd);

    auto query_layout = layout{ov::PartialShape{-1, 128}, data_types::f16, format::bfyx};
    auto key_layout = query_layout;
    auto value_layout = query_layout;
    auto key_cache_layout = layout{ov::PartialShape{-1, 2, 64, 16}, data_types::f16, format::bfyx};
    auto dynamic_i32_layout = layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx};
    auto value_cache_layout = key_cache_layout;
    auto past_lens_layout = dynamic_i32_layout;
    auto subsequence_begins_layout = dynamic_i32_layout;
    auto block_indices_layout = dynamic_i32_layout;
    auto block_indices_begins_layout = dynamic_i32_layout;
    auto scale_layout = layout{ov::PartialShape{1}, data_types::f16, format::bfyx};
    auto sliding_window_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto alibi_layout = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};
    auto max_context_len_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};

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
    topology.add(input_layout("input_data", input_data_layout));
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
    topology.add(data("const_one", const_one_mem));
    topology.add(shape_of("shape_of", input_info("input_data"), data_types::i32));
    topology.add(gather("gather", input_info("shape_of"), input_info("const_one"), 0, 1, ov::Shape{}));
    topology.add(broadcast("broadcast", input_info("gather"), input_info("max_context_len"), {}, ov::op::BroadcastType::BIDIRECTIONAL));
    topology.add(pa_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input_data", input_data_mem);
    network.set_input_data("query", qkv_mem);
    network.set_input_data("key", qkv_mem);
    network.set_input_data("value", qkv_mem);
    network.set_input_data("key_cache", key_cache_mem);
    network.set_input_data("value_cache", value_cache_mem);
    network.set_input_data("past_lens", past_lens_mem);
    network.set_input_data("subsequence_begins", subsequence_begins_mem);
    network.set_input_data("block_indices", block_indices_mem);
    network.set_input_data("block_indices_begins", block_indices_begins_mem);
    network.set_input_data("scale", scale_mem);
    network.set_input_data("sliding_window", sliding_window_mem);
    network.set_input_data("alibi", alibi_mem);

    // Set original max_context_len value
    auto max_context_len_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto max_context_len_mem = engine.allocate_memory(max_context_len_mem_layout);
    set_values(max_context_len_mem, {9});

    network.set_input_data("max_context_len", max_context_len_mem);

    // 1st network execution
    network.execute();

    auto broadcast_inst = network.get_primitive("broadcast");
    ASSERT_EQ(broadcast_inst->get_node().get_dependant_shape_of_nodes().size(), 2);

    // Verify broadcast shape after first execution
    auto broadcast_shape = broadcast_inst->get_impl_params()->get_output_layout().get_shape();
    ASSERT_EQ(broadcast_shape, ov::Shape{9});

    network.set_input_data("input_data", input_data_mem);
    network.set_input_data("query", qkv_mem);
    network.set_input_data("key", qkv_mem);
    network.set_input_data("value", qkv_mem);
    network.set_input_data("key_cache", key_cache_mem);
    network.set_input_data("value_cache", value_cache_mem);
    network.set_input_data("past_lens", past_lens_mem);
    network.set_input_data("subsequence_begins", subsequence_begins_mem);
    network.set_input_data("block_indices", block_indices_mem);
    network.set_input_data("block_indices_begins", block_indices_begins_mem);
    network.set_input_data("scale", scale_mem);
    network.set_input_data("sliding_window", sliding_window_mem);
    network.set_input_data("alibi", alibi_mem);

    // Update max_context_len value, which should be taken into account in shape recalculation for broadcast
    set_values(max_context_len_mem, {8});

    network.set_input_data("max_context_len", max_context_len_mem);

    // 2nd network execution with updated max_context_len
    network.execute();

    // Check if broadcast shape was recalculated
    broadcast_shape = broadcast_inst->get_impl_params()->get_output_layout().get_shape();
    ASSERT_EQ(broadcast_shape, ov::Shape{8});
}
}  // update_shape_test
