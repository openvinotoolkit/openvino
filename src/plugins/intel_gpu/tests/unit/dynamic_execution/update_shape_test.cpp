// Copyright (C) 2018-2026 Intel Corporation
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
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>

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

    auto score_aggregation_window_layout = layout{ov::PartialShape{0}, data_types::i32, format::bfyx};
    auto score_aggregation_window_mem = engine.allocate_memory(score_aggregation_window_layout);

    auto rotated_block_indices_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto rotated_block_indices_mem = engine.allocate_memory(rotated_block_indices_layout);

    auto rotation_deltas_layout = layout{ov::PartialShape{1, 1}, data_types::i32, format::bfyx};
    auto rotation_deltas_mem = engine.allocate_memory(rotation_deltas_layout);

    auto rotation_trig_lut_layout = layout{ov::PartialShape{1, 1}, data_types::f32, format::bfyx};
    auto rotation_trig_lut_mem = engine.allocate_memory(rotation_trig_lut_layout);

    auto xattention_threshold_layout = layout{ov::PartialShape{1}, data_types::f32, format::bfyx};
    auto xattention_threshold_mem = engine.allocate_memory(xattention_threshold_layout);

    auto xattention_block_size_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto xattention_block_size_mem = engine.allocate_memory(xattention_block_size_layout);

    auto xattention_stride_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};;
    auto xattention_stride_mem = engine.allocate_memory(xattention_stride_layout);

    auto sinks_layout = layout{ov::PartialShape{0, 0, 0, 0}, data_types::f32, format::bfyx};;
    auto sinks_mem = engine.allocate_memory(sinks_layout);

    auto adaptive_rkv_start_size_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto adaptive_rkv_start_size_mem = engine.allocate_memory(adaptive_rkv_start_size_layout);

    auto adaptive_rkv_evictable_sizes_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_evictable_sizes_mem = engine.allocate_memory(adaptive_rkv_evictable_sizes_layout);

    auto adaptive_rkv_diversity_block_set_indices_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_diversity_block_set_indices_mem = engine.allocate_memory(adaptive_rkv_diversity_block_set_indices_layout);

    auto adaptive_rkv_diversity_block_set_indices_begins_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_diversity_block_set_indices_begins_mem = engine.allocate_memory(adaptive_rkv_diversity_block_set_indices_begins_layout);

    auto token_type_ids_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto token_type_ids_mem = engine.allocate_memory(token_type_ids_layout);
    set_values(token_type_ids_mem, {0});
    auto qq_bias_layout = layout{ov::PartialShape{16}, data_types::u8, format::bfyx};
    auto qq_bias_mem = engine.allocate_memory(qq_bias_layout);
    auto qq_bias_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto qq_bias_begins_mem = engine.allocate_memory(qq_bias_begins_layout);

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
                                         input_info("max_context_len"),
                                         input_info("score_aggregation_window"),
                                         input_info("rotated_block_indices"),
                                         input_info("rotation_deltas"),
                                         input_info("rotation_trig_lut"),
                                         input_info("xattention_threshold"),
                                         input_info("xattention_block_size"),
                                         input_info("xattention_stride"),
                                         input_info("sinks"),
                                         input_info("adaptive_rkv_start_size"),
                                         input_info("adaptive_rkv_evictable_sizes"),
                                         input_info("adaptive_rkv_diversity_block_set_indices"),
                                         input_info("adaptive_rkv_diversity_block_set_indices_begins"),
                                         input_info("token_type_ids"),
                                         input_info("qq_bias"),
                                         input_info("qq_bias_begins")
    };

    auto pa_prim = paged_attention("paged_attention", pa_inputs);
    pa_prim.k_head_size = 64;
    pa_prim.v_head_size = 64;
    pa_prim.kv_heads_num = 2;
    pa_prim.heads_num = 2;
    pa_prim.scale_val = 1.f;
    pa_prim.has_alibi = false;
    pa_prim.num_outputs = 1;
    pa_prim.has_rotated_blocks = false;
    pa_prim.is_key_by_channel = true;

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
    topology.add(input_layout("score_aggregation_window", score_aggregation_window_layout));
    topology.add(input_layout("rotated_block_indices", rotated_block_indices_layout));
    topology.add(input_layout("rotation_deltas", rotation_deltas_layout));
    topology.add(input_layout("rotation_trig_lut", rotation_trig_lut_layout));
    topology.add(input_layout("xattention_threshold", xattention_threshold_layout));
    topology.add(input_layout("xattention_block_size", xattention_block_size_layout));
    topology.add(input_layout("xattention_stride", xattention_stride_layout));
    topology.add(input_layout("sinks", sinks_layout));
    topology.add(input_layout("adaptive_rkv_start_size", adaptive_rkv_start_size_layout));
    topology.add(input_layout("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_layout));
    topology.add(input_layout("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_layout));
    topology.add(input_layout("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_layout));
    topology.add(input_layout("token_type_ids", token_type_ids_layout));
    topology.add(input_layout("qq_bias", qq_bias_layout));
    topology.add(input_layout("qq_bias_begins", qq_bias_begins_layout));
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
    network.set_input_data("score_aggregation_window", score_aggregation_window_mem);
    network.set_input_data("rotated_block_indices", rotated_block_indices_mem);
    network.set_input_data("rotation_deltas", rotation_deltas_mem);
    network.set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
    network.set_input_data("xattention_threshold", xattention_threshold_mem);
    network.set_input_data("xattention_block_size", xattention_block_size_mem);
    network.set_input_data("xattention_stride", xattention_stride_mem);
    network.set_input_data("sinks", sinks_mem);
    network.set_input_data("adaptive_rkv_start_size", adaptive_rkv_start_size_mem);
    network.set_input_data("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem);
    network.set_input_data("token_type_ids", token_type_ids_mem);
    network.set_input_data("qq_bias", qq_bias_mem);
    network.set_input_data("qq_bias_begins", qq_bias_begins_mem);

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
    network.set_input_data("score_aggregation_window", score_aggregation_window_mem);
    network.set_input_data("rotated_block_indices", rotated_block_indices_mem);
    network.set_input_data("rotation_deltas", rotation_deltas_mem);
    network.set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
    network.set_input_data("xattention_threshold", xattention_threshold_mem);
    network.set_input_data("xattention_block_size", xattention_block_size_mem);
    network.set_input_data("xattention_stride", xattention_stride_mem);
    network.set_input_data("sinks", sinks_mem);
    network.set_input_data("adaptive_rkv_start_size", adaptive_rkv_start_size_mem);
    network.set_input_data("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem);
    network.set_input_data("token_type_ids", token_type_ids_mem);
    network.set_input_data("qq_bias", qq_bias_mem);
    network.set_input_data("qq_bias_begins", qq_bias_begins_mem);

    // Update max_context_len value, which should be taken into account in shape recalculation for broadcast
    set_values(max_context_len_mem, {8});

    network.set_input_data("max_context_len", max_context_len_mem);

    // 2nd network execution with updated max_context_len
    network.execute();

    // Check if broadcast shape was recalculated
    broadcast_shape = broadcast_inst->get_impl_params()->get_output_layout().get_shape();
    ASSERT_EQ(broadcast_shape, ov::Shape{8});
}

TEST(update_shape_test, paged_attention_mixed_stage_token_type_ids_buffer_layout) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    auto qkv_mem_layout = layout{ov::PartialShape{9, 128}, data_types::f16, format::bfyx};
    auto qkv_mem = engine.allocate_memory(qkv_mem_layout);
    auto qkv_rnd = rg.generate_random_1d<ov::float16>(qkv_mem_layout.count(), 0, 10);
    set_values(qkv_mem, qkv_rnd);

    auto key_cache_mem_layout = layout{ov::PartialShape{2, 2, 64, 16}, data_types::f16, format::bfyx};
    auto value_cache_mem_layout = layout{ov::PartialShape{2, 2, 16, 64}, data_types::f16, format::bfyx};
    auto key_cache_mem = engine.allocate_memory(key_cache_mem_layout);
    auto value_cache_mem = engine.allocate_memory(value_cache_mem_layout);
    auto cache_rnd = rg.generate_random_1d<ov::float16>(key_cache_mem_layout.count(), 0, 10);
    set_values(key_cache_mem, cache_rnd);
    set_values(value_cache_mem, cache_rnd);

    auto past_lens_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto past_lens_mem = engine.allocate_memory(past_lens_mem_layout);
    set_values(past_lens_mem, {8});

    auto subsequence_begins_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto subsequence_begins_mem = engine.allocate_memory(subsequence_begins_mem_layout);
    set_values(subsequence_begins_mem, {0, 9});

    auto block_indices_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_indices_mem = engine.allocate_memory(block_indices_mem_layout);
    set_values(block_indices_mem, {0, 1});

    auto block_indices_begins_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_mem_layout);
    set_values(block_indices_begins_mem, {0, 2});

    auto scale_mem_layout = layout{ov::PartialShape{1}, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_mem_layout);
    set_values<ov::float16>(scale_mem, {1});

    auto sliding_window_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto sliding_window_mem = engine.allocate_memory(sliding_window_mem_layout);
    set_values(sliding_window_mem, {0});

    auto alibi_mem_layout = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};
    auto alibi_mem = engine.allocate_memory(alibi_mem_layout);

    auto max_context_len_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto max_context_len_mem = engine.allocate_memory(max_context_len_mem_layout);
    set_values(max_context_len_mem, {17});

    auto score_aggregation_window_layout = layout{ov::PartialShape{0}, data_types::i32, format::bfyx};
    auto score_aggregation_window_mem = engine.allocate_memory(score_aggregation_window_layout);

    auto rotated_block_indices_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto rotated_block_indices_mem = engine.allocate_memory(rotated_block_indices_layout);

    auto rotation_deltas_layout = layout{ov::PartialShape{1, 1}, data_types::i32, format::bfyx};
    auto rotation_deltas_mem = engine.allocate_memory(rotation_deltas_layout);

    auto rotation_trig_lut_layout = layout{ov::PartialShape{1, 1}, data_types::f32, format::bfyx};
    auto rotation_trig_lut_mem = engine.allocate_memory(rotation_trig_lut_layout);

    auto xattention_threshold_layout = layout{ov::PartialShape{1}, data_types::f32, format::bfyx};
    auto xattention_threshold_mem = engine.allocate_memory(xattention_threshold_layout);

    auto xattention_block_size_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto xattention_block_size_mem = engine.allocate_memory(xattention_block_size_layout);

    auto xattention_stride_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto xattention_stride_mem = engine.allocate_memory(xattention_stride_layout);

    auto sinks_layout = layout{ov::PartialShape{0, 0, 0, 0}, data_types::f32, format::bfyx};
    auto sinks_mem = engine.allocate_memory(sinks_layout);

    auto adaptive_rkv_start_size_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto adaptive_rkv_start_size_mem = engine.allocate_memory(adaptive_rkv_start_size_layout);

    auto adaptive_rkv_evictable_sizes_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_evictable_sizes_mem = engine.allocate_memory(adaptive_rkv_evictable_sizes_layout);

    auto adaptive_rkv_diversity_block_set_indices_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_diversity_block_set_indices_mem = engine.allocate_memory(adaptive_rkv_diversity_block_set_indices_layout);

    auto adaptive_rkv_diversity_block_set_indices_begins_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_diversity_block_set_indices_begins_mem = engine.allocate_memory(adaptive_rkv_diversity_block_set_indices_begins_layout);

    auto token_type_ids_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto token_type_ids_mem = engine.allocate_memory(token_type_ids_layout);
    set_values(token_type_ids_mem, {1});

    auto qq_bias_layout = layout{ov::PartialShape{16}, data_types::u8, format::bfyx};
    auto qq_bias_mem = engine.allocate_memory(qq_bias_layout);
    auto qq_bias_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto qq_bias_begins_mem = engine.allocate_memory(qq_bias_begins_layout);

    auto query_layout = layout{ov::PartialShape{-1, 128}, data_types::f16, format::bfyx};
    auto key_layout = query_layout;
    auto value_layout = query_layout;
    auto key_cache_layout = layout{ov::PartialShape{-1, 2, 64, 16}, data_types::f16, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{-1, 2, 16, 64}, data_types::f16, format::bfyx};
    auto dynamic_i32_layout = layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx};
    auto past_lens_layout = dynamic_i32_layout;
    auto subsequence_begins_layout = dynamic_i32_layout;
    auto block_indices_layout = dynamic_i32_layout;
    auto block_indices_begins_layout = dynamic_i32_layout;
    auto score_aggregation_layout = dynamic_i32_layout;
    auto token_type_ids_input_layout = dynamic_i32_layout;

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
                                         input_info("max_context_len"),
                                         input_info("score_aggregation_window"),
                                         input_info("rotated_block_indices"),
                                         input_info("rotation_deltas"),
                                         input_info("rotation_trig_lut"),
                                         input_info("xattention_threshold"),
                                         input_info("xattention_block_size"),
                                         input_info("xattention_stride"),
                                         input_info("sinks"),
                                         input_info("adaptive_rkv_start_size"),
                                         input_info("adaptive_rkv_evictable_sizes"),
                                         input_info("adaptive_rkv_diversity_block_set_indices"),
                                         input_info("adaptive_rkv_diversity_block_set_indices_begins"),
                                         input_info("token_type_ids"),
                                         input_info("qq_bias"),
                                         input_info("qq_bias_begins")};

    auto pa_prim = paged_attention("paged_attention", pa_inputs);
    pa_prim.k_head_size = 64;
    pa_prim.v_head_size = 64;
    pa_prim.kv_heads_num = 2;
    pa_prim.heads_num = 2;
    pa_prim.scale_val = 1.f;
    pa_prim.has_alibi = false;
    pa_prim.num_outputs = 1;
    pa_prim.has_rotated_blocks = false;
    pa_prim.has_token_type_ids = true;
    pa_prim.is_key_by_channel = true;

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
    topology.add(input_layout("scale", scale_mem_layout));
    topology.add(input_layout("sliding_window", sliding_window_mem_layout));
    topology.add(input_layout("alibi", alibi_mem_layout));
    topology.add(input_layout("max_context_len", max_context_len_mem_layout));
    topology.add(input_layout("score_aggregation_window", score_aggregation_layout));
    topology.add(input_layout("rotated_block_indices", rotated_block_indices_layout));
    topology.add(input_layout("rotation_deltas", rotation_deltas_layout));
    topology.add(input_layout("rotation_trig_lut", rotation_trig_lut_layout));
    topology.add(input_layout("xattention_threshold", xattention_threshold_layout));
    topology.add(input_layout("xattention_block_size", xattention_block_size_layout));
    topology.add(input_layout("xattention_stride", xattention_stride_layout));
    topology.add(input_layout("sinks", sinks_layout));
    topology.add(input_layout("adaptive_rkv_start_size", adaptive_rkv_start_size_layout));
    topology.add(input_layout("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_layout));
    topology.add(input_layout("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_layout));
    topology.add(input_layout("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_layout));
    topology.add(input_layout("token_type_ids", token_type_ids_input_layout));
    topology.add(input_layout("qq_bias", qq_bias_layout));
    topology.add(input_layout("qq_bias_begins", qq_bias_begins_layout));
    topology.add(pa_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"paged_attention", {format::any, "paged_attention_opt", impl_types::ocl}}}));
    network network(engine, topology, config);

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
    network.set_input_data("max_context_len", max_context_len_mem);
    network.set_input_data("score_aggregation_window", score_aggregation_window_mem);
    network.set_input_data("rotated_block_indices", rotated_block_indices_mem);
    network.set_input_data("rotation_deltas", rotation_deltas_mem);
    network.set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
    network.set_input_data("xattention_threshold", xattention_threshold_mem);
    network.set_input_data("xattention_block_size", xattention_block_size_mem);
    network.set_input_data("xattention_stride", xattention_stride_mem);
    network.set_input_data("sinks", sinks_mem);
    network.set_input_data("adaptive_rkv_start_size", adaptive_rkv_start_size_mem);
    network.set_input_data("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem);
    network.set_input_data("token_type_ids", token_type_ids_mem);
    network.set_input_data("qq_bias", qq_bias_mem);
    network.set_input_data("qq_bias_begins", qq_bias_begins_mem);

    OV_ASSERT_NO_THROW(network.execute());

    auto pa_inst = network.get_primitive("paged_attention");
    const auto& intermediate_mems = pa_inst->get_intermediates_memories();

    // Allocation-time and execution-time micro/non-micro decisions must agree; a mismatch shows up
    // as the wrong buffer count. token_type_ids no longer forces the non-micro path in MIXED.
    // The micro path has one extra buffer for the token_type_ids group bounds.
    const bool micro_layout = engine.get_device_info().supports_immad;
    ASSERT_EQ(intermediate_mems.size(), micro_layout ? 5u : 7u);
}
TEST(update_shape_test, paged_attention_mixed_stage_h256_micro_qtile64_buffers) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    // MIXED stage on the micro path: 2 subsequences with q_len 100 / past_len 300 each, head size 256, GQA 8/2.
    // q_len > small-q tile (16) keeps the regular MIXED micro kernel; 100 is not a multiple of the q tile 64.
    auto qkv_mem_layout = layout{ov::PartialShape{200, 2048}, data_types::f16, format::bfyx};
    auto qkv_mem = engine.allocate_memory(qkv_mem_layout);
    auto qkv_rnd = rg.generate_random_1d<ov::float16>(qkv_mem_layout.count(), 0, 10);
    set_values(qkv_mem, qkv_rnd);

    auto key_cache_mem_layout = layout{ov::PartialShape{50, 2, 256, 16}, data_types::f16, format::bfyx};
    auto value_cache_mem_layout = layout{ov::PartialShape{50, 2, 16, 256}, data_types::f16, format::bfyx};
    auto key_cache_mem = engine.allocate_memory(key_cache_mem_layout);
    auto value_cache_mem = engine.allocate_memory(value_cache_mem_layout);
    auto cache_rnd = rg.generate_random_1d<ov::float16>(key_cache_mem_layout.count(), 0, 10);
    set_values(key_cache_mem, cache_rnd);
    set_values(value_cache_mem, cache_rnd);

    auto past_lens_mem_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto past_lens_mem = engine.allocate_memory(past_lens_mem_layout);
    set_values(past_lens_mem, {300, 300});

    auto subsequence_begins_mem_layout = layout{ov::PartialShape{3}, data_types::i32, format::bfyx};
    auto subsequence_begins_mem = engine.allocate_memory(subsequence_begins_mem_layout);
    set_values(subsequence_begins_mem, {0, 100, 200});

    // 25 blocks of 16 tokens per subsequence (past_len + q_len == 400 tokens), 50 physical blocks total
    auto block_indices_mem_layout = layout{ov::PartialShape{50}, data_types::i32, format::bfyx};
    auto block_indices_mem = engine.allocate_memory(block_indices_mem_layout);
    std::vector<int> block_indices_vals;
    for (int b = 0; b < 50; ++b) {
        block_indices_vals.push_back(b);
    }
    set_values(block_indices_mem, block_indices_vals);

    auto block_indices_begins_mem_layout = layout{ov::PartialShape{3}, data_types::i32, format::bfyx};
    auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_mem_layout);
    set_values(block_indices_begins_mem, {0, 25, 50});

    auto scale_mem_layout = layout{ov::PartialShape{1}, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_mem_layout);
    set_values<ov::float16>(scale_mem, {1});

    auto sliding_window_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto sliding_window_mem = engine.allocate_memory(sliding_window_mem_layout);
    set_values(sliding_window_mem, {0});

    auto alibi_mem_layout = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};
    auto alibi_mem = engine.allocate_memory(alibi_mem_layout);

    auto max_context_len_mem_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto max_context_len_mem = engine.allocate_memory(max_context_len_mem_layout);
    set_values(max_context_len_mem, {400});

    auto score_aggregation_window_layout = layout{ov::PartialShape{0}, data_types::i32, format::bfyx};
    auto score_aggregation_window_mem = engine.allocate_memory(score_aggregation_window_layout);

    auto rotated_block_indices_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto rotated_block_indices_mem = engine.allocate_memory(rotated_block_indices_layout);

    auto rotation_deltas_layout = layout{ov::PartialShape{1, 1}, data_types::i32, format::bfyx};
    auto rotation_deltas_mem = engine.allocate_memory(rotation_deltas_layout);

    auto rotation_trig_lut_layout = layout{ov::PartialShape{1, 1}, data_types::f32, format::bfyx};
    auto rotation_trig_lut_mem = engine.allocate_memory(rotation_trig_lut_layout);

    auto xattention_threshold_layout = layout{ov::PartialShape{1}, data_types::f32, format::bfyx};
    auto xattention_threshold_mem = engine.allocate_memory(xattention_threshold_layout);

    auto xattention_block_size_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto xattention_block_size_mem = engine.allocate_memory(xattention_block_size_layout);

    auto xattention_stride_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto xattention_stride_mem = engine.allocate_memory(xattention_stride_layout);

    auto sinks_layout = layout{ov::PartialShape{0, 0, 0, 0}, data_types::f32, format::bfyx};
    auto sinks_mem = engine.allocate_memory(sinks_layout);

    auto adaptive_rkv_start_size_layout = layout{ov::PartialShape{}, data_types::i32, format::bfyx};
    auto adaptive_rkv_start_size_mem = engine.allocate_memory(adaptive_rkv_start_size_layout);

    auto adaptive_rkv_evictable_sizes_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_evictable_sizes_mem = engine.allocate_memory(adaptive_rkv_evictable_sizes_layout);

    auto adaptive_rkv_diversity_block_set_indices_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_diversity_block_set_indices_mem = engine.allocate_memory(adaptive_rkv_diversity_block_set_indices_layout);

    auto adaptive_rkv_diversity_block_set_indices_begins_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto adaptive_rkv_diversity_block_set_indices_begins_mem = engine.allocate_memory(adaptive_rkv_diversity_block_set_indices_begins_layout);

    auto token_type_ids_layout = layout{ov::PartialShape{1}, data_types::i32, format::bfyx};
    auto token_type_ids_mem = engine.allocate_memory(token_type_ids_layout);
    set_values(token_type_ids_mem, {1});

    auto qq_bias_layout = layout{ov::PartialShape{16}, data_types::u8, format::bfyx};
    auto qq_bias_mem = engine.allocate_memory(qq_bias_layout);
    auto qq_bias_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto qq_bias_begins_mem = engine.allocate_memory(qq_bias_begins_layout);

    auto query_layout = layout{ov::PartialShape{-1, 2048}, data_types::f16, format::bfyx};
    auto key_layout = query_layout;
    auto value_layout = query_layout;
    auto key_cache_layout = layout{ov::PartialShape{-1, 2, 256, 16}, data_types::f16, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{-1, 2, 16, 256}, data_types::f16, format::bfyx};
    auto dynamic_i32_layout = layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx};
    auto past_lens_layout = dynamic_i32_layout;
    auto subsequence_begins_layout = dynamic_i32_layout;
    auto block_indices_layout = dynamic_i32_layout;
    auto block_indices_begins_layout = dynamic_i32_layout;
    auto score_aggregation_layout = dynamic_i32_layout;
    auto token_type_ids_input_layout = dynamic_i32_layout;

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
                                         input_info("max_context_len"),
                                         input_info("score_aggregation_window"),
                                         input_info("rotated_block_indices"),
                                         input_info("rotation_deltas"),
                                         input_info("rotation_trig_lut"),
                                         input_info("xattention_threshold"),
                                         input_info("xattention_block_size"),
                                         input_info("xattention_stride"),
                                         input_info("sinks"),
                                         input_info("adaptive_rkv_start_size"),
                                         input_info("adaptive_rkv_evictable_sizes"),
                                         input_info("adaptive_rkv_diversity_block_set_indices"),
                                         input_info("adaptive_rkv_diversity_block_set_indices_begins"),
                                         input_info("token_type_ids"),
                                         input_info("qq_bias"),
                                         input_info("qq_bias_begins")};

    auto pa_prim = paged_attention("paged_attention", pa_inputs);
    pa_prim.k_head_size = 256;
    pa_prim.v_head_size = 256;
    pa_prim.kv_heads_num = 2;
    pa_prim.heads_num = 8;
    pa_prim.scale_val = 1.f;
    pa_prim.has_alibi = false;
    pa_prim.num_outputs = 1;
    pa_prim.has_rotated_blocks = false;
    pa_prim.has_token_type_ids = true;
    pa_prim.is_key_by_channel = true;

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
    topology.add(input_layout("scale", scale_mem_layout));
    topology.add(input_layout("sliding_window", sliding_window_mem_layout));
    topology.add(input_layout("alibi", alibi_mem_layout));
    topology.add(input_layout("max_context_len", max_context_len_mem_layout));
    topology.add(input_layout("score_aggregation_window", score_aggregation_layout));
    topology.add(input_layout("rotated_block_indices", rotated_block_indices_layout));
    topology.add(input_layout("rotation_deltas", rotation_deltas_layout));
    topology.add(input_layout("rotation_trig_lut", rotation_trig_lut_layout));
    topology.add(input_layout("xattention_threshold", xattention_threshold_layout));
    topology.add(input_layout("xattention_block_size", xattention_block_size_layout));
    topology.add(input_layout("xattention_stride", xattention_stride_layout));
    topology.add(input_layout("sinks", sinks_layout));
    topology.add(input_layout("adaptive_rkv_start_size", adaptive_rkv_start_size_layout));
    topology.add(input_layout("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_layout));
    topology.add(input_layout("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_layout));
    topology.add(input_layout("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_layout));
    topology.add(input_layout("token_type_ids", token_type_ids_input_layout));
    topology.add(input_layout("qq_bias", qq_bias_layout));
    topology.add(input_layout("qq_bias_begins", qq_bias_begins_layout));
    topology.add(pa_prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(
        ov::intel_gpu::ImplForcingMap{{"paged_attention", {format::any, "paged_attention_opt", impl_types::ocl}}}));
    network network(engine, topology, config);

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
    network.set_input_data("max_context_len", max_context_len_mem);
    network.set_input_data("score_aggregation_window", score_aggregation_window_mem);
    network.set_input_data("rotated_block_indices", rotated_block_indices_mem);
    network.set_input_data("rotation_deltas", rotation_deltas_mem);
    network.set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
    network.set_input_data("xattention_threshold", xattention_threshold_mem);
    network.set_input_data("xattention_block_size", xattention_block_size_mem);
    network.set_input_data("xattention_stride", xattention_stride_mem);
    network.set_input_data("sinks", sinks_mem);
    network.set_input_data("adaptive_rkv_start_size", adaptive_rkv_start_size_mem);
    network.set_input_data("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem);
    network.set_input_data("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem);
    network.set_input_data("token_type_ids", token_type_ids_mem);
    network.set_input_data("qq_bias", qq_bias_mem);
    network.set_input_data("qq_bias_begins", qq_bias_begins_mem);

    OV_ASSERT_NO_THROW(network.execute());

    auto pa_inst = network.get_primitive("paged_attention");
    const auto& intermediate_mems = pa_inst->get_intermediates_memories();

    const auto& device_info = engine.get_device_info();
    const bool xe2_xe3_immad = device_info.supports_immad && (device_info.arch == gpu_arch::xe2 || device_info.arch == gpu_arch::xe3);
    if (!xe2_xe3_immad) {
        GTEST_SKIP() << "xe2_h256_pa_mixed (q tile 64) is selected on Xe2/Xe3 with DPAS only, arch = "
                     << static_cast<int>(device_info.arch) << ", supports_immad = " << device_info.supports_immad;
    }

    auto dump_sizes = [&]() {
        std::string res;
        for (size_t i = 0; i < intermediate_mems.size(); ++i) {
            res += "[" + std::to_string(i) + "]=" + std::to_string(intermediate_mems[i]->size()) + " ";
        }
        return res;
    };

    ASSERT_GE(intermediate_mems.size(), 4u) << dump_sizes();

    // get_internal_buffer_descs() sizes the internal buffers with block_size = get_query_block_size(...),
    // i.e. the MIXED micro kernel query tile (wg_tile_n of pa_sdpa_micro_mixed):
    //     aligned_seq_len = block * ceil_div(q_len - (block - past_len % block), block) + block   (past_len % block != 0)
    // For head size 256 on Xe2/Xe3 f96fb1dc55 selects xe2_h256_pa_mixed (q tile 64) instead of the generic
    // h256 PA config (q tile 32), which grows aligned_seq_len (and hence the buffers sized from it).
    const auto aligned_seq_len = [](int64_t q_len, int64_t past_len, int64_t block) {
        const int64_t occupied_slots = past_len % block;
        int64_t len = q_len;
        int64_t aligned = 0;
        if (past_len != 0 && len + occupied_slots > block) {
            aligned += block;
            len -= block - occupied_slots;
        }
        aligned += ((len + block - 1) / block) * block;
        return aligned;
    };
    const size_t aligned_qtile64 = 2 * static_cast<size_t>(aligned_seq_len(100, 300, 64));  // 384
    const size_t aligned_qtile32 = 2 * static_cast<size_t>(aligned_seq_len(100, 300, 32));   // 256

    const size_t qtile64_block_idx_bytes = ((aligned_qtile64 + 7) / 8) * 2 * 4;  // 384
    const size_t qtile32_block_idx_bytes = ((aligned_qtile32 + 7) / 8) * 2 * 4;  // 256
    ASSERT_EQ(intermediate_mems[3]->size(), qtile64_block_idx_bytes)
        << "expected the h256 PA MIXED micro config with q tile 64 (xe2_h256_pa_mixed, aligned_seq_len "
        << aligned_qtile64 << "), sizes: " << dump_sizes();
    EXPECT_NE(intermediate_mems[3]->size(), qtile32_block_idx_bytes)
        << "size matches the generic h256 PA config with q tile 32 (aligned_seq_len " << aligned_qtile32
        << "): " << dump_sizes();
    EXPECT_EQ(intermediate_mems[0]->size(), ((aligned_qtile64 + 15) / 16) * 4) << dump_sizes();
    // [4]: token_type_ids group bounds for the micro kernel, two int32 values per token
    ASSERT_EQ(intermediate_mems.size(), 5u) << dump_sizes();
    ASSERT_EQ(intermediate_mems[4]->size(), 200u * 2 * sizeof(int32_t)) << dump_sizes();
}

TEST(update_shape_test, fc_fused_eltwise_input_from_optimized_reshape) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    const int64_t K = 16;
    const int64_t N = 8;
    tests::random_generator rg(GET_SUITE_NAME);
    auto gate_w_data = rg.generate_random_1d<ov::float16>(N * K, -1, 1);
    auto side_w_data = rg.generate_random_1d<ov::float16>(N * K, -1, 1);
    auto gate_w = engine.allocate_memory(layout{ov::PartialShape{N, K}, data_types::f16, format::bfyx});
    auto side_w = engine.allocate_memory(layout{ov::PartialShape{N, K}, data_types::f16, format::bfyx});
    set_values(gate_w, gate_w_data);
    set_values(side_w, side_w_data);

    topology topology(input_layout("input", layout{ov::PartialShape{-1, 1, K}, data_types::f16, format::bfyx}),
                      data("gate_w", gate_w),
                      data("side_w", side_w),
                      activation("act", input_info("input"), activation_func::relu),
                      fully_connected("gate", input_info("act"), "gate_w", "", 3),
                      reshape("act_2d", input_info("act"), true, std::vector<int64_t>{-1, K}, ov::PartialShape{-1, K}),
                      fully_connected("side", input_info("act_2d"), "side_w", "", 2),
                      reshape("side_3d", input_info("side"), true, std::vector<int64_t>{-1, 1, N}, ov::PartialShape{-1, 1, N}),
                      eltwise("prod", {input_info("gate"), input_info("side_3d")}, eltwise_mode::prod),
                      reorder("output", input_info("prod"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    const auto& fc_node = network.get_primitive("output")->get_node();
    ASSERT_TRUE(fc_node.is_type<fully_connected>());
    ASSERT_TRUE(std::any_of(fc_node.get_fused_primitives().begin(), fc_node.get_fused_primitives().end(), [](const fused_primitive_desc& fd) {
        return fd.is_type<eltwise>();
    })) << "eltwise prod is expected to be fused into gate";
    ASSERT_TRUE(network.get_primitive("side_3d")->get_node().can_be_optimized());

    for (const int64_t batch : {7, 1, 9}) {
        auto input_data = rg.generate_random_1d<ov::float16>(batch * K, -1, 1);
        auto input_mem = engine.allocate_memory(layout{ov::PartialShape{batch, 1, K}, data_types::f16, format::bfyx});
        set_values(input_mem, input_data);
        network.set_input_data("input", input_mem);

        std::map<primitive_id, network_output> outputs;
        OV_ASSERT_NO_THROW(outputs = network.execute());
        ASSERT_TRUE(network.get_primitive("output")->get_impl()->is_onednn());

        ASSERT_EQ(outputs.at("output").get_layout().get_partial_shape(), (ov::PartialShape{batch, 1, N}));
        auto output_mem = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(output_mem, get_test_stream());
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                float gate = 0.f;
                float side = 0.f;
                for (int64_t k = 0; k < K; ++k) {
                    const float act = std::max(0.f, static_cast<float>(input_data[b * K + k]));
                    gate += act * static_cast<float>(gate_w_data[n * K + k]);
                    side += act * static_cast<float>(side_w_data[n * K + k]);
                }
                ASSERT_NEAR(output_ptr[b * N + n], gate * side, 5e-2f) << "batch " << batch << ", b " << b << ", n " << n;
            }
        }
    }
}

}  // update_shape_test
