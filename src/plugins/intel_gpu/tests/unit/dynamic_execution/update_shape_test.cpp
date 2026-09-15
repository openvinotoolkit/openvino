// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>
#include <intel_gpu/primitives/select.hpp>
#include <intel_gpu/primitives/shape_of.hpp>
#include <intel_gpu/primitives/strided_slice.hpp>
#include <intel_gpu/primitives/tile.hpp>

#include "program_wrapper.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace update_shape_tests {
TEST(update_shape_test, boolean_broadcast_in_shapeof_subgraph) {
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory(layout{ov::PartialShape{2, 3}, data_types::f32, format::bfyx});
    auto indices_mem = engine.allocate_memory(layout{ov::PartialShape{2}, data_types::i32, format::bfyx});
    auto value_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::boolean, format::bfyx});
    set_values<int32_t>(indices_mem, {0, 1});
    set_values<uint8_t>(value_mem, {1});

    topology topology(input_layout("input", layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}),
                      data("indices", indices_mem),
                      data("value", value_mem),
                      shape_of("shape_of", input_info("input"), data_types::i64),
                      gather("target_shape", input_info("shape_of"), input_info("indices"), 0, 1, ov::Shape{2}),
                      broadcast("broadcast", input_info("value"), input_info("target_shape"), {}, ov::op::BroadcastType::NUMPY));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto output = network.execute().at("broadcast").get_memory();

    ASSERT_TRUE(network.get_program()->get_node("broadcast").is_in_shape_of_subgraph());
    ASSERT_EQ(output->get_layout().get_shape(), ov::Shape({2, 3}));
    mem_lock<uint8_t, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<uint8_t>(output_ptr.begin(), output_ptr.end()), std::vector<uint8_t>(6, 1));
}

TEST(update_shape_test, boolean_concat_in_shapeof_subgraph) {
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory(layout{ov::PartialShape{2, 3}, data_types::f32, format::bfyx});
    auto indices_mem = engine.allocate_memory(layout{ov::PartialShape{2}, data_types::i32, format::bfyx});
    auto value_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::boolean, format::bfyx});
    set_values<int32_t>(indices_mem, {0, 1});
    set_values<uint8_t>(value_mem, {1});

    topology topology(input_layout("input", layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}),
                      data("indices", indices_mem),
                      data("value", value_mem),
                      shape_of("shape_of", input_info("input"), data_types::i64),
                      gather("target_shape", input_info("shape_of"), input_info("indices"), 0, 1, ov::Shape{2}),
                      broadcast("broadcast", input_info("value"), input_info("target_shape"), {}, ov::op::BroadcastType::NUMPY),
                      concatenation("concat", {input_info("broadcast"), input_info("broadcast")}, 1));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto output = network.execute().at("concat").get_memory();

    ASSERT_TRUE(network.get_program()->get_node("concat").is_in_shape_of_subgraph());
    ASSERT_EQ(output->get_layout().get_shape(), ov::Shape({2, 6}));
    mem_lock<uint8_t, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<uint8_t>(output_ptr.begin(), output_ptr.end()), std::vector<uint8_t>(12, 1));
}

TEST(update_shape_test, boolean_activation_in_shapeof_subgraph) {
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory(layout{ov::PartialShape{2, 3}, data_types::f32, format::bfyx});
    auto indices_mem = engine.allocate_memory(layout{ov::PartialShape{2}, data_types::i32, format::bfyx});
    auto value_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::boolean, format::bfyx});
    set_values<int32_t>(indices_mem, {0, 1});
    set_values<uint8_t>(value_mem, {1});

    topology topology(input_layout("input", layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}),
                      data("indices", indices_mem),
                      data("value", value_mem),
                      shape_of("shape_of", input_info("input"), data_types::i64),
                      gather("target_shape", input_info("shape_of"), input_info("indices"), 0, 1, ov::Shape{2}),
                      broadcast("broadcast", input_info("value"), input_info("target_shape"), {}, ov::op::BroadcastType::NUMPY),
                      activation("logical_not", input_info("broadcast"), activation_func::negation));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto output = network.execute().at("logical_not").get_memory();

    ASSERT_TRUE(network.get_program()->get_node("logical_not").is_in_shape_of_subgraph());
    ASSERT_EQ(output->get_layout().get_shape(), ov::Shape({2, 3}));
    mem_lock<uint8_t, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<uint8_t>(output_ptr.begin(), output_ptr.end()), std::vector<uint8_t>(6, 0));
}

TEST(update_shape_test, boolean_cpu_ops_in_shapeof_subgraph) {
    auto& engine = get_test_engine();

    auto input_mem = engine.allocate_memory(layout{ov::PartialShape{2, 3}, data_types::f32, format::bfyx});
    auto shape_indices_mem = engine.allocate_memory(layout{ov::PartialShape{2}, data_types::i32, format::bfyx});
    auto value_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::boolean, format::bfyx});
    auto false_value_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::boolean, format::bfyx});
    auto numeric_true_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::i64, format::bfyx});
    auto numeric_false_mem = engine.allocate_memory(layout{ov::PartialShape{}, data_types::i64, format::bfyx});
    auto scatter_indices_mem = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::i32, format::bfyx});
    auto scatter_updates_mem = engine.allocate_memory(layout{ov::PartialShape{2, 1}, data_types::boolean, format::bfyx});
    set_values<int32_t>(shape_indices_mem, {0, 1});
    set_values<uint8_t>(value_mem, {1});
    set_values<uint8_t>(false_value_mem, {0});
    set_values<int64_t>(numeric_true_mem, {7});
    set_values<int64_t>(numeric_false_mem, {9});
    set_values<int32_t>(scatter_indices_mem, {1});
    set_values<uint8_t>(scatter_updates_mem, {0, 0});

    topology topology(input_layout("input", layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}),
                      data("shape_indices", shape_indices_mem),
                      data("value", value_mem),
                      data("false_value", false_value_mem),
                      data("numeric_true", numeric_true_mem),
                      data("numeric_false", numeric_false_mem),
                      data("scatter_indices", scatter_indices_mem),
                      data("scatter_updates", scatter_updates_mem),
                      shape_of("shape_of", input_info("input"), data_types::i64),
                      gather("target_shape", input_info("shape_of"), input_info("shape_indices"), 0, 1, ov::Shape{2}),
                      broadcast("broadcast", input_info("value"), input_info("target_shape"), {}, ov::op::BroadcastType::NUMPY),
                      crop("crop_bool", input_info("broadcast"), tensor(batch(2), feature(2)), tensor(batch(0), feature(0))),
                      eltwise("logical_and_bool", {input_info("broadcast"), input_info("value")}, eltwise_mode::logic_and, data_types::boolean),
                      gather("gather_bool", input_info("broadcast"), input_info("shape_indices"), 0, 1, ov::Shape{2, 3}),
                      reduce("reduce_bool", input_info("broadcast"), reduce_mode::logical_and, {1}, true),
                      reorder("reorder_bool", input_info("broadcast"), layout{ov::PartialShape{2, 3}, data_types::u8, format::bfyx}),
                      cldnn::select("select_bool", input_info("broadcast"), input_info("broadcast"), input_info("false_value")),
                      cldnn::select("select_numeric", input_info("broadcast"), input_info("numeric_true"), input_info("numeric_false")),
                      scatter_update("scatter_bool", input_info("broadcast"), input_info("scatter_indices"), input_info("scatter_updates"), 1),
                      strided_slice("slice_bool", input_info("broadcast"), {0, 0}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {2, 2}),
                      tile("tile_bool", input_info("broadcast"), std::vector<int64_t>{1, 2}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    for (const auto* id : {"crop_bool",
                           "logical_and_bool",
                           "gather_bool",
                           "reduce_bool",
                           "reorder_bool",
                           "select_bool",
                           "select_numeric",
                           "scatter_bool",
                           "slice_bool",
                           "tile_bool"}) {
        ASSERT_TRUE(network.get_program()->get_node(id).is_in_shape_of_subgraph()) << id;
    }

    ASSERT_EQ(outputs.at("crop_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 2}));
    ASSERT_EQ(outputs.at("logical_and_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(outputs.at("gather_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(outputs.at("reduce_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 1}));
    ASSERT_EQ(outputs.at("reorder_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(outputs.at("select_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(outputs.at("scatter_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(outputs.at("slice_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 2}));
    ASSERT_EQ(outputs.at("tile_bool").get_memory()->get_layout().get_shape(), ov::Shape({2, 6}));

    for (const auto* id : {"crop_bool", "logical_and_bool", "gather_bool", "reduce_bool", "reorder_bool", "select_bool", "slice_bool", "tile_bool"}) {
        mem_lock<uint8_t, mem_lock_type::read> output_ptr(outputs.at(id).get_memory(), get_test_stream());
        ASSERT_TRUE(std::all_of(output_ptr.begin(), output_ptr.end(), [](uint8_t value) {
            return value == 1;
        })) << id;
    }

    mem_lock<uint8_t, mem_lock_type::read> scatter_output(outputs.at("scatter_bool").get_memory(), get_test_stream());
    ASSERT_EQ(std::vector<uint8_t>(scatter_output.begin(), scatter_output.end()), (std::vector<uint8_t>{1, 0, 1, 1, 0, 1}));

    mem_lock<int64_t, mem_lock_type::read> select_output(outputs.at("select_numeric").get_memory(), get_test_stream());
    ASSERT_EQ(std::vector<int64_t>(select_output.begin(), select_output.end()), std::vector<int64_t>(6, 7));
}

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
    const bool micro_layout = engine.get_device_info().supports_immad;
    ASSERT_EQ(intermediate_mems.size(), micro_layout ? 4u : 7u);
}
}  // update_shape_test
