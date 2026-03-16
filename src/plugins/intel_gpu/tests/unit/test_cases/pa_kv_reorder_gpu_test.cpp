// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/pa_kv_reorder.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>

#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {

size_t key_offset(size_t block, size_t head, size_t k, size_t token, size_t kv_heads, size_t k_head_size, size_t block_size) {
    return block * kv_heads * k_head_size * block_size +
           head * k_head_size * block_size +
           k * block_size +
           token;
}

size_t value_offset(size_t block, size_t head, size_t token, size_t v, size_t kv_heads, size_t v_head_size, size_t block_size) {
    return block * kv_heads * block_size * v_head_size +
           head * block_size * v_head_size +
           token * v_head_size +
           v;
}

void fill_key_cache(memory::ptr key_cache_mem,
                    size_t blocks_num,
                    size_t kv_heads,
                    size_t k_head_size,
                    size_t block_size,
                    std::vector<ov::float16>& values) {
    values.resize(key_cache_mem->count());
    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t h = 0; h < kv_heads; h++) {
            for (size_t k = 0; k < k_head_size; k++) {
                for (size_t t = 0; t < block_size; t++) {
                    const size_t off = key_offset(b, h, k, t, kv_heads, k_head_size, block_size);
                    values[off] = ov::float16(static_cast<float>(1000 * b + 100 * h + 10 * k + t));
                }
            }
        }
    }
    set_values(key_cache_mem, values);
}

void fill_value_cache(memory::ptr value_cache_mem,
                      size_t blocks_num,
                      size_t kv_heads,
                      size_t v_head_size,
                      size_t block_size,
                      std::vector<ov::float16>& values) {
    values.resize(value_cache_mem->count());
    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t h = 0; h < kv_heads; h++) {
            for (size_t t = 0; t < block_size; t++) {
                for (size_t v = 0; v < v_head_size; v++) {
                    const size_t off = value_offset(b, h, t, v, kv_heads, v_head_size, block_size);
                    values[off] = ov::float16(static_cast<float>(1000 * b + 100 * h + 10 * t + v));
                }
            }
        }
    }
    set_values(value_cache_mem, values);
}

}  // namespace

TEST(pa_kv_reorder_gpu, copy_between_blocks_single_sequence) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 2;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 4;
    constexpr size_t v_head_size = 3;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, k_head_size, block_size}, data_types::f16, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, v_head_size}, data_types::f16, format::bfyx};
    auto block_indices_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_indices_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_update_indices_layout = layout{ov::PartialShape{4}, data_types::i32, format::bfyx};
    auto block_update_indices_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};

    auto key_cache_mem = engine.allocate_memory(key_cache_layout);
    auto value_cache_mem = engine.allocate_memory(value_cache_layout);
    auto block_indices_mem = engine.allocate_memory(block_indices_layout);
    auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_layout);
    auto block_update_indices_mem = engine.allocate_memory(block_update_indices_layout);
    auto block_update_indices_begins_mem = engine.allocate_memory(block_update_indices_begins_layout);

    std::vector<ov::float16> key_cache_ref;
    std::vector<ov::float16> value_cache_ref;
    fill_key_cache(key_cache_mem, blocks_num, kv_heads, k_head_size, block_size, key_cache_ref);
    fill_value_cache(value_cache_mem, blocks_num, kv_heads, v_head_size, block_size, value_cache_ref);

    set_values<int32_t>(block_indices_mem, {0, 1});
    set_values<int32_t>(block_indices_begins_mem, {0, 2});
    set_values<int32_t>(block_update_indices_mem, {
        0, 17,
        15, 16,
    });
    set_values<int32_t>(block_update_indices_begins_mem, {0, 2});

    topology topo;
    topo.add(input_layout("key_cache", key_cache_layout));
    topo.add(input_layout("value_cache", value_cache_layout));
    topo.add(input_layout("block_indices", block_indices_layout));
    topo.add(input_layout("block_indices_begins", block_indices_begins_layout));
    topo.add(input_layout("block_update_indices", block_update_indices_layout));
    topo.add(input_layout("block_update_indices_begins", block_update_indices_begins_layout));

    auto pa_reorder = pa_kv_reorder("pa_kv_reorder",
                                    {input_info("key_cache"),
                                     input_info("value_cache"),
                                     input_info("block_indices"),
                                     input_info("block_indices_begins"),
                                     input_info("block_update_indices"),
                                     input_info("block_update_indices_begins")});
    pa_reorder.kv_heads_num = kv_heads;
    pa_reorder.adjusted_k_head_size = k_head_size;
    pa_reorder.adjusted_paged_attention_block_size = block_size;
    pa_reorder.adjusted_v_head_size = v_head_size;
    pa_reorder.cache_dt = data_types::f16;
    pa_reorder.is_kv_compressed = false;
    topo.add(pa_reorder);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("key_cache", key_cache_mem);
    network->set_input_data("value_cache", value_cache_mem);
    network->set_input_data("block_indices", block_indices_mem);
    network->set_input_data("block_indices_begins", block_indices_begins_mem);
    network->set_input_data("block_update_indices", block_update_indices_mem);
    network->set_input_data("block_update_indices_begins", block_update_indices_begins_mem);
    network->execute();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> key_ptr(key_cache_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> value_ptr(value_cache_mem, get_test_stream());

    for (size_t k = 0; k < k_head_size; k++) {
        const auto src0 = key_cache_ref[key_offset(0, 0, k, 0, kv_heads, k_head_size, block_size)];
        const auto dst17 = key_ptr[key_offset(1, 0, k, 1, kv_heads, k_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = key_cache_ref[key_offset(0, 0, k, 15, kv_heads, k_head_size, block_size)];
        const auto dst16 = key_ptr[key_offset(1, 0, k, 0, kv_heads, k_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t v = 0; v < v_head_size; v++) {
        const auto src0 = value_cache_ref[value_offset(0, 0, 0, v, kv_heads, v_head_size, block_size)];
        const auto dst17 = value_ptr[value_offset(1, 0, 1, v, kv_heads, v_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = value_cache_ref[value_offset(0, 0, 15, v, kv_heads, v_head_size, block_size)];
        const auto dst16 = value_ptr[value_offset(1, 0, 0, v, kv_heads, v_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    ASSERT_EQ(key_ptr[key_offset(0, 0, 0, 0, kv_heads, k_head_size, block_size)],
              key_cache_ref[key_offset(0, 0, 0, 0, kv_heads, k_head_size, block_size)]);
    ASSERT_EQ(value_ptr[value_offset(0, 0, 0, 0, kv_heads, v_head_size, block_size)],
              value_cache_ref[value_offset(0, 0, 0, 0, kv_heads, v_head_size, block_size)]);
}

TEST(pa_kv_reorder_gpu, updates_are_scoped_per_sequence) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 3;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 2;
    constexpr size_t v_head_size = 2;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, k_head_size, block_size}, data_types::f16, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, v_head_size}, data_types::f16, format::bfyx};
    auto block_indices_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto block_indices_begins_layout = layout{ov::PartialShape{3}, data_types::i32, format::bfyx};
    auto block_update_indices_layout = layout{ov::PartialShape{4}, data_types::i32, format::bfyx};
    auto block_update_indices_begins_layout = layout{ov::PartialShape{3}, data_types::i32, format::bfyx};

    auto key_cache_mem = engine.allocate_memory(key_cache_layout);
    auto value_cache_mem = engine.allocate_memory(value_cache_layout);
    auto block_indices_mem = engine.allocate_memory(block_indices_layout);
    auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_layout);
    auto block_update_indices_mem = engine.allocate_memory(block_update_indices_layout);
    auto block_update_indices_begins_mem = engine.allocate_memory(block_update_indices_begins_layout);

    std::vector<ov::float16> key_cache_ref;
    std::vector<ov::float16> value_cache_ref;
    fill_key_cache(key_cache_mem, blocks_num, kv_heads, k_head_size, block_size, key_cache_ref);
    fill_value_cache(value_cache_mem, blocks_num, kv_heads, v_head_size, block_size, value_cache_ref);

    // Sequence 0 uses physical block 0, sequence 1 uses physical block 2.
    set_values<int32_t>(block_indices_mem, {0, 2});
    set_values<int32_t>(block_indices_begins_mem, {0, 1, 2});
    set_values<int32_t>(block_update_indices_mem, {
        1, 3,  // seq0: slot1 -> slot3 in block0
        2, 4,  // seq1: slot2 -> slot4 in block2
    });
    set_values<int32_t>(block_update_indices_begins_mem, {0, 1, 2});

    topology topo;
    topo.add(input_layout("key_cache", key_cache_layout));
    topo.add(input_layout("value_cache", value_cache_layout));
    topo.add(input_layout("block_indices", block_indices_layout));
    topo.add(input_layout("block_indices_begins", block_indices_begins_layout));
    topo.add(input_layout("block_update_indices", block_update_indices_layout));
    topo.add(input_layout("block_update_indices_begins", block_update_indices_begins_layout));

    auto pa_reorder = pa_kv_reorder("pa_kv_reorder",
                                    {input_info("key_cache"),
                                     input_info("value_cache"),
                                     input_info("block_indices"),
                                     input_info("block_indices_begins"),
                                     input_info("block_update_indices"),
                                     input_info("block_update_indices_begins")});
    pa_reorder.kv_heads_num = kv_heads;
    pa_reorder.adjusted_k_head_size = k_head_size;
    pa_reorder.adjusted_paged_attention_block_size = block_size;
    pa_reorder.adjusted_v_head_size = v_head_size;
    pa_reorder.cache_dt = data_types::f16;
    pa_reorder.is_kv_compressed = false;
    topo.add(pa_reorder);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("key_cache", key_cache_mem);
    network->set_input_data("value_cache", value_cache_mem);
    network->set_input_data("block_indices", block_indices_mem);
    network->set_input_data("block_indices_begins", block_indices_begins_mem);
    network->set_input_data("block_update_indices", block_update_indices_mem);
    network->set_input_data("block_update_indices_begins", block_update_indices_begins_mem);
    network->execute();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> key_ptr(key_cache_mem, get_test_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> value_ptr(value_cache_mem, get_test_stream());

    for (size_t k = 0; k < k_head_size; k++) {
        ASSERT_EQ(key_ptr[key_offset(0, 0, k, 3, kv_heads, k_head_size, block_size)],
                  key_cache_ref[key_offset(0, 0, k, 1, kv_heads, k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_offset(2, 0, k, 4, kv_heads, k_head_size, block_size)],
                  key_cache_ref[key_offset(2, 0, k, 2, kv_heads, k_head_size, block_size)]);

        // Unused middle block must stay untouched.
        ASSERT_EQ(key_ptr[key_offset(1, 0, k, 4, kv_heads, k_head_size, block_size)],
                  key_cache_ref[key_offset(1, 0, k, 4, kv_heads, k_head_size, block_size)]);
    }

    for (size_t v = 0; v < v_head_size; v++) {
        ASSERT_EQ(value_ptr[value_offset(0, 0, 3, v, kv_heads, v_head_size, block_size)],
                  value_cache_ref[value_offset(0, 0, 1, v, kv_heads, v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_offset(2, 0, 4, v, kv_heads, v_head_size, block_size)],
                  value_cache_ref[value_offset(2, 0, 2, v, kv_heads, v_head_size, block_size)]);

        ASSERT_EQ(value_ptr[value_offset(1, 0, 4, v, kv_heads, v_head_size, block_size)],
                  value_cache_ref[value_offset(1, 0, 4, v, kv_heads, v_head_size, block_size)]);
    }
}
