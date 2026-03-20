// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/pa_kv_reorder.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>

#include <cstring>
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

size_t key_comp_byte_offset(size_t block,
                            size_t head,
                            size_t token,
                            size_t byte_in_fp16,
                            bool is_zp,
                            size_t kv_heads,
                            size_t k_head_size,
                            size_t adjusted_k_head_size,
                            size_t block_size) {
    const size_t block_base = block * kv_heads * adjusted_k_head_size * block_size +
                              head * adjusted_k_head_size * block_size;
    const size_t comp_base = block_base + k_head_size * block_size;
    const size_t token_base = (is_zp ? (block_size + token) : token) * sizeof(ov::float16);
    return comp_base + token_base + byte_in_fp16;
}

size_t value_comp_byte_offset(size_t block,
                              size_t head,
                              size_t token,
                              size_t byte_in_fp16,
                              bool is_zp,
                              size_t kv_heads,
                              size_t v_head_size,
                              size_t adjusted_v_head_size,
                              size_t block_size) {
    const size_t block_base = block * kv_heads * block_size * adjusted_v_head_size +
                              head * block_size * adjusted_v_head_size;
    const size_t comp_base = block_base + v_head_size * block_size;
    const size_t token_base = (is_zp ? (block_size + token) : token) * sizeof(ov::float16);
    return comp_base + token_base + byte_in_fp16;
}

size_t value_data_offset_compressed(size_t block,
                                    size_t head,
                                    size_t token,
                                    size_t v,
                                    size_t kv_heads,
                                    size_t v_head_size,
                                    size_t adjusted_v_head_size,
                                    size_t block_size) {
    const size_t block_base = block * kv_heads * block_size * adjusted_v_head_size +
                              head * block_size * adjusted_v_head_size;
    return block_base + token * v_head_size + v;
}

ov::float16 read_fp16_from_i8_buffer(const cldnn::mem_lock<int8_t, mem_lock_type::read>& ptr, size_t byte_offset) {
    const auto lo = static_cast<uint8_t>(ptr[byte_offset]);
    const auto hi = static_cast<uint8_t>(ptr[byte_offset + 1]);
    const uint16_t bits = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
    return ov::float16::from_bits(bits);
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
    topo.add(mutable_data("key_cache", key_cache_mem));
    topo.add(mutable_data("value_cache", value_cache_mem));
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
    topo.add(mutable_data("key_cache", key_cache_mem));
    topo.add(mutable_data("value_cache", value_cache_mem));
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

TEST(pa_kv_reorder_gpu, copy_between_blocks_single_sequence_compressed) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 2;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 4;
    constexpr size_t v_head_size = 3;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 2;
    constexpr size_t adjusted_k_head_size = k_head_size + scales_zp_size;
    constexpr size_t adjusted_v_head_size = v_head_size + scales_zp_size;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, adjusted_k_head_size, block_size}, data_types::i8, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_v_head_size}, data_types::i8, format::bfyx};
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

    std::vector<int8_t> key_cache_ref(key_cache_mem->count());
    std::vector<int8_t> value_cache_ref(value_cache_mem->count());
    for (size_t i = 0; i < key_cache_ref.size(); i++) {
        key_cache_ref[i] = static_cast<int8_t>((static_cast<int>(i) % 101) - 50);
    }
    for (size_t i = 0; i < value_cache_ref.size(); i++) {
        value_cache_ref[i] = static_cast<int8_t>((static_cast<int>(i) % 97) - 48);
    }
    set_values<int8_t>(key_cache_mem, key_cache_ref);
    set_values<int8_t>(value_cache_mem, value_cache_ref);

    set_values<int32_t>(block_indices_mem, {0, 1});
    set_values<int32_t>(block_indices_begins_mem, {0, 2});
    set_values<int32_t>(block_update_indices_mem, {
        0, 17,
        15, 16,
    });
    set_values<int32_t>(block_update_indices_begins_mem, {0, 2});

    topology topo;
    topo.add(mutable_data("key_cache", key_cache_mem));
    topo.add(mutable_data("value_cache", value_cache_mem));
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
    pa_reorder.adjusted_k_head_size = adjusted_k_head_size;
    pa_reorder.adjusted_paged_attention_block_size = block_size;
    pa_reorder.adjusted_v_head_size = adjusted_v_head_size;
    pa_reorder.cache_dt = data_types::i8;
    pa_reorder.is_kv_compressed = true;
    pa_reorder.scales_zp_size = scales_zp_size;
    topo.add(pa_reorder);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("block_indices", block_indices_mem);
    network->set_input_data("block_indices_begins", block_indices_begins_mem);
    network->set_input_data("block_update_indices", block_update_indices_mem);
    network->set_input_data("block_update_indices_begins", block_update_indices_begins_mem);
    network->execute();

    cldnn::mem_lock<int8_t, mem_lock_type::read> key_ptr(key_cache_mem, get_test_stream());
    cldnn::mem_lock<int8_t, mem_lock_type::read> value_ptr(value_cache_mem, get_test_stream());

    for (size_t k = 0; k < k_head_size; k++) {
        const auto src0 = key_cache_ref[key_offset(0, 0, k, 0, kv_heads, adjusted_k_head_size, block_size)];
        const auto dst17 = key_ptr[key_offset(1, 0, k, 1, kv_heads, adjusted_k_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = key_cache_ref[key_offset(0, 0, k, 15, kv_heads, adjusted_k_head_size, block_size)];
        const auto dst16 = key_ptr[key_offset(1, 0, k, 0, kv_heads, adjusted_k_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t v = 0; v < v_head_size; v++) {
        const auto src0 = value_cache_ref[value_data_offset_compressed(0, 0, 0, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        const auto dst17 = value_ptr[value_data_offset_compressed(1, 0, 1, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = value_cache_ref[value_data_offset_compressed(0, 0, 15, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        const auto dst16 = value_ptr[value_data_offset_compressed(1, 0, 0, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
        ASSERT_EQ(key_ptr[key_comp_byte_offset(1, 0, 1, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(0, 0, 0, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_comp_byte_offset(1, 0, 1, byte, true, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(0, 0, 0, byte, true, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);

        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 1, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 0, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 1, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 0, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
    }
}

TEST(pa_kv_reorder_gpu, updates_are_scoped_per_sequence_compressed) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 3;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 2;
    constexpr size_t v_head_size = 2;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 2;
    constexpr size_t adjusted_k_head_size = k_head_size + scales_zp_size;
    constexpr size_t adjusted_v_head_size = v_head_size + scales_zp_size;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, adjusted_k_head_size, block_size}, data_types::i8, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_v_head_size}, data_types::i8, format::bfyx};
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

    std::vector<int8_t> key_cache_ref(key_cache_mem->count());
    std::vector<int8_t> value_cache_ref(value_cache_mem->count());
    for (size_t i = 0; i < key_cache_ref.size(); i++) {
        key_cache_ref[i] = static_cast<int8_t>((static_cast<int>(i) % 89) - 44);
    }
    for (size_t i = 0; i < value_cache_ref.size(); i++) {
        value_cache_ref[i] = static_cast<int8_t>((static_cast<int>(i) % 83) - 41);
    }
    set_values<int8_t>(key_cache_mem, key_cache_ref);
    set_values<int8_t>(value_cache_mem, value_cache_ref);

    set_values<int32_t>(block_indices_mem, {0, 2});
    set_values<int32_t>(block_indices_begins_mem, {0, 1, 2});
    set_values<int32_t>(block_update_indices_mem, {
        1, 3,  // seq0: slot1 -> slot3 in block0
        2, 4,  // seq1: slot2 -> slot4 in block2
    });
    set_values<int32_t>(block_update_indices_begins_mem, {0, 1, 2});

    topology topo;
    topo.add(mutable_data("key_cache", key_cache_mem));
    topo.add(mutable_data("value_cache", value_cache_mem));
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
    pa_reorder.adjusted_k_head_size = adjusted_k_head_size;
    pa_reorder.adjusted_paged_attention_block_size = block_size;
    pa_reorder.adjusted_v_head_size = adjusted_v_head_size;
    pa_reorder.cache_dt = data_types::i8;
    pa_reorder.is_kv_compressed = true;
    pa_reorder.scales_zp_size = scales_zp_size;
    topo.add(pa_reorder);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("block_indices", block_indices_mem);
    network->set_input_data("block_indices_begins", block_indices_begins_mem);
    network->set_input_data("block_update_indices", block_update_indices_mem);
    network->set_input_data("block_update_indices_begins", block_update_indices_begins_mem);
    network->execute();

    cldnn::mem_lock<int8_t, mem_lock_type::read> key_ptr(key_cache_mem, get_test_stream());
    cldnn::mem_lock<int8_t, mem_lock_type::read> value_ptr(value_cache_mem, get_test_stream());

    for (size_t k = 0; k < k_head_size; k++) {
        ASSERT_EQ(key_ptr[key_offset(0, 0, k, 3, kv_heads, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_offset(0, 0, k, 1, kv_heads, adjusted_k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_offset(2, 0, k, 4, kv_heads, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_offset(2, 0, k, 2, kv_heads, adjusted_k_head_size, block_size)]);

        ASSERT_EQ(key_ptr[key_offset(1, 0, k, 4, kv_heads, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_offset(1, 0, k, 4, kv_heads, adjusted_k_head_size, block_size)]);
    }

    for (size_t v = 0; v < v_head_size; v++) {
        ASSERT_EQ(value_ptr[value_data_offset_compressed(0, 0, 3, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_data_offset_compressed(0, 0, 1, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_data_offset_compressed(2, 0, 4, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_data_offset_compressed(2, 0, 2, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);

        ASSERT_EQ(value_ptr[value_data_offset_compressed(1, 0, 4, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_data_offset_compressed(1, 0, 4, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
    }

    for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
        ASSERT_EQ(key_ptr[key_comp_byte_offset(0, 0, 3, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(0, 0, 1, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_comp_byte_offset(0, 0, 3, byte, true, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(0, 0, 1, byte, true, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);

        ASSERT_EQ(key_ptr[key_comp_byte_offset(2, 0, 4, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(2, 0, 2, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_comp_byte_offset(2, 0, 4, byte, true, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(2, 0, 2, byte, true, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);

        ASSERT_EQ(value_ptr[value_comp_byte_offset(0, 0, 3, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 1, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset(0, 0, 3, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 1, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);

        ASSERT_EQ(value_ptr[value_comp_byte_offset(2, 0, 4, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(2, 0, 2, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset(2, 0, 4, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(2, 0, 2, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);

        ASSERT_EQ(key_ptr[key_comp_byte_offset(1, 0, 4, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(1, 0, 4, byte, false, kv_heads, k_head_size, adjusted_k_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 4, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(1, 0, 4, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
    }
}

TEST(pa_kv_reorder_gpu, copy_between_blocks_single_sequence_compressed_key_by_channel) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 2;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 4;
    constexpr size_t v_head_size = 3;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 2;
    constexpr size_t adjusted_paged_attention_block_size = cldnn::paged_attention::block_size + scales_zp_size;
    constexpr size_t adjusted_v_head_size = v_head_size + scales_zp_size;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, k_head_size, adjusted_paged_attention_block_size}, data_types::i8, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_v_head_size}, data_types::i8, format::bfyx};
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

    std::vector<int8_t> key_cache_ref(key_cache_mem->count());
    std::vector<int8_t> value_cache_ref(value_cache_mem->count());

    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t h = 0; h < kv_heads; h++) {
            for (size_t k = 0; k < k_head_size; k++) {
                for (size_t t = 0; t < block_size; t++) {
                    key_cache_ref[key_offset(b, h, k, t, kv_heads, k_head_size, adjusted_paged_attention_block_size)] =
                        static_cast<int8_t>(static_cast<int>(10 * b + 3 * k + t) - 32);
                }

                const size_t comp_scale_byte_offset = key_offset(b, h, k, block_size, kv_heads, k_head_size, adjusted_paged_attention_block_size);
                const size_t comp_zp_byte_offset = comp_scale_byte_offset + sizeof(ov::float16);
                const ov::float16 scale_inv = ov::float16(1.0f);
                const ov::float16 zp = ov::float16(0.0f);
                std::memcpy(key_cache_ref.data() + comp_scale_byte_offset, &scale_inv, sizeof(ov::float16));
                std::memcpy(key_cache_ref.data() + comp_zp_byte_offset, &zp, sizeof(ov::float16));
            }
        }
    }

    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t h = 0; h < kv_heads; h++) {
            for (size_t t = 0; t < block_size; t++) {
                for (size_t v = 0; v < v_head_size; v++) {
                    value_cache_ref[value_data_offset_compressed(b, h, t, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)] =
                        static_cast<int8_t>(static_cast<int>(13 * b + 5 * t + v) - 40);
                }

                const ov::float16 scale = ov::float16(0.25f * static_cast<float>(1 + ((b + t) % 3)));
                const ov::float16 zp = ov::float16(static_cast<float>((static_cast<int>(t) % 5) - 2));
                for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
                    const auto scale_off = value_comp_byte_offset(b,
                                                                  h,
                                                                  t,
                                                                  byte,
                                                                  false,
                                                                  kv_heads,
                                                                  v_head_size,
                                                                  adjusted_v_head_size,
                                                                  block_size);
                    const auto zp_off = value_comp_byte_offset(b,
                                                               h,
                                                               t,
                                                               byte,
                                                               true,
                                                               kv_heads,
                                                               v_head_size,
                                                               adjusted_v_head_size,
                                                               block_size);
                    value_cache_ref[scale_off] = reinterpret_cast<const int8_t*>(&scale)[byte];
                    value_cache_ref[zp_off] = reinterpret_cast<const int8_t*>(&zp)[byte];
                }
            }
        }
    }

    set_values<int8_t>(key_cache_mem, key_cache_ref);
    set_values<int8_t>(value_cache_mem, value_cache_ref);

    set_values<int32_t>(block_indices_mem, {0, 1});
    set_values<int32_t>(block_indices_begins_mem, {0, 2});
    set_values<int32_t>(block_update_indices_mem, {
        0, 17,
        15, 16,
    });
    set_values<int32_t>(block_update_indices_begins_mem, {0, 2});

    topology topo;
    topo.add(mutable_data("key_cache", key_cache_mem));
    topo.add(mutable_data("value_cache", value_cache_mem));
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
    pa_reorder.adjusted_paged_attention_block_size = adjusted_paged_attention_block_size;
    pa_reorder.adjusted_v_head_size = adjusted_v_head_size;
    pa_reorder.cache_dt = data_types::i8;
    pa_reorder.is_kv_compressed = true;
    pa_reorder.is_key_by_channel = true;
    pa_reorder.scales_zp_size = scales_zp_size;
    topo.add(pa_reorder);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("block_indices", block_indices_mem);
    network->set_input_data("block_indices_begins", block_indices_begins_mem);
    network->set_input_data("block_update_indices", block_update_indices_mem);
    network->set_input_data("block_update_indices_begins", block_update_indices_begins_mem);
    network->execute();

    cldnn::mem_lock<int8_t, mem_lock_type::read> key_ptr(key_cache_mem, get_test_stream());
    cldnn::mem_lock<int8_t, mem_lock_type::read> value_ptr(value_cache_mem, get_test_stream());

    // In key-by-channel mode, key cache is re-quantized per channel on destination block.
    // Validate by dequantized semantics instead of raw byte equality.
    for (size_t k = 0; k < k_head_size; k++) {
        const auto src0_q = key_cache_ref[key_offset(0, 0, k, 0, kv_heads, k_head_size, adjusted_paged_attention_block_size)];
        const auto src15_q = key_cache_ref[key_offset(0, 0, k, 15, kv_heads, k_head_size, adjusted_paged_attention_block_size)];

        const size_t comp_scale_byte_offset = key_offset(1, 0, k, block_size, kv_heads, k_head_size, adjusted_paged_attention_block_size);
        const size_t comp_zp_byte_offset = comp_scale_byte_offset + sizeof(ov::float16);
        const float dst_scale_inv = static_cast<float>(read_fp16_from_i8_buffer(key_ptr, comp_scale_byte_offset));
        const float dst_zp = static_cast<float>(read_fp16_from_i8_buffer(key_ptr, comp_zp_byte_offset));

        const float dst17_dequant = (static_cast<float>(key_ptr[key_offset(1, 0, k, 1, kv_heads, k_head_size, adjusted_paged_attention_block_size)]) - dst_zp) * dst_scale_inv;
        const float dst16_dequant = (static_cast<float>(key_ptr[key_offset(1, 0, k, 0, kv_heads, k_head_size, adjusted_paged_attention_block_size)]) - dst_zp) * dst_scale_inv;

        ASSERT_NEAR(dst17_dequant, static_cast<float>(src0_q), 1.0f);
        ASSERT_NEAR(dst16_dequant, static_cast<float>(src15_q), 1.0f);
    }

    // Value cache remains per-token compressed copy behavior.
    for (size_t v = 0; v < v_head_size; v++) {
        const auto src0 = value_cache_ref[value_data_offset_compressed(0, 0, 0, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        const auto dst17 = value_ptr[value_data_offset_compressed(1, 0, 1, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = value_cache_ref[value_data_offset_compressed(0, 0, 15, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        const auto dst16 = value_ptr[value_data_offset_compressed(1, 0, 0, v, kv_heads, v_head_size, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 1, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 0, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 1, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 0, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size)]);
    }
}
