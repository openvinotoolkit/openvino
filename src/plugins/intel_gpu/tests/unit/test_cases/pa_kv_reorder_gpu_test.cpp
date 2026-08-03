// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>
#include <intel_gpu/primitives/pa_kv_reorder.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

size_t key_offset(size_t block, size_t head, size_t k, size_t token, size_t kv_heads, size_t k_head_size, size_t block_size) {
    return block * kv_heads * k_head_size * block_size + head * k_head_size * block_size + k * block_size + token;
}

size_t value_offset(size_t block, size_t head, size_t token, size_t v, size_t kv_heads, size_t v_head_size, size_t block_size) {
    return block * kv_heads * block_size * v_head_size + head * block_size * v_head_size + token * v_head_size + v;
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
    const size_t block_base = block * kv_heads * adjusted_k_head_size * block_size + head * adjusted_k_head_size * block_size;
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
    const size_t block_base = block * kv_heads * block_size * adjusted_v_head_size + head * block_size * adjusted_v_head_size;
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
    const size_t block_base = block * kv_heads * block_size * adjusted_v_head_size + head * block_size * adjusted_v_head_size;
    return block_base + token * v_head_size + v;
}

// u4 V BY_TOKEN layout: each token row is `packed_v_head_size` data bytes followed inline
// by `[fp16 scale][fp16 zp]` (mirrors quantize_and_save_per_token in pa_kv_cache_update_ref.cl
// when out_data_pitch == 1).
size_t value_data_offset_int4_per_token(size_t block, size_t head, size_t token, size_t v, size_t kv_heads, size_t adjusted_v_head_size, size_t block_size) {
    const size_t block_base = block * kv_heads * block_size * adjusted_v_head_size + head * block_size * adjusted_v_head_size;
    return block_base + token * adjusted_v_head_size + v;
}

size_t value_comp_byte_offset_int4_per_token(size_t block,
                                             size_t head,
                                             size_t token,
                                             size_t byte_in_fp16,
                                             bool is_zp,
                                             size_t kv_heads,
                                             size_t packed_v_head_size,
                                             size_t adjusted_v_head_size,
                                             size_t block_size) {
    const size_t block_base = block * kv_heads * block_size * adjusted_v_head_size + head * block_size * adjusted_v_head_size;
    const size_t row_base = block_base + token * adjusted_v_head_size + packed_v_head_size;
    return row_base + (is_zp ? sizeof(ov::float16) : 0) + byte_in_fp16;
}

void run_copy_between_blocks_single_sequence_compressed_int4_test(data_types cache_dt) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 2;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 32;
    constexpr size_t v_head_size = 32;
    constexpr size_t packed_k_head_size = k_head_size / 2;
    constexpr size_t packed_v_head_size = v_head_size / 2;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 4;
    constexpr size_t adjusted_k_head_size = packed_k_head_size + scales_zp_size;
    constexpr size_t adjusted_v_head_size = packed_v_head_size + scales_zp_size;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, adjusted_k_head_size, block_size}, data_types::u8, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_v_head_size}, data_types::u8, format::bfyx};
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

    std::vector<uint8_t> key_cache_ref(key_cache_mem->count());
    std::vector<uint8_t> value_cache_ref(value_cache_mem->count());
    for (size_t i = 0; i < key_cache_ref.size(); i++) {
        key_cache_ref[i] = static_cast<uint8_t>(i % 251);
    }
    for (size_t i = 0; i < value_cache_ref.size(); i++) {
        value_cache_ref[i] = static_cast<uint8_t>((3 * i + 17) % 251);
    }
    set_values<uint8_t>(key_cache_mem, key_cache_ref);
    set_values<uint8_t>(value_cache_mem, value_cache_ref);

    set_values<int32_t>(block_indices_mem, {0, 1});
    set_values<int32_t>(block_indices_begins_mem, {0, 2});
    set_values<int32_t>(block_update_indices_mem,
                        {
                            0,
                            17,
                            15,
                            16,
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
    pa_reorder.cache_dt = cache_dt;
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
    network->get_stream().finish();

    cldnn::mem_lock<uint8_t, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

    for (size_t k = 0; k < packed_k_head_size; k++) {
        const auto src0 = key_cache_ref[key_offset(0, 0, k, 0, kv_heads, adjusted_k_head_size, block_size)];
        const auto dst17 = key_ptr[key_offset(1, 0, k, 1, kv_heads, adjusted_k_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = key_cache_ref[key_offset(0, 0, k, 15, kv_heads, adjusted_k_head_size, block_size)];
        const auto dst16 = key_ptr[key_offset(1, 0, k, 0, kv_heads, adjusted_k_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t v = 0; v < packed_v_head_size; v++) {
        const auto src0 = value_cache_ref[value_data_offset_int4_per_token(0, 0, 0, v, kv_heads, adjusted_v_head_size, block_size)];
        const auto dst17 = value_ptr[value_data_offset_int4_per_token(1, 0, 1, v, kv_heads, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = value_cache_ref[value_data_offset_int4_per_token(0, 0, 15, v, kv_heads, adjusted_v_head_size, block_size)];
        const auto dst16 = value_ptr[value_data_offset_int4_per_token(1, 0, 0, v, kv_heads, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
        ASSERT_EQ(key_ptr[key_comp_byte_offset(1, 0, 1, byte, false, kv_heads, packed_k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(0, 0, 0, byte, false, kv_heads, packed_k_head_size, adjusted_k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_comp_byte_offset(1, 0, 1, byte, true, kv_heads, packed_k_head_size, adjusted_k_head_size, block_size)],
                  key_cache_ref[key_comp_byte_offset(0, 0, 0, byte, true, kv_heads, packed_k_head_size, adjusted_k_head_size, block_size)]);

        ASSERT_EQ(value_ptr[value_comp_byte_offset_int4_per_token(1, 0, 1, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset_int4_per_token(0, 0, 0, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset_int4_per_token(1, 0, 1, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset_int4_per_token(0, 0, 0, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)]);
    }
}

template <typename ByteT>
ov::float16 read_fp16_from_byte_buffer(const cldnn::mem_lock<ByteT, mem_lock_type::read>& ptr, size_t byte_offset) {
    static_assert(std::is_same_v<ByteT, int8_t> || std::is_same_v<ByteT, uint8_t>, "ByteT must be int8_t or uint8_t");
    const auto lo = static_cast<uint8_t>(ptr[byte_offset]);
    const auto hi = static_cast<uint8_t>(ptr[byte_offset + 1]);
    const uint16_t bits = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
    return ov::float16::from_bits(bits);
}

ov::float16 read_fp16_from_u8_vector(const std::vector<uint8_t>& buffer, size_t byte_offset) {
    const auto lo = buffer[byte_offset];
    const auto hi = buffer[byte_offset + 1];
    const uint16_t bits = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
    return ov::float16::from_bits(bits);
}

void fill_key_cache(memory::ptr key_cache_mem, size_t blocks_num, size_t kv_heads, size_t k_head_size, size_t block_size, std::vector<ov::float16>& values) {
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
    set_values<int32_t>(block_update_indices_mem,
                        {
                            0,
                            17,
                            15,
                            16,
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
    network->get_stream().finish();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

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

    ASSERT_EQ(key_ptr[key_offset(0, 0, 0, 0, kv_heads, k_head_size, block_size)], key_cache_ref[key_offset(0, 0, 0, 0, kv_heads, k_head_size, block_size)]);
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
    set_values<int32_t>(block_update_indices_mem,
                        {
                            1,
                            3,  // seq0: slot1 -> slot3 in block0
                            2,
                            4,  // seq1: slot2 -> slot4 in block2
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
    network->get_stream().finish();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<ov::float16, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

    for (size_t k = 0; k < k_head_size; k++) {
        ASSERT_EQ(key_ptr[key_offset(0, 0, k, 3, kv_heads, k_head_size, block_size)], key_cache_ref[key_offset(0, 0, k, 1, kv_heads, k_head_size, block_size)]);
        ASSERT_EQ(key_ptr[key_offset(2, 0, k, 4, kv_heads, k_head_size, block_size)], key_cache_ref[key_offset(2, 0, k, 2, kv_heads, k_head_size, block_size)]);

        // Unused middle block must stay untouched.
        ASSERT_EQ(key_ptr[key_offset(1, 0, k, 4, kv_heads, k_head_size, block_size)], key_cache_ref[key_offset(1, 0, k, 4, kv_heads, k_head_size, block_size)]);
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
    set_values<int32_t>(block_update_indices_mem,
                        {
                            0,
                            17,
                            15,
                            16,
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
    network->get_stream().finish();

    cldnn::mem_lock<int8_t, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<int8_t, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

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

TEST(pa_kv_reorder_gpu, copy_between_blocks_single_sequence_compressed_u4) {
    run_copy_between_blocks_single_sequence_compressed_int4_test(data_types::u4);
}

TEST(pa_kv_reorder_gpu, copy_between_blocks_single_sequence_compressed_i4) {
    run_copy_between_blocks_single_sequence_compressed_int4_test(data_types::i4);
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
    set_values<int32_t>(block_update_indices_mem,
                        {
                            1,
                            3,  // seq0: slot1 -> slot3 in block0
                            2,
                            4,  // seq1: slot2 -> slot4 in block2
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
    network->get_stream().finish();

    cldnn::mem_lock<int8_t, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<int8_t, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

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
                    const auto scale_off = value_comp_byte_offset(b, h, t, byte, false, kv_heads, v_head_size, adjusted_v_head_size, block_size);
                    const auto zp_off = value_comp_byte_offset(b, h, t, byte, true, kv_heads, v_head_size, adjusted_v_head_size, block_size);
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
    set_values<int32_t>(block_update_indices_mem,
                        {
                            0,
                            17,
                            15,
                            16,
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
    network->get_stream().finish();

    cldnn::mem_lock<int8_t, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<int8_t, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

    // In key-by-channel mode, key cache is re-quantized per channel on destination block.
    // Validate by dequantized semantics instead of raw byte equality.
    for (size_t k = 0; k < k_head_size; k++) {
        const auto src0_q = key_cache_ref[key_offset(0, 0, k, 0, kv_heads, k_head_size, adjusted_paged_attention_block_size)];
        const auto src15_q = key_cache_ref[key_offset(0, 0, k, 15, kv_heads, k_head_size, adjusted_paged_attention_block_size)];

        const size_t comp_scale_byte_offset = key_offset(1, 0, k, block_size, kv_heads, k_head_size, adjusted_paged_attention_block_size);
        const size_t comp_zp_byte_offset = comp_scale_byte_offset + sizeof(ov::float16);
        const float dst_scale_inv = static_cast<float>(read_fp16_from_byte_buffer(key_ptr, comp_scale_byte_offset));
        const float dst_zp = static_cast<float>(read_fp16_from_byte_buffer(key_ptr, comp_zp_byte_offset));

        const float dst17_dequant =
            (static_cast<float>(key_ptr[key_offset(1, 0, k, 1, kv_heads, k_head_size, adjusted_paged_attention_block_size)]) - dst_zp) * dst_scale_inv;
        const float dst16_dequant =
            (static_cast<float>(key_ptr[key_offset(1, 0, k, 0, kv_heads, k_head_size, adjusted_paged_attention_block_size)]) - dst_zp) * dst_scale_inv;

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

TEST(pa_kv_reorder_gpu, copy_between_blocks_single_sequence_compressed_u4_key_by_channel) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 2;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 32;
    constexpr size_t v_head_size = 16;
    constexpr size_t subgroup_size = 16;
    constexpr size_t block_size = cldnn::paged_attention::block_size;
    // u4 BY_CHANNEL key layout: each column = one head dim with 16 tokens packed as 8 bytes
    // (lo nibble = token 2t, hi nibble = token 2t+1) followed by [scale_inv (f16)][zp (f16)] = 12 bytes/col.
    // Number of columns = k_head_size (NOT halved, since BY_CHANNEL packs along the token axis).
    constexpr size_t packed_block_size = block_size / 2;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 2;
    constexpr size_t adjusted_paged_attention_block_size = packed_block_size + scales_zp_size;
    // u4 V is per-token inline: each token row = packed_v_head_size bytes + [scale][zp] (f16 each).
    constexpr size_t packed_v_head_size = ((v_head_size / 2 + subgroup_size - 1) / subgroup_size) * subgroup_size;
    constexpr size_t adjusted_v_head_size = packed_v_head_size + scales_zp_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, k_head_size, adjusted_paged_attention_block_size}, data_types::u8, format::bfyx};
    auto value_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_v_head_size}, data_types::u8, format::bfyx};
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

    std::vector<uint8_t> key_cache_ref(key_cache_mem->count(), 0);
    std::vector<uint8_t> value_cache_ref(value_cache_mem->count(), 0);

    auto write_fp16_bytes = [](std::vector<uint8_t>& buffer, size_t byte_offset, ov::float16 value) {
        const auto bits = value.to_bits();
        buffer[byte_offset] = static_cast<uint8_t>(bits & 0xFF);
        buffer[byte_offset + 1] = static_cast<uint8_t>((bits >> 8) & 0xFF);
    };

    // Fill key cache: each head dim h gets a column with 8 packed token bytes (16 tokens / 2)
    // followed by [scale_inv (f16)][zp (f16)]. Within a packed byte, lo nibble = token 2t, hi = token 2t+1.
    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t h = 0; h < k_head_size; h++) {
            const float scale_inv = 0.10f + 0.01f * static_cast<float>(h);
            const float zp = static_cast<float>((static_cast<int>(h) % 5) + 1);

            const size_t comp_base = key_offset(b, 0, h, packed_block_size, kv_heads, k_head_size, adjusted_paged_attention_block_size);
            write_fp16_bytes(key_cache_ref, comp_base + 0 * sizeof(ov::float16), ov::float16(scale_inv));
            write_fp16_bytes(key_cache_ref, comp_base + 1 * sizeof(ov::float16), ov::float16(zp));

            for (size_t byte_in_col = 0; byte_in_col < packed_block_size; byte_in_col++) {
                const size_t t_lo = byte_in_col * 2;
                const size_t t_hi = byte_in_col * 2 + 1;
                const uint8_t q_lo = static_cast<uint8_t>((3 * t_lo + h + b) & 0xF);
                const uint8_t q_hi = static_cast<uint8_t>((3 * t_hi + h + b) & 0xF);
                key_cache_ref[key_offset(b, 0, h, byte_in_col, kv_heads, k_head_size, adjusted_paged_attention_block_size)] =
                    static_cast<uint8_t>((q_lo & 0xF) | ((q_hi & 0xF) << 4));
            }
        }
    }

    // Fill value cache with u4 per-token-inline layout: each token row holds packed data
    // followed by inline [scale][zp] (matches quantize_and_save_per_token, pitch == 1).
    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t t = 0; t < block_size; t++) {
            for (size_t p = 0; p < packed_v_head_size; p++) {
                value_cache_ref[value_data_offset_int4_per_token(b, 0, t, p, kv_heads, adjusted_v_head_size, block_size)] =
                    static_cast<uint8_t>((7 * t + 5 * p + b) % 251);
            }

            const ov::float16 scale = ov::float16(0.125f * static_cast<float>(1 + ((b + t) % 3)));
            const ov::float16 zp = ov::float16(static_cast<float>((static_cast<int>(t) % 7) - 3));
            for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
                const auto scale_off =
                    value_comp_byte_offset_int4_per_token(b, 0, t, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size);
                const auto zp_off = value_comp_byte_offset_int4_per_token(b, 0, t, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size);
                value_cache_ref[scale_off] = reinterpret_cast<const uint8_t*>(&scale)[byte];
                value_cache_ref[zp_off] = reinterpret_cast<const uint8_t*>(&zp)[byte];
            }
        }
    }

    set_values<uint8_t>(key_cache_mem, key_cache_ref);
    set_values<uint8_t>(value_cache_mem, value_cache_ref);

    set_values<int32_t>(block_indices_mem, {0, 1});
    set_values<int32_t>(block_indices_begins_mem, {0, 2});
    set_values<int32_t>(block_update_indices_mem,
                        {
                            0,
                            17,
                            15,
                            16,
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
    pa_reorder.adjusted_v_head_size = v_head_size + scales_zp_size;
    pa_reorder.cache_dt = data_types::u4;
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
    network->get_stream().finish();

    cldnn::mem_lock<uint8_t, mem_lock_type::read> key_ptr(key_cache_mem, network->get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> value_ptr(value_cache_mem, network->get_stream());

    auto read_nibble_at_token = [&](const auto& buffer, size_t block, size_t h, size_t token) {
        const size_t byte_off = key_offset(block, 0, h, token / 2, kv_heads, k_head_size, adjusted_paged_attention_block_size);
        const uint8_t packed = buffer[byte_off];
        return (token % 2 == 0) ? static_cast<uint8_t>(packed & 0xF) : static_cast<uint8_t>((packed >> 4) & 0xF);
    };

    auto read_col_comp = [&](const auto& buffer, size_t block, size_t h) {
        const size_t comp_base = key_offset(block, 0, h, packed_block_size, kv_heads, k_head_size, adjusted_paged_attention_block_size);
        const float scale_inv = static_cast<float>(read_fp16_from_byte_buffer(buffer, comp_base + 0 * sizeof(ov::float16)));
        const float zp = static_cast<float>(read_fp16_from_byte_buffer(buffer, comp_base + 1 * sizeof(ov::float16)));
        return std::make_pair(scale_inv, zp);
    };

    auto read_col_comp_ref = [&](const std::vector<uint8_t>& buffer, size_t block, size_t h) {
        const size_t comp_base = key_offset(block, 0, h, packed_block_size, kv_heads, k_head_size, adjusted_paged_attention_block_size);
        const float scale_inv = static_cast<float>(read_fp16_from_u8_vector(buffer, comp_base + 0 * sizeof(ov::float16)));
        const float zp = static_cast<float>(read_fp16_from_u8_vector(buffer, comp_base + 1 * sizeof(ov::float16)));
        return std::make_pair(scale_inv, zp);
    };

    // Reorder maps src token 0 -> dst token 17 (block 1, slot 1) and src token 15 -> dst token 16 (block 1, slot 0).
    // Compare in dequantized space because the dst column is requantized end-to-end on cross-block copies.
    for (size_t h = 0; h < k_head_size; h++) {
        const auto [src_scale_inv, src_zp] = read_col_comp_ref(key_cache_ref, 0, h);
        const auto [dst_scale_inv, dst_zp] = read_col_comp(key_ptr, 1, h);

        const uint8_t src_q0 = read_nibble_at_token(key_cache_ref, 0, h, 0);
        const uint8_t src_q15 = read_nibble_at_token(key_cache_ref, 0, h, 15);
        const float src_val0 = (static_cast<float>(src_q0) - src_zp) * src_scale_inv;
        const float src_val15 = (static_cast<float>(src_q15) - src_zp) * src_scale_inv;

        const uint8_t dst_q1 = read_nibble_at_token(key_ptr, 1, h, 1);
        const uint8_t dst_q0 = read_nibble_at_token(key_ptr, 1, h, 0);
        const float dst_val1 = (static_cast<float>(dst_q1) - dst_zp) * dst_scale_inv;
        const float dst_val0 = (static_cast<float>(dst_q0) - dst_zp) * dst_scale_inv;

        ASSERT_NEAR(dst_val1, src_val0, 1.0f);
        ASSERT_NEAR(dst_val0, src_val15, 1.0f);
    }

    for (size_t p = 0; p < packed_v_head_size; p++) {
        const auto src0 = value_cache_ref[value_data_offset_int4_per_token(0, 0, 0, p, kv_heads, adjusted_v_head_size, block_size)];
        const auto dst17 = value_ptr[value_data_offset_int4_per_token(1, 0, 1, p, kv_heads, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = value_cache_ref[value_data_offset_int4_per_token(0, 0, 15, p, kv_heads, adjusted_v_head_size, block_size)];
        const auto dst16 = value_ptr[value_data_offset_int4_per_token(1, 0, 0, p, kv_heads, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
        ASSERT_EQ(value_ptr[value_comp_byte_offset_int4_per_token(1, 0, 1, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset_int4_per_token(0, 0, 0, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset_int4_per_token(1, 0, 1, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset_int4_per_token(0, 0, 0, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)]);
    }
}
