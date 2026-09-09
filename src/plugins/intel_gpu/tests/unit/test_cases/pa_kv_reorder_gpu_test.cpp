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

#include "graph/impls/ocl/kernel_selector_helper.h"  // check_cm_jit_support
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

// ---------------------------------------------------------------------------------------------
// CM kernel path (impls/cm/pa_kv_cache_reorder_ref.cm).
//
// The CM kernel is selected when has_xattention is set: cm::PaKVReorderImplementationManager
// requires it and ocl::PA_KV_reorder rejects it, so the two managers are mutually exclusive and
// no force_implementations is needed.
//
// CM uses its own cache layout, different from the OCL one the helpers above describe:
//   K: token-major data [BLOCK_SIZE][K_HEAD_SIZE], then all scales, then all zps, each indexed
//      as [sub][channel]. Per-channel quant groups are SUB_BLOCK_SIZE tokens.
//   V: per-token rows [BLOCK_SIZE][V_HEAD_SIZE], then BLOCK_SIZE scales, then BLOCK_SIZE zps.
// ---------------------------------------------------------------------------------------------

// Mirrors reorder_sub_block_size in impls/cm/pa_kv_reorder.cpp.
constexpr size_t cm_sub_block_size = 16;

bool cm_reorder_available() {
    auto& engine = tests::get_test_engine();
    auto config = tests::get_test_default_config(engine);
    return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
}

size_t cm_key_head_base(size_t block, size_t head, size_t kv_heads, size_t k_head_size, size_t block_size) {
    const size_t per_head = k_head_size * (block_size + block_size / cm_sub_block_size * 2 * sizeof(ov::float16));
    return (block * kv_heads + head) * per_head;
}

size_t cm_key_data_offset(size_t block, size_t head, size_t slot, size_t channel, size_t kv_heads, size_t k_head_size, size_t block_size) {
    return cm_key_head_base(block, head, kv_heads, k_head_size, block_size) + slot * k_head_size + channel;
}

// byte offset of the fp16 scale (is_zp == false) or zp (is_zp == true) of one (sub-block, channel).
size_t cm_key_comp_offset(size_t block,
                          size_t head,
                          size_t sub,
                          size_t channel,
                          bool is_zp,
                          size_t kv_heads,
                          size_t k_head_size,
                          size_t block_size) {
    const size_t num_subs = block_size / cm_sub_block_size;
    const size_t comp_base = cm_key_head_base(block, head, kv_heads, k_head_size, block_size) + k_head_size * block_size;
    const size_t region = is_zp ? num_subs * k_head_size : 0;
    return comp_base + (region + sub * k_head_size + channel) * sizeof(ov::float16);
}

size_t cm_value_head_base(size_t block, size_t head, size_t kv_heads, size_t v_head_size, size_t block_size) {
    const size_t per_head = (v_head_size + 2 * sizeof(ov::float16)) * block_size;
    return (block * kv_heads + head) * per_head;
}

size_t cm_value_data_offset(size_t block, size_t head, size_t slot, size_t v, size_t kv_heads, size_t v_head_size, size_t block_size) {
    return cm_value_head_base(block, head, kv_heads, v_head_size, block_size) + slot * v_head_size + v;
}

size_t cm_value_comp_offset(size_t block, size_t head, size_t slot, bool is_zp, size_t kv_heads, size_t v_head_size, size_t block_size) {
    const size_t comp_base = cm_value_head_base(block, head, kv_heads, v_head_size, block_size) + v_head_size * block_size;
    return comp_base + ((is_zp ? block_size : 0) + slot) * sizeof(ov::float16);
}

void write_fp16_at(std::vector<uint8_t>& buffer, size_t byte_offset, float value) {
    const auto bits = ov::float16(value).to_bits();
    buffer[byte_offset] = static_cast<uint8_t>(bits & 0xFF);
    buffer[byte_offset + 1] = static_cast<uint8_t>((bits >> 8) & 0xFF);
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

// ---- CM kernel path ------------------------------------------------------------------------

namespace {

constexpr size_t cm_blocks_num = 2;
constexpr size_t cm_kv_heads = 1;
constexpr size_t cm_k_head_size = 16;
constexpr size_t cm_v_head_size = 16;
// The CM impl is only reachable with has_xattention, which pairs with the 256-token block size.
constexpr size_t cm_block_size = cldnn::paged_attention::block_size_xattn;
constexpr size_t cm_num_subs = cm_block_size / cm_sub_block_size;
constexpr size_t cm_scales_zp_size = 2 * sizeof(ov::float16);

layout cm_key_layout() {
    return layout{ov::PartialShape{static_cast<int64_t>(cm_blocks_num),
                                   static_cast<int64_t>(cm_kv_heads),
                                   static_cast<int64_t>(cm_k_head_size),
                                   static_cast<int64_t>(cm_block_size + cm_num_subs * cm_scales_zp_size)},
                  data_types::u8,
                  format::bfyx};
}

layout cm_value_layout() {
    return layout{ov::PartialShape{static_cast<int64_t>(cm_blocks_num),
                                   static_cast<int64_t>(cm_kv_heads),
                                   static_cast<int64_t>(cm_block_size),
                                   static_cast<int64_t>(cm_v_head_size + cm_scales_zp_size)},
                  data_types::u8,
                  format::bfyx};
}

size_t cm_k_off(size_t block, size_t slot, size_t channel) {
    return cm_key_data_offset(block, 0, slot, channel, cm_kv_heads, cm_k_head_size, cm_block_size);
}

size_t cm_k_comp_off(size_t block, size_t sub, size_t channel, bool is_zp) {
    return cm_key_comp_offset(block, 0, sub, channel, is_zp, cm_kv_heads, cm_k_head_size, cm_block_size);
}

size_t cm_v_off(size_t block, size_t slot, size_t v) {
    return cm_value_data_offset(block, 0, slot, v, cm_kv_heads, cm_v_head_size, cm_block_size);
}

size_t cm_v_comp_off(size_t block, size_t slot, bool is_zp) {
    return cm_value_comp_offset(block, 0, slot, is_zp, cm_kv_heads, cm_v_head_size, cm_block_size);
}

// Every K sub-block starts with scale_inv = 1 and zp = 0, so a stored byte dequantizes to
// itself. That keeps the expected values in the assertions readable.
void cm_fill_caches(std::vector<uint8_t>& key_ref, std::vector<uint8_t>& value_ref) {
    for (size_t b = 0; b < cm_blocks_num; b++) {
        for (size_t t = 0; t < cm_block_size; t++) {
            for (size_t c = 0; c < cm_k_head_size; c++) {
                key_ref[cm_k_off(b, t, c)] = static_cast<uint8_t>((7 * t + 3 * c + b) % 256);
            }
            for (size_t v = 0; v < cm_v_head_size; v++) {
                value_ref[cm_v_off(b, t, v)] = static_cast<uint8_t>((5 * t + 11 * v + b) % 256);
            }
            write_fp16_at(value_ref, cm_v_comp_off(b, t, false), 0.25f * static_cast<float>(1 + (t % 3)));
            write_fp16_at(value_ref, cm_v_comp_off(b, t, true), static_cast<float>((t % 5) - 2));
        }
        for (size_t s = 0; s < cm_num_subs; s++) {
            for (size_t c = 0; c < cm_k_head_size; c++) {
                write_fp16_at(key_ref, cm_k_comp_off(b, s, c, false), 1.0f);
                write_fp16_at(key_ref, cm_k_comp_off(b, s, c, true), 0.0f);
            }
        }
    }
}

void cm_run_by_channel_reorder(const std::vector<uint8_t>& key_ref,
                               const std::vector<uint8_t>& value_ref,
                               const std::vector<int32_t>& pairs,
                               std::vector<uint8_t>& key_out,
                               std::vector<uint8_t>& value_out) {
    auto& engine = get_test_engine();

    auto block_indices_layout = layout{ov::PartialShape{static_cast<int64_t>(cm_blocks_num)}, data_types::i32, format::bfyx};
    auto block_indices_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};
    auto update_layout = layout{ov::PartialShape{static_cast<int64_t>(pairs.size())}, data_types::i32, format::bfyx};
    auto update_begins_layout = layout{ov::PartialShape{2}, data_types::i32, format::bfyx};

    auto key_mem = engine.allocate_memory(cm_key_layout());
    auto value_mem = engine.allocate_memory(cm_value_layout());
    auto block_indices_mem = engine.allocate_memory(block_indices_layout);
    auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_layout);
    auto update_mem = engine.allocate_memory(update_layout);
    auto update_begins_mem = engine.allocate_memory(update_begins_layout);

    set_values<uint8_t>(key_mem, key_ref);
    set_values<uint8_t>(value_mem, value_ref);
    set_values<int32_t>(block_indices_mem, {0, 1});
    set_values<int32_t>(block_indices_begins_mem, {0, static_cast<int32_t>(cm_blocks_num)});
    set_values<int32_t>(update_mem, pairs);
    set_values<int32_t>(update_begins_mem, {0, static_cast<int32_t>(pairs.size() / 2)});

    topology topo;
    topo.add(mutable_data("key_cache", key_mem));
    topo.add(mutable_data("value_cache", value_mem));
    topo.add(input_layout("block_indices", block_indices_layout));
    topo.add(input_layout("block_indices_begins", block_indices_begins_layout));
    topo.add(input_layout("block_update_indices", update_layout));
    topo.add(input_layout("block_update_indices_begins", update_begins_layout));

    auto prim = pa_kv_reorder("pa_kv_reorder",
                              {input_info("key_cache"),
                               input_info("value_cache"),
                               input_info("block_indices"),
                               input_info("block_indices_begins"),
                               input_info("block_update_indices"),
                               input_info("block_update_indices_begins")});
    prim.kv_heads_num = cm_kv_heads;
    prim.adjusted_k_head_size = cm_k_head_size;
    prim.adjusted_paged_attention_block_size = cm_block_size + cm_scales_zp_size;
    prim.adjusted_v_head_size = cm_v_head_size + cm_scales_zp_size;
    prim.cache_dt = data_types::u8;
    prim.is_kv_compressed = true;
    prim.is_key_by_channel = true;
    prim.scales_zp_size = cm_scales_zp_size;

    prim.has_xattention = true;
    topo.add(prim);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("block_indices", block_indices_mem);
    network->set_input_data("block_indices_begins", block_indices_begins_mem);
    network->set_input_data("block_update_indices", update_mem);
    network->set_input_data("block_update_indices_begins", update_begins_mem);
    network->execute();
    network->get_stream().finish();

    cldnn::mem_lock<uint8_t, mem_lock_type::read> key_ptr(key_mem, network->get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> value_ptr(value_mem, network->get_stream());
    key_out.assign(key_ptr.begin(), key_ptr.end());
    value_out.assign(value_ptr.begin(), value_ptr.end());
}

}  // namespace

// src and dst share a physical block and a sub-block, so requantize_dst_subblock takes its fast
// path: the quantized row is copied verbatim and scale/zp are left alone.
TEST(pa_kv_reorder_gpu, cm_key_by_channel_same_sub_block_is_exact_copy) {
    if (!cm_reorder_available())
        GTEST_SKIP() << "CM JIT support and an Xe1+ device are required for the CM reorder kernel";

    std::vector<uint8_t> key_ref(cm_key_layout().count(), 0);
    std::vector<uint8_t> value_ref(cm_value_layout().count(), 0);
    cm_fill_caches(key_ref, value_ref);

    std::vector<uint8_t> key_out;
    std::vector<uint8_t> value_out;
    // Both slots live in block 0, sub-block 1.
    cm_run_by_channel_reorder(key_ref, value_ref, {20, 25}, key_out, value_out);

    for (size_t c = 0; c < cm_k_head_size; c++) {
        ASSERT_EQ(key_out[cm_k_off(0, 25, c)], key_ref[cm_k_off(0, 20, c)]) << "channel " << c;

        // No requantization: every other row and all scale/zp bytes stay byte-identical.
        for (size_t t = 0; t < cm_block_size; t++) {
            if (t == 25)
                continue;
            ASSERT_EQ(key_out[cm_k_off(0, t, c)], key_ref[cm_k_off(0, t, c)]) << "slot " << t << " channel " << c;
        }
        for (size_t s = 0; s < cm_num_subs; s++) {
            for (bool is_zp : {false, true}) {
                const auto off = cm_k_comp_off(0, s, c, is_zp);
                ASSERT_EQ(key_out[off], key_ref[off]) << "sub " << s << " channel " << c << " is_zp " << is_zp;
                ASSERT_EQ(key_out[off + 1], key_ref[off + 1]);
            }
        }
    }
}

// dst lands in a sequence's last sub-block, which pa_kv_cache_update only partially filled.
// touched_len = max(src, dst) + 1 must confine both the range recomputation and the stores to
// the rows that were actually written.
TEST(pa_kv_reorder_gpu, cm_key_by_channel_partial_sub_block_leaves_tail_untouched) {
    if (!cm_reorder_available())
        GTEST_SKIP() << "CM JIT support and an Xe1+ device are required for the CM reorder kernel";

    // src 100 (block 0, sub-block 6) -> dst 264 (block 1, sub-block 0, row 8). Cross-block, so
    // the slow path runs; touched_len = 265 leaves rows 0..8 valid and rows 9..15 never written.
    constexpr size_t src_id = 100;
    constexpr size_t dst_id = 264;
    constexpr size_t dst_slot = dst_id - cm_block_size;  // 8
    constexpr size_t valid_rows = dst_slot + 1;          // 9

    std::vector<uint8_t> key_ref(cm_key_layout().count(), 0);
    std::vector<uint8_t> value_ref(cm_value_layout().count(), 0);
    cm_fill_caches(key_ref, value_ref);

    constexpr float expected_range = 35.0f;
    for (size_t c = 0; c < cm_k_head_size; c++) {
        for (size_t r = 0; r < cm_sub_block_size; r++) {
            key_ref[cm_k_off(1, r, c)] = (r < valid_rows) ? static_cast<uint8_t>(10 + 5 * r) : static_cast<uint8_t>(200 + r);
        }
        // scale_inv = 1 / zp = 0 for every sub-block, so this byte dequantizes to 30.
        key_ref[cm_k_off(0, src_id, c)] = 30;
    }

    std::vector<uint8_t> key_out;
    std::vector<uint8_t> value_out;
    cm_run_by_channel_reorder(key_ref, value_ref, {static_cast<int32_t>(src_id), static_cast<int32_t>(dst_id)}, key_out, value_out);

    for (size_t c = 0; c < cm_k_head_size; c++) {
        for (size_t r = valid_rows; r < cm_sub_block_size; r++) {
            ASSERT_EQ(key_out[cm_k_off(1, r, c)], key_ref[cm_k_off(1, r, c)]) << "tail row " << r << " channel " << c;
        }

        // scale/zp must describe the valid rows' range, not the tail's.
        const float scale_inv = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(1, 0, c, false)));
        const float zp = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(1, 0, c, true)));
        ASSERT_NEAR(scale_inv, expected_range / 255.0f, 0.01f) << "channel " << c;

        // Valid rows keep their values; the dst row now carries the src value.
        for (size_t r = 0; r < valid_rows; r++) {
            const float expected = (r == dst_slot) ? 30.0f : static_cast<float>(10 + 5 * r);
            const float dequant = (static_cast<float>(key_out[cm_k_off(1, r, c)]) - zp) * scale_inv;
            ASSERT_NEAR(dequant, expected, 0.3f) << "row " << r << " channel " << c;
        }

        // A sub-block the reorder never targets stays byte-identical.
        for (size_t r = 0; r < cm_sub_block_size; r++) {
            const size_t slot = cm_sub_block_size + r;
            ASSERT_EQ(key_out[cm_k_off(1, slot, c)], key_ref[cm_k_off(1, slot, c)]) << "sub-block 1 row " << r;
        }
    }

    // V is a plain per-token copy in this mode.
    for (size_t v = 0; v < cm_v_head_size; v++) {
        ASSERT_EQ(value_out[cm_v_off(1, dst_slot, v)], value_ref[cm_v_off(0, src_id, v)]) << "v " << v;
    }
    for (bool is_zp : {false, true}) {
        const auto dst_off = cm_v_comp_off(1, dst_slot, is_zp);
        const auto src_off = cm_v_comp_off(0, src_id, is_zp);
        ASSERT_EQ(value_out[dst_off], value_ref[src_off]);
        ASSERT_EQ(value_out[dst_off + 1], value_ref[src_off + 1]);
    }
}

TEST(pa_kv_reorder_gpu, cm_key_by_channel_two_pairs_single_sequence) {
    if (!cm_reorder_available())
        GTEST_SKIP() << "CM JIT support and an Xe1+ device are required for the CM reorder kernel";

    // Pair A: src 30 (block 0, sub-block 1) -> dst 264 (block 1, sub-block 0, row 8). dst is the
    //         largest position, so touched_len = 265 and this sub-block has 9 valid rows.
    // Pair B: src 100 (block 0, sub-block 6) -> dst 50 (block 0, sub-block 3, row 2). Same block
    //         but a different sub-block, so it still takes the requantizing path; the sub-block
    //         sits well below touched_len, so all 16 rows are valid.
    constexpr size_t a_src = 30;
    constexpr size_t a_dst = 264;
    constexpr size_t a_dst_slot = a_dst - cm_block_size;  // 8
    constexpr size_t a_valid_rows = a_dst_slot + 1;       // 9

    constexpr size_t b_src = 100;
    constexpr size_t b_dst = 50;
    constexpr size_t b_sub_first_slot = 48;
    constexpr size_t b_dst_row = b_dst - b_sub_first_slot;  // 2

    std::vector<uint8_t> key_ref(cm_key_layout().count(), 0);
    std::vector<uint8_t> value_ref(cm_value_layout().count(), 0);
    cm_fill_caches(key_ref, value_ref);

    // A's dst sub-block: rows 0..8 hold 10..50, the tail holds much larger never-written bytes.
    // Row 8 is the dst and becomes 30, so the surviving values are 10..45 -> range 35.
    constexpr float a_expected_range = 35.0f;
    // B's dst sub-block: rows hold 60..90. Row 2 becomes 70, which is inside that span, so the
    // range stays 30.
    constexpr float b_expected_range = 30.0f;

    for (size_t c = 0; c < cm_k_head_size; c++) {
        for (size_t r = 0; r < cm_sub_block_size; r++) {
            key_ref[cm_k_off(1, r, c)] = (r < a_valid_rows) ? static_cast<uint8_t>(10 + 5 * r) : static_cast<uint8_t>(200 + r);
            key_ref[cm_k_off(0, b_sub_first_slot + r, c)] = static_cast<uint8_t>(60 + 2 * r);
        }
        // scale_inv = 1 / zp = 0 everywhere, so these bytes dequantize to themselves.
        key_ref[cm_k_off(0, a_src, c)] = 30;
        key_ref[cm_k_off(0, b_src, c)] = 70;
    }

    std::vector<uint8_t> key_out;
    std::vector<uint8_t> value_out;
    cm_run_by_channel_reorder(key_ref,
                              value_ref,
                              {static_cast<int32_t>(a_src),
                               static_cast<int32_t>(a_dst),
                               static_cast<int32_t>(b_src),
                               static_cast<int32_t>(b_dst)},
                              key_out,
                              value_out);

    for (size_t c = 0; c < cm_k_head_size; c++) {
        // ---- pair A: partial sub-block, tail must survive untouched.
        for (size_t r = a_valid_rows; r < cm_sub_block_size; r++) {
            ASSERT_EQ(key_out[cm_k_off(1, r, c)], key_ref[cm_k_off(1, r, c)]) << "A tail row " << r << " channel " << c;
        }

        const float a_scale = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(1, 0, c, false)));
        const float a_zp = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(1, 0, c, true)));
        ASSERT_NEAR(a_scale, a_expected_range / 255.0f, 0.01f) << "channel " << c;

        for (size_t r = 0; r < a_valid_rows; r++) {
            const float expected = (r == a_dst_slot) ? 30.0f : static_cast<float>(10 + 5 * r);
            const float dequant = (static_cast<float>(key_out[cm_k_off(1, r, c)]) - a_zp) * a_scale;
            ASSERT_NEAR(dequant, expected, 0.3f) << "A row " << r << " channel " << c;
        }

        // ---- pair B: full sub-block, every row requantized.
        const float b_scale = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(0, 3, c, false)));
        const float b_zp = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(0, 3, c, true)));
        ASSERT_NEAR(b_scale, b_expected_range / 255.0f, 0.01f) << "channel " << c;

        for (size_t r = 0; r < cm_sub_block_size; r++) {
            const float expected = (r == b_dst_row) ? 70.0f : static_cast<float>(60 + 2 * r);
            const float dequant = (static_cast<float>(key_out[cm_k_off(0, b_sub_first_slot + r, c)]) - b_zp) * b_scale;
            ASSERT_NEAR(dequant, expected, 0.3f) << "B row " << r << " channel " << c;
        }

        // ---- sub-blocks that only supplied sources, or were never named, stay byte-identical.
        for (size_t sub : {size_t{1}, size_t{6}}) {
            for (size_t r = 0; r < cm_sub_block_size; r++) {
                const size_t slot = sub * cm_sub_block_size + r;
                ASSERT_EQ(key_out[cm_k_off(0, slot, c)], key_ref[cm_k_off(0, slot, c)]) << "block 0 sub " << sub << " row " << r;
            }
            for (bool is_zp : {false, true}) {
                const auto off = cm_k_comp_off(0, sub, c, is_zp);
                ASSERT_EQ(key_out[off], key_ref[off]) << "block 0 sub " << sub << " is_zp " << is_zp;
                ASSERT_EQ(key_out[off + 1], key_ref[off + 1]);
            }
        }
        for (size_t r = 0; r < cm_sub_block_size; r++) {
            const size_t slot = cm_sub_block_size + r;
            ASSERT_EQ(key_out[cm_k_off(1, slot, c)], key_ref[cm_k_off(1, slot, c)]) << "block 1 sub 1 row " << r;
        }
    }

    // V is a plain per-token copy for both pairs.
    for (size_t v = 0; v < cm_v_head_size; v++) {
        ASSERT_EQ(value_out[cm_v_off(1, a_dst_slot, v)], value_ref[cm_v_off(0, a_src, v)]) << "A v " << v;
        ASSERT_EQ(value_out[cm_v_off(0, b_dst, v)], value_ref[cm_v_off(0, b_src, v)]) << "B v " << v;
    }
    for (bool is_zp : {false, true}) {
        ASSERT_EQ(value_out[cm_v_comp_off(1, a_dst_slot, is_zp)], value_ref[cm_v_comp_off(0, a_src, is_zp)]);
        ASSERT_EQ(value_out[cm_v_comp_off(1, a_dst_slot, is_zp) + 1], value_ref[cm_v_comp_off(0, a_src, is_zp) + 1]);
        ASSERT_EQ(value_out[cm_v_comp_off(0, b_dst, is_zp)], value_ref[cm_v_comp_off(0, b_src, is_zp)]);
        ASSERT_EQ(value_out[cm_v_comp_off(0, b_dst, is_zp) + 1], value_ref[cm_v_comp_off(0, b_src, is_zp) + 1]);
    }
}

TEST(pa_kv_reorder_gpu, cm_key_by_channel_chained_moves_single_sequence) {
    if (!cm_reorder_available())
        GTEST_SKIP() << "CM JIT support and an Xe1+ device are required for the CM reorder kernel";

    // Pair 1: src 100 (block 0, sub-block 6, row 4) -> dst 20  (block 0, sub-block 1, row 4).
    // Pair 2: src 300 (block 1, sub-block 2)        -> dst 100 (the slot pair 1 just read).
    constexpr size_t first_src = 100;
    constexpr size_t first_src_sub = 6;
    constexpr size_t first_src_row = first_src - first_src_sub * cm_sub_block_size;  // 4
    constexpr size_t first_dst = 20;
    constexpr size_t first_dst_sub = 1;
    constexpr size_t first_dst_row = first_dst - first_dst_sub * cm_sub_block_size;  // 4

    constexpr size_t second_src = 300;
    constexpr size_t second_src_slot = second_src - cm_block_size;  // 44
    constexpr size_t second_dst = first_src;

    std::vector<uint8_t> key_ref(cm_key_layout().count(), 0);
    std::vector<uint8_t> value_ref(cm_value_layout().count(), 0);
    cm_fill_caches(key_ref, value_ref);

    // Sub-block 6 holds 40..100 (step 4), so slot 100 arrives holding 56. Sub-block 1 holds
    // 30..105 (step 5). Block 1's source slot holds 77.
    constexpr float relayed_value = 40.0f + 4.0f * static_cast<float>(first_src_row);  // 56, 100 -> 20
    constexpr float incoming_value = 77.0f;                                            // 300 -> 100
    // Pair 1 drops 56 into sub-block 1, inside its 30..105 span, so that range stays 75.
    // Pair 2 drops 77 into sub-block 6, inside its 40..100 span, so that range stays 60.
    constexpr float first_expected_range = 75.0f;
    constexpr float second_expected_range = 60.0f;

    for (size_t c = 0; c < cm_k_head_size; c++) {
        for (size_t r = 0; r < cm_sub_block_size; r++) {
            key_ref[cm_k_off(0, first_src_sub * cm_sub_block_size + r, c)] = static_cast<uint8_t>(40 + 4 * r);
            key_ref[cm_k_off(0, first_dst_sub * cm_sub_block_size + r, c)] = static_cast<uint8_t>(30 + 5 * r);
        }
        // scale_inv = 1 / zp = 0 everywhere, so this byte dequantizes to incoming_value.
        key_ref[cm_k_off(1, second_src_slot, c)] = static_cast<uint8_t>(incoming_value);
    }

    std::vector<uint8_t> key_out;
    std::vector<uint8_t> value_out;
    cm_run_by_channel_reorder(
        key_ref,
        value_ref,
        {static_cast<int32_t>(first_src), static_cast<int32_t>(first_dst), static_cast<int32_t>(second_src), static_cast<int32_t>(second_dst)},
        key_out,
        value_out);

    for (size_t c = 0; c < cm_k_head_size; c++) {
        // ---- pair 1's dst sub-block: row 4 must hold what slot 100 contained on entry (56),
        // not the 77 that pair 2 writes into slot 100 afterwards.
        const float first_scale = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(0, first_dst_sub, c, false)));
        const float first_zp = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(0, first_dst_sub, c, true)));
        ASSERT_NEAR(first_scale, first_expected_range / 255.0f, 0.01f) << "channel " << c;

        for (size_t r = 0; r < cm_sub_block_size; r++) {
            const float expected = (r == first_dst_row) ? relayed_value : static_cast<float>(30 + 5 * r);
            const float dequant = (static_cast<float>(key_out[cm_k_off(0, first_dst_sub * cm_sub_block_size + r, c)]) - first_zp) * first_scale;
            ASSERT_NEAR(dequant, expected, 0.3f) << "pair 1 row " << r << " channel " << c;
        }

        // ---- pair 2's dst sub-block: the same sub-block pair 1 sourced from, now requantized
        // with row 4 replaced by the value that came from block 1.
        const float second_scale = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(0, first_src_sub, c, false)));
        const float second_zp = static_cast<float>(read_fp16_from_u8_vector(key_out, cm_k_comp_off(0, first_src_sub, c, true)));
        ASSERT_NEAR(second_scale, second_expected_range / 255.0f, 0.01f) << "channel " << c;

        for (size_t r = 0; r < cm_sub_block_size; r++) {
            const float expected = (r == first_src_row) ? incoming_value : static_cast<float>(40 + 4 * r);
            const float dequant = (static_cast<float>(key_out[cm_k_off(0, first_src_sub * cm_sub_block_size + r, c)]) - second_zp) * second_scale;
            ASSERT_NEAR(dequant, expected, 0.3f) << "pair 2 row " << r << " channel " << c;
        }

        // ---- the source sub-block in block 1 is only read, never written.
        for (size_t r = 0; r < cm_sub_block_size; r++) {
            const size_t slot = 2 * cm_sub_block_size + r;
            ASSERT_EQ(key_out[cm_k_off(1, slot, c)], key_ref[cm_k_off(1, slot, c)]) << "block 1 sub 2 row " << r;
        }
        for (bool is_zp : {false, true}) {
            const auto off = cm_k_comp_off(1, 2, c, is_zp);
            ASSERT_EQ(key_out[off], key_ref[off]) << "block 1 sub 2 is_zp " << is_zp;
            ASSERT_EQ(key_out[off + 1], key_ref[off + 1]);
        }
    }

    // V is copied verbatim, so both hops are exact: slot 20 ends up with slot 100's entry bytes,
    // and slot 100 with block 1's, scale and zp included.
    for (size_t v = 0; v < cm_v_head_size; v++) {
        ASSERT_EQ(value_out[cm_v_off(0, first_dst, v)], value_ref[cm_v_off(0, first_src, v)]) << "pair 1 v " << v;
        ASSERT_EQ(value_out[cm_v_off(0, second_dst, v)], value_ref[cm_v_off(1, second_src_slot, v)]) << "pair 2 v " << v;
    }
    for (bool is_zp : {false, true}) {
        ASSERT_EQ(value_out[cm_v_comp_off(0, first_dst, is_zp)], value_ref[cm_v_comp_off(0, first_src, is_zp)]);
        ASSERT_EQ(value_out[cm_v_comp_off(0, first_dst, is_zp) + 1], value_ref[cm_v_comp_off(0, first_src, is_zp) + 1]);
        ASSERT_EQ(value_out[cm_v_comp_off(0, second_dst, is_zp)], value_ref[cm_v_comp_off(1, second_src_slot, is_zp)]);
        ASSERT_EQ(value_out[cm_v_comp_off(0, second_dst, is_zp) + 1], value_ref[cm_v_comp_off(1, second_src_slot, is_zp) + 1]);
    }
}
