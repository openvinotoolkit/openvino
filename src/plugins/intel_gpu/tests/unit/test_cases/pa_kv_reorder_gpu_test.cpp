// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
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

// The K page is token-major ([.., block_size, k_head_size]: a token's head dims are contiguous) on every device
// (paged_attention.hpp). page_head_size sizes the PAGE and defaults to k_head_size; a BY_TOKEN compressed cache passes its
// adjusted head size there while k_head_size stays the DATA row pitch (the scale/zp arrays trail the data region and are
// addressed by key_comp_byte_offset, not here).
size_t key_offset(size_t block,
                  size_t head,
                  size_t k,
                  size_t token,
                  size_t kv_heads,
                  size_t k_head_size,
                  size_t block_size,
                  size_t page_head_size = 0) {
    const size_t phs = page_head_size != 0 ? page_head_size : k_head_size;
    const size_t page_base = block * kv_heads * phs * block_size + head * phs * block_size;
    return page_base + token * k_head_size + k;
}

// The BY_CHANNEL K page: block_size rows of codes (k_head_size bytes; INT4 k_head_size / 2, channel 2b in the low nibble of
// byte b and 2b + 1 in the high one), then one (inv_scale, zp) f16 pair per channel.
size_t key_by_channel_row_bytes(size_t k_head_size, bool int4) {
    return int4 ? k_head_size / 2 : k_head_size;
}

size_t key_by_channel_page_base(size_t block, size_t head, size_t kv_heads, size_t k_head_size, size_t block_size, bool int4) {
    return (block * kv_heads + head) * (key_by_channel_row_bytes(k_head_size, int4) * block_size + 2 * sizeof(ov::float16) * k_head_size);
}

// The byte holding channel `channel` of token `token` (INT4: the byte of its channel pair).
size_t key_by_channel_offset(size_t block, size_t head, size_t channel, size_t token, size_t kv_heads, size_t k_head_size, size_t block_size, bool int4) {
    return key_by_channel_page_base(block, head, kv_heads, k_head_size, block_size, int4) + token * key_by_channel_row_bytes(k_head_size, int4) +
           (int4 ? channel / 2 : channel);
}

// The byte offset of channel `channel`'s inv_scale; its zp follows it.
size_t key_by_channel_comp_offset(size_t block, size_t head, size_t channel, size_t kv_heads, size_t k_head_size, size_t block_size, bool int4) {
    return key_by_channel_page_base(block, head, kv_heads, k_head_size, block_size, int4) + block_size * key_by_channel_row_bytes(k_head_size, int4) +
           2 * sizeof(ov::float16) * channel;
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

// Builds a pa_kv_reorder over a BY_TOKEN compressed cache whose K is declared `cache_dt`; the pages are the i8 ones
// (the token-major K page has no 4-bit BY_TOKEN form), so only the declared precision differs between calls.
void build_by_token_compressed_reorder(data_types cache_dt) {
    auto& engine = get_test_engine();

    constexpr size_t blocks_num = 2;
    constexpr size_t kv_heads = 1;
    constexpr size_t k_head_size = 32;
    constexpr size_t v_head_size = 32;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 2;
    constexpr size_t block_size = cldnn::paged_attention::block_size;

    auto key_cache_mem =
        engine.allocate_memory({ov::PartialShape{blocks_num, kv_heads, block_size, k_head_size + scales_zp_size}, data_types::i8, format::bfyx});
    auto value_cache_mem =
        engine.allocate_memory({ov::PartialShape{blocks_num, kv_heads, block_size, v_head_size + scales_zp_size}, data_types::i8, format::bfyx});
    auto index_layout = [](int64_t n) {
        return layout{ov::PartialShape{n}, data_types::i32, format::bfyx};
    };

    topology topo;
    topo.add(mutable_data("key_cache", key_cache_mem));
    topo.add(mutable_data("value_cache", value_cache_mem));
    topo.add(input_layout("block_indices", index_layout(2)));
    topo.add(input_layout("block_indices_begins", index_layout(2)));
    topo.add(input_layout("block_update_indices", index_layout(4)));
    topo.add(input_layout("block_update_indices_begins", index_layout(2)));

    auto pa_reorder = pa_kv_reorder("pa_kv_reorder",
                                    {input_info("key_cache"),
                                     input_info("value_cache"),
                                     input_info("block_indices"),
                                     input_info("block_indices_begins"),
                                     input_info("block_update_indices"),
                                     input_info("block_update_indices_begins")});
    pa_reorder.kv_heads_num = kv_heads;
    pa_reorder.adjusted_k_head_size = k_head_size + scales_zp_size;
    pa_reorder.adjusted_paged_attention_block_size = block_size;
    pa_reorder.adjusted_v_head_size = v_head_size + scales_zp_size;
    pa_reorder.cache_dt = cache_dt;
    pa_reorder.is_kv_compressed = true;
    pa_reorder.scales_zp_size = scales_zp_size;
    topo.add(pa_reorder);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);
}

// A 4-bit K cache is BY_CHANNEL only: PA_KV_reorder::validate_impl() refuses a 4-bit BY_TOKEN one, so building the op
// fails, while the same op declared i8 builds.
void expect_int4_by_token_key_refused(data_types cache_dt) {
    ASSERT_NO_THROW(build_by_token_compressed_reorder(data_types::i8));
    try {
        build_by_token_compressed_reorder(cache_dt);
        FAIL() << "a 4-bit BY_TOKEN K cache built";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("pa_kv_reorder"), std::string::npos) << e.what();
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

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, k_head_size}, data_types::f16, format::bfyx};
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

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, k_head_size}, data_types::f16, format::bfyx};
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

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_k_head_size}, data_types::i8, format::bfyx};
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
        const auto src0 = key_cache_ref[key_offset(0, 0, k, 0, kv_heads, k_head_size, block_size, adjusted_k_head_size)];
        const auto dst17 = key_ptr[key_offset(1, 0, k, 1, kv_heads, k_head_size, block_size, adjusted_k_head_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = key_cache_ref[key_offset(0, 0, k, 15, kv_heads, k_head_size, block_size, adjusted_k_head_size)];
        const auto dst16 = key_ptr[key_offset(1, 0, k, 0, kv_heads, k_head_size, block_size, adjusted_k_head_size)];
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

TEST(pa_kv_reorder_gpu, compressed_u4_by_token_key_is_refused) {
    expect_int4_by_token_key_refused(data_types::u4);
}

TEST(pa_kv_reorder_gpu, compressed_i4_by_token_key_is_refused) {
    expect_int4_by_token_key_refused(data_types::i4);
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

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, block_size, adjusted_k_head_size}, data_types::i8, format::bfyx};
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
        ASSERT_EQ(key_ptr[key_offset(0, 0, k, 3, kv_heads, k_head_size, block_size, adjusted_k_head_size)],
                  key_cache_ref[key_offset(0, 0, k, 1, kv_heads, k_head_size, block_size, adjusted_k_head_size)]);
        ASSERT_EQ(key_ptr[key_offset(2, 0, k, 4, kv_heads, k_head_size, block_size, adjusted_k_head_size)],
                  key_cache_ref[key_offset(2, 0, k, 2, kv_heads, k_head_size, block_size, adjusted_k_head_size)]);

        ASSERT_EQ(key_ptr[key_offset(1, 0, k, 4, kv_heads, k_head_size, block_size, adjusted_k_head_size)],
                  key_cache_ref[key_offset(1, 0, k, 4, kv_heads, k_head_size, block_size, adjusted_k_head_size)]);
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

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, adjusted_paged_attention_block_size, k_head_size}, data_types::i8, format::bfyx};
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
                    key_cache_ref[key_by_channel_offset(b, h, k, t, kv_heads, k_head_size, block_size, false)] =
                        static_cast<int8_t>(static_cast<int>(10 * b + 3 * k + t) - 32);
                }

                const size_t comp_scale_byte_offset = key_by_channel_comp_offset(b, h, k, kv_heads, k_head_size, block_size, false);
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
        const auto src0_q = key_cache_ref[key_by_channel_offset(0, 0, k, 0, kv_heads, k_head_size, block_size, false)];
        const auto src15_q = key_cache_ref[key_by_channel_offset(0, 0, k, 15, kv_heads, k_head_size, block_size, false)];

        const size_t comp_scale_byte_offset = key_by_channel_comp_offset(1, 0, k, kv_heads, k_head_size, block_size, false);
        const size_t comp_zp_byte_offset = comp_scale_byte_offset + sizeof(ov::float16);
        const float dst_scale_inv = static_cast<float>(read_fp16_from_byte_buffer(key_ptr, comp_scale_byte_offset));
        const float dst_zp = static_cast<float>(read_fp16_from_byte_buffer(key_ptr, comp_zp_byte_offset));

        const float dst17_dequant =
            (static_cast<float>(key_ptr[key_by_channel_offset(1, 0, k, 1, kv_heads, k_head_size, block_size, false)]) - dst_zp) * dst_scale_inv;
        const float dst16_dequant =
            (static_cast<float>(key_ptr[key_by_channel_offset(1, 0, k, 0, kv_heads, k_head_size, block_size, false)]) - dst_zp) * dst_scale_inv;

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
    // u4 BY_CHANNEL K page: block_size rows of k_head_size / 2 bytes (channel 2b in the low nibble of byte b, 2b + 1 in the
    // high one), then one (inv_scale, zp) f16 pair per channel: [blocks, heads, block_size / 2 + 4, k_head_size] bytes.
    constexpr size_t packed_block_size = block_size / 2;
    constexpr size_t scales_zp_size = sizeof(ov::float16) * 2;
    constexpr size_t adjusted_paged_attention_block_size = packed_block_size + scales_zp_size;
    // u4 V page: block_size rows of packed_v_head_size bytes (the writer's row, a multiple of 16), then the per-token scale
    // and zp f16 arrays. The reorder copies whole rows, so the nibble order inside one does not matter here.
    constexpr size_t packed_v_head_size = ((v_head_size / 2 + subgroup_size - 1) / subgroup_size) * subgroup_size;
    constexpr size_t adjusted_v_head_size = packed_v_head_size + scales_zp_size;

    auto key_cache_layout = layout{ov::PartialShape{blocks_num, kv_heads, adjusted_paged_attention_block_size, k_head_size}, data_types::u8, format::bfyx};
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

    // Key cache: channel h of token t holds (3 * t + h + b) & 0xF; channel h's pair is (0.10 + 0.01 * h, h % 5 + 1).
    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t h = 0; h < k_head_size; h++) {
            const float scale_inv = 0.10f + 0.01f * static_cast<float>(h);
            const float zp = static_cast<float>((static_cast<int>(h) % 5) + 1);
            const size_t comp_base = key_by_channel_comp_offset(b, 0, h, kv_heads, k_head_size, block_size, true);
            write_fp16_bytes(key_cache_ref, comp_base + 0 * sizeof(ov::float16), ov::float16(scale_inv));
            write_fp16_bytes(key_cache_ref, comp_base + 1 * sizeof(ov::float16), ov::float16(zp));

            for (size_t t = 0; t < block_size; t++) {
                const uint8_t q = static_cast<uint8_t>((3 * t + h + b) & 0xF);
                uint8_t& byte = key_cache_ref[key_by_channel_offset(b, 0, h, t, kv_heads, k_head_size, block_size, true)];
                byte = (h % 2 == 0) ? static_cast<uint8_t>((byte & 0xF0u) | q) : static_cast<uint8_t>((byte & 0x0Fu) | (q << 4));
            }
        }
    }

    for (size_t b = 0; b < blocks_num; b++) {
        for (size_t t = 0; t < block_size; t++) {
            for (size_t p = 0; p < packed_v_head_size; p++) {
                value_cache_ref[value_data_offset_compressed(b, 0, t, p, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)] =
                    static_cast<uint8_t>((7 * t + 5 * p + b) % 251);
            }

            const ov::float16 scale = ov::float16(0.125f * static_cast<float>(1 + ((b + t) % 3)));
            const ov::float16 zp = ov::float16(static_cast<float>((static_cast<int>(t) % 7) - 3));
            for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
                const auto scale_off = value_comp_byte_offset(b, 0, t, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size);
                const auto zp_off = value_comp_byte_offset(b, 0, t, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size);
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
    // What ops/pa_kv_reorder.cpp sets for an f16 inference precision (unpacked sizes; the impl derives the packed ones).
    pa_reorder.kv_heads_num = kv_heads;
    pa_reorder.adjusted_k_head_size = k_head_size;
    pa_reorder.adjusted_paged_attention_block_size = block_size + scales_zp_size;
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

    auto read_nibble = [&](const auto& buffer, size_t block, size_t h, size_t token) {
        const uint8_t packed = buffer[key_by_channel_offset(block, 0, h, token, kv_heads, k_head_size, block_size, true)];
        return (h % 2 == 0) ? static_cast<uint8_t>(packed & 0xF) : static_cast<uint8_t>((packed >> 4) & 0xF);
    };

    auto read_col_comp = [&](const auto& buffer, size_t block, size_t h) {
        const size_t comp_base = key_by_channel_comp_offset(block, 0, h, kv_heads, k_head_size, block_size, true);
        const uint16_t scale_bits = static_cast<uint16_t>(buffer[comp_base] | (buffer[comp_base + 1] << 8));
        const uint16_t zp_bits = static_cast<uint16_t>(buffer[comp_base + 2] | (buffer[comp_base + 3] << 8));
        return std::make_pair(static_cast<float>(ov::float16::from_bits(scale_bits)), static_cast<float>(ov::float16::from_bits(zp_bits)));
    };

    // Reorder maps src token 0 -> dst token 17 (block 1, slot 1) and src token 15 -> dst token 16 (block 1, slot 0).
    // Compare in dequantized space because the dst column is requantized end-to-end on cross-block copies.
    for (size_t h = 0; h < k_head_size; h++) {
        const auto [src_scale_inv, src_zp] = read_col_comp(key_cache_ref, 0, h);
        const auto [dst_scale_inv, dst_zp] = read_col_comp(key_ptr, 1, h);

        const float src_val0 = (static_cast<float>(read_nibble(key_cache_ref, 0, h, 0)) - src_zp) * src_scale_inv;
        const float src_val15 = (static_cast<float>(read_nibble(key_cache_ref, 0, h, 15)) - src_zp) * src_scale_inv;
        const float dst_val1 = (static_cast<float>(read_nibble(key_ptr, 1, h, 1)) - dst_zp) * dst_scale_inv;
        const float dst_val0 = (static_cast<float>(read_nibble(key_ptr, 1, h, 0)) - dst_zp) * dst_scale_inv;

        ASSERT_NEAR(dst_val1, src_val0, 1.0f);
        ASSERT_NEAR(dst_val0, src_val15, 1.0f);
    }

    for (size_t p = 0; p < packed_v_head_size; p++) {
        const auto src0 = value_cache_ref[value_data_offset_compressed(0, 0, 0, p, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)];
        const auto dst17 = value_ptr[value_data_offset_compressed(1, 0, 1, p, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst17, src0);

        const auto src15 = value_cache_ref[value_data_offset_compressed(0, 0, 15, p, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)];
        const auto dst16 = value_ptr[value_data_offset_compressed(1, 0, 0, p, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)];
        ASSERT_EQ(dst16, src15);
    }

    for (size_t byte = 0; byte < sizeof(ov::float16); byte++) {
        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 1, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 0, byte, false, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)]);
        ASSERT_EQ(value_ptr[value_comp_byte_offset(1, 0, 1, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)],
                  value_cache_ref[value_comp_byte_offset(0, 0, 0, byte, true, kv_heads, packed_v_head_size, adjusted_v_head_size, block_size)]);
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

namespace {

// pa_kv_reorder on the token-major K page of each cache kind against a host model of the reorder. A random logical cache
// (codes and (inv_scale, zp) pairs) is laid out as the plugin lays out its pages (paged_attention.hpp and
// pa_kv_cache_update_ref.cl), reordered on the GPU, read back into the logical form and compared with the same moves
// applied on the host: bit for bit where the reorder copies (f16, i8 BY_TOKEN, every V page, a u4 move inside one page),
// in dequantized values where it requantizes the destination column of a BY_CHANNEL page.
enum class lt_kind { f16, i8_by_token, i8_by_channel, u4_by_channel };

struct lt_case {
    lt_kind kind;
    // Dynamic index inputs make the impl shape-agnostic: it compiles the kernel and picks it at dispatch.
    bool dynamic_indices;
};

constexpr size_t lt_blocks = 4;
constexpr size_t lt_heads = 2;
constexpr size_t lt_k = 64;
constexpr size_t lt_v = 64;
constexpr size_t lt_bs = cldnn::paged_attention::block_size;
constexpr size_t lt_pv = lt_v / 2;  // the u4 V row, already a multiple of 16

// Two sequences on four pages: cross-page moves (requantizing the destination column of a BY_CHANNEL page), a move
// inside one page, and one per KV head (two heads).
// seq 0 (pages 0, 2): 0 -> 17, 15 -> 16, 3 -> 5 (same page), 30 -> 6; seq 1 (pages 3, 1): 20 -> 2, 1 -> 30.
const std::vector<int32_t> lt_block_indices{0, 2, 3, 1};
const std::vector<int32_t> lt_block_indices_begins{0, 2, 4};
const std::vector<int32_t> lt_updates{0, 17, 15, 16, 3, 5, 30, 6, 20, 2, 1, 30};
const std::vector<int32_t> lt_update_begins{0, 4, 6};

struct lt_cache {
    std::vector<uint16_t> k;       // [block][head][channel][token]: f16 bits, an i8 byte or a nibble
    std::vector<uint16_t> k_comp;  // [block][head][n][2]: n = token (BY_TOKEN) or channel (BY_CHANNEL)
    std::vector<uint16_t> v;       // [block][head][token][dim]
    std::vector<uint16_t> v_comp;  // [block][head][token][2]
};

struct lt_loc {
    size_t byte;
    int shift;  // nibble shift, or -1 for a whole element
};

bool lt_compressed(lt_kind kind) {
    return kind != lt_kind::f16;
}
bool lt_by_channel(lt_kind kind) {
    return kind == lt_kind::i8_by_channel || kind == lt_kind::u4_by_channel;
}
bool lt_int4(lt_kind kind) {
    return kind == lt_kind::u4_by_channel;
}
size_t lt_k_comp_n(lt_kind kind) {
    return lt_by_channel(kind) ? lt_k : lt_bs;
}
// The K page's [rows, row length] as materialized: [block_size (+ 4 comp rows for BY_CHANNEL; u4 halves the data rows),
// k_head_size (+ 4 for BY_TOKEN)].
int64_t lt_k_block_dim(lt_kind kind) {
    return static_cast<int64_t>(lt_by_channel(kind) ? (lt_int4(kind) ? lt_bs / 2 : lt_bs) + 4 : lt_bs);
}
int64_t lt_k_head_dim(lt_kind kind) {
    return static_cast<int64_t>(lt_compressed(kind) && !lt_by_channel(kind) ? lt_k + 4 : lt_k);
}
size_t lt_k_page(lt_kind kind) {
    return static_cast<size_t>(lt_k_block_dim(kind) * lt_k_head_dim(kind)) * (kind == lt_kind::f16 ? 2 : 1);
}
int64_t lt_v_row(lt_kind kind) {
    return static_cast<int64_t>(kind == lt_kind::f16 ? lt_v : (lt_int4(kind) ? lt_pv + 4 : lt_v + 4));
}
size_t lt_v_page(lt_kind kind) {
    return lt_bs * static_cast<size_t>(lt_v_row(kind)) * (kind == lt_kind::f16 ? 2 : 1);
}

// A token's channels are contiguous; u4 packs two adjacent CHANNELS per byte (channel 2b low, 2b + 1 high).
lt_loc lt_k_loc(lt_kind kind, size_t b, size_t h, size_t c, size_t t) {
    const size_t page = (b * lt_heads + h) * lt_k_page(kind);
    switch (kind) {
    case lt_kind::f16:
        return {page + 2 * (t * lt_k + c), -1};
    case lt_kind::i8_by_token:
    case lt_kind::i8_by_channel:
        return {page + t * lt_k + c, -1};
    case lt_kind::u4_by_channel:
        return {page + t * (lt_k / 2) + c / 2, static_cast<int>(4 * (c % 2))};
    }
    return {0, -1};
}

// Byte offsets of the (inv_scale, zp) f16 pair of token / channel n.
std::pair<size_t, size_t> lt_k_comp_loc(lt_kind kind, size_t b, size_t h, size_t n) {
    const size_t page = (b * lt_heads + h) * lt_k_page(kind);
    switch (kind) {
    case lt_kind::i8_by_token: {
        const size_t scale = page + lt_bs * lt_k + 2 * n;  // two trailing per-token arrays
        return {scale, scale + 2 * lt_bs};
    }
    case lt_kind::i8_by_channel: {
        const size_t scale = page + lt_bs * lt_k + 4 * n;  // one pair per channel after the data
        return {scale, scale + 2};
    }
    case lt_kind::u4_by_channel: {
        const size_t scale = page + lt_bs / 2 * lt_k + 4 * n;
        return {scale, scale + 2};
    }
    default:
        return {0, 0};
    }
}

// The u4 V page has split nibbles (byte b = dims b and b + PV) and no inline comp.
lt_loc lt_v_loc(lt_kind kind, size_t b, size_t h, size_t t, size_t d) {
    const size_t page = (b * lt_heads + h) * lt_v_page(kind);
    if (kind == lt_kind::f16)
        return {page + 2 * (t * lt_v + d), -1};
    if (!lt_int4(kind))
        return {page + t * lt_v + d, -1};
    return {page + t * lt_pv + d % lt_pv, static_cast<int>(4 * (d / lt_pv))};
}

std::pair<size_t, size_t> lt_v_comp_loc(lt_kind kind, size_t b, size_t h, size_t t) {
    const size_t page = (b * lt_heads + h) * lt_v_page(kind);
    const size_t scale = page + lt_bs * (lt_int4(kind) ? lt_pv : lt_v) + 2 * t;
    return {scale, scale + 2 * lt_bs};
}

void lt_put(std::vector<uint8_t>& buf, lt_loc loc, uint16_t code, bool f16) {
    if (loc.shift >= 0) {
        buf[loc.byte] = static_cast<uint8_t>((buf[loc.byte] & ~(0xF << loc.shift)) | ((code & 0xF) << loc.shift));
    } else if (f16) {
        buf[loc.byte] = static_cast<uint8_t>(code & 0xFF);
        buf[loc.byte + 1] = static_cast<uint8_t>(code >> 8);
    } else {
        buf[loc.byte] = static_cast<uint8_t>(code & 0xFF);
    }
}

uint16_t lt_get(const std::vector<uint8_t>& buf, lt_loc loc, bool f16) {
    if (loc.shift >= 0)
        return static_cast<uint16_t>((buf[loc.byte] >> loc.shift) & 0xF);
    if (f16)
        return static_cast<uint16_t>(buf[loc.byte] | (buf[loc.byte + 1] << 8));
    return buf[loc.byte];
}

size_t lt_k_idx(size_t b, size_t h, size_t c, size_t t) {
    return ((b * lt_heads + h) * lt_k + c) * lt_bs + t;
}
size_t lt_v_idx(size_t b, size_t h, size_t t, size_t d) {
    return ((b * lt_heads + h) * lt_bs + t) * lt_v + d;
}
size_t lt_k_comp_idx(lt_kind kind, size_t b, size_t h, size_t n) {
    return ((b * lt_heads + h) * lt_k_comp_n(kind) + n) * 2;
}
size_t lt_v_comp_idx(size_t b, size_t h, size_t t) {
    return ((b * lt_heads + h) * lt_bs + t) * 2;
}

lt_cache lt_random(lt_kind kind, uint32_t seed) {
    std::mt19937 rng(seed);
    auto uniform = [&](float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    auto code = [&]() -> uint16_t {
        if (kind == lt_kind::f16)
            return ov::float16(uniform(-4.0f, 4.0f)).to_bits();
        return static_cast<uint16_t>(rng() % (lt_int4(kind) ? 16 : 256));
    };
    auto pair = [&](uint16_t* dst) {
        dst[0] = ov::float16(uniform(0.02f, 0.1f)).to_bits();
        dst[1] = ov::float16(lt_int4(kind) ? static_cast<float>(rng() % 16) : uniform(-20.0f, 20.0f)).to_bits();
    };
    lt_cache c;
    c.k.resize(lt_blocks * lt_heads * lt_k * lt_bs);
    c.v.resize(lt_blocks * lt_heads * lt_bs * lt_v);
    for (auto& x : c.k)
        x = code();
    for (auto& x : c.v)
        x = code();
    if (lt_compressed(kind)) {
        c.k_comp.resize(lt_blocks * lt_heads * lt_k_comp_n(kind) * 2);
        c.v_comp.resize(lt_blocks * lt_heads * lt_bs * 2);
        for (size_t i = 0; i < c.k_comp.size(); i += 2)
            pair(&c.k_comp[i]);
        for (size_t i = 0; i < c.v_comp.size(); i += 2)
            pair(&c.v_comp[i]);
    }
    return c;
}

// Moves between the logical cache and the page bytes (store = true: logical -> bytes).
void lt_transfer(lt_kind kind, bool store, lt_cache& c, std::vector<uint8_t>& kb, std::vector<uint8_t>& vb) {
    const bool f16 = kind == lt_kind::f16;
    if (store) {
        kb.assign(lt_blocks * lt_heads * lt_k_page(kind), 0);
        vb.assign(lt_blocks * lt_heads * lt_v_page(kind), 0);
    } else {
        c.k.assign(lt_blocks * lt_heads * lt_k * lt_bs, 0);
        c.v.assign(lt_blocks * lt_heads * lt_bs * lt_v, 0);
        c.k_comp.assign(lt_compressed(kind) ? lt_blocks * lt_heads * lt_k_comp_n(kind) * 2 : 0, 0);
        c.v_comp.assign(lt_compressed(kind) ? lt_blocks * lt_heads * lt_bs * 2 : 0, 0);
    }
    auto move = [&](std::vector<uint8_t>& buf, lt_loc loc, uint16_t& x, bool wide) {
        if (store)
            lt_put(buf, loc, x, wide);
        else
            x = lt_get(buf, loc, wide);
    };
    for (size_t b = 0; b < lt_blocks; b++) {
        for (size_t h = 0; h < lt_heads; h++) {
            for (size_t c_ = 0; c_ < lt_k; c_++)
                for (size_t t = 0; t < lt_bs; t++)
                    move(kb, lt_k_loc(kind, b, h, c_, t), c.k[lt_k_idx(b, h, c_, t)], f16);
            for (size_t t = 0; t < lt_bs; t++)
                for (size_t d = 0; d < lt_v; d++)
                    move(vb, lt_v_loc(kind, b, h, t, d), c.v[lt_v_idx(b, h, t, d)], f16);
            if (!lt_compressed(kind))
                continue;
            for (size_t n = 0; n < lt_k_comp_n(kind); n++) {
                const auto loc = lt_k_comp_loc(kind, b, h, n);
                const size_t i = lt_k_comp_idx(kind, b, h, n);
                move(kb, {loc.first, -1}, c.k_comp[i], true);
                move(kb, {loc.second, -1}, c.k_comp[i + 1], true);
            }
            for (size_t t = 0; t < lt_bs; t++) {
                const auto loc = lt_v_comp_loc(kind, b, h, t);
                const size_t i = lt_v_comp_idx(b, h, t);
                move(vb, {loc.first, -1}, c.v_comp[i], true);
                move(vb, {loc.second, -1}, c.v_comp[i + 1], true);
            }
        }
    }
}

void lt_copy_in(cldnn::memory::ptr mem, const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(mem->size(), bytes.size());
    cldnn::mem_lock<uint8_t, mem_lock_type::write> lock(mem, get_test_stream());
    std::memcpy(lock.data(), bytes.data(), bytes.size());
}

void lt_copy_out(cldnn::memory::ptr mem, cldnn::stream& stream, std::vector<uint8_t>& bytes) {
    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock(mem, stream);
    bytes.assign(lock.data(), lock.data() + mem->size());
}

void lt_run(lt_kind kind, bool dynamic_indices, std::vector<uint8_t>& kb, std::vector<uint8_t>& vb) {
    auto& engine = get_test_engine();
    const bool compressed = lt_compressed(kind);
    const bool by_channel = lt_by_channel(kind);
    const auto mem_dt = kind == lt_kind::f16 ? data_types::f16 : (lt_int4(kind) ? data_types::u8 : data_types::i8);
    const int64_t blocks = static_cast<int64_t>(lt_blocks);
    const int64_t heads = static_cast<int64_t>(lt_heads);
    const auto k_shape = ov::PartialShape{blocks, heads, lt_k_block_dim(kind), lt_k_head_dim(kind)};
    const auto v_shape = ov::PartialShape{blocks, heads, static_cast<int64_t>(lt_bs), lt_v_row(kind)};
    auto key_mem = engine.allocate_memory({k_shape, mem_dt, format::bfyx});
    auto value_mem = engine.allocate_memory({v_shape, mem_dt, format::bfyx});
    lt_copy_in(key_mem, kb);
    lt_copy_in(value_mem, vb);

    auto index_layout = [&](size_t n) {
        return dynamic_indices ? layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx}
                               : layout{ov::PartialShape{static_cast<int64_t>(n)}, data_types::i32, format::bfyx};
    };
    auto index_mem = [&](const std::vector<int32_t>& values) {
        auto mem = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(values.size())}, data_types::i32, format::bfyx});
        set_values<int32_t>(mem, values);
        return mem;
    };

    topology topo;
    topo.add(mutable_data("key_cache", key_mem));
    topo.add(mutable_data("value_cache", value_mem));
    topo.add(input_layout("block_indices", index_layout(lt_block_indices.size())));
    topo.add(input_layout("block_indices_begins", index_layout(lt_block_indices_begins.size())));
    topo.add(input_layout("block_update_indices", index_layout(lt_updates.size())));
    topo.add(input_layout("block_update_indices_begins", index_layout(lt_update_begins.size())));

    auto prim = pa_kv_reorder("pa_kv_reorder",
                              {input_info("key_cache"),
                               input_info("value_cache"),
                               input_info("block_indices"),
                               input_info("block_indices_begins"),
                               input_info("block_update_indices"),
                               input_info("block_update_indices_begins")});
    // What ops/pa_kv_reorder.cpp sets for an f16 inference precision.
    const size_t scales_zp_size = compressed ? 4 : 0;
    prim.kv_heads_num = lt_heads;
    prim.adjusted_k_head_size = static_cast<size_t>(lt_k_head_dim(kind));
    prim.adjusted_paged_attention_block_size = lt_bs + (by_channel ? scales_zp_size : 0);
    prim.adjusted_v_head_size = lt_v + scales_zp_size;
    prim.cache_dt = kind == lt_kind::f16 ? data_types::f16 : (lt_int4(kind) ? data_types::u4 : data_types::i8);
    prim.is_kv_compressed = compressed;
    prim.is_key_by_channel = by_channel;
    prim.scales_zp_size = scales_zp_size;
    topo.add(prim);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);
    network->set_input_data("block_indices", index_mem(lt_block_indices));
    network->set_input_data("block_indices_begins", index_mem(lt_block_indices_begins));
    network->set_input_data("block_update_indices", index_mem(lt_updates));
    network->set_input_data("block_update_indices_begins", index_mem(lt_update_begins));
    network->execute();
    network->get_stream().finish();

    lt_copy_out(key_mem, network->get_stream(), kb);
    lt_copy_out(value_mem, network->get_stream(), vb);
}

float lt_f16(uint16_t bits) {
    return static_cast<float>(ov::float16::from_bits(bits));
}

// The dequantized K value of channel c, token t (BY_CHANNEL: the channel's pair; i8 codes are signed).
float lt_k_value(lt_kind kind, const lt_cache& c, size_t b, size_t h, size_t ch, size_t t) {
    const uint16_t code = c.k[lt_k_idx(b, h, ch, t)];
    const size_t i = lt_k_comp_idx(kind, b, h, ch);
    const float q = lt_int4(kind) ? static_cast<float>(code) : static_cast<float>(static_cast<int8_t>(code & 0xFF));
    return (q - lt_f16(c.k_comp[i + 1])) * lt_f16(c.k_comp[i]);
}

// The host model of the reorder (pa_kv_cache_reorder_ref.cl): per sequence, the moves in order. A move copies the token's
// V row and V pair, and its K row and (BY_TOKEN) K pair. On a BY_CHANNEL page it writes the source's dequantized value into
// the destination column and requantizes that whole column (every channel; i8 also within one page, u4 copies the byte
// there). `k_val` holds the dequantized BY_CHANNEL K, `k_err` the error each value may have picked up (0.6 of a
// quantization step per requantization: half a step of rounding plus f16 arithmetic), `requantized` the pages whose K
// pairs changed.
struct lt_model {
    lt_cache exact;
    std::vector<float> k_val;
    std::vector<float> k_err;
    std::vector<bool> requantized;
};

// The step the kernel's requantization picks for a column (requantize_and_store_by_channel_tm /
// requantize_by_channel_int4_tm).
float lt_requant_step(lt_kind kind, float min_v, float max_v) {
    float range = (lt_int4(kind) && max_v == min_v) ? 0.001f : max_v - min_v;
    const float min_range = std::fabs(max_v * 0.1f);
    if (range <= min_range)
        range += std::max(1.0f, min_range);
    return range / (lt_int4(kind) ? 15.0f : 255.0f);
}

lt_model lt_reference(lt_kind kind, const lt_cache& in) {
    lt_model m{in, {}, {}, std::vector<bool>(lt_blocks * lt_heads, false)};
    const bool by_channel = lt_by_channel(kind);
    if (by_channel) {
        m.k_val.resize(in.k.size());
        m.k_err.assign(in.k.size(), 0.0f);
        for (size_t b = 0; b < lt_blocks; b++)
            for (size_t h = 0; h < lt_heads; h++)
                for (size_t ch = 0; ch < lt_k; ch++)
                    for (size_t t = 0; t < lt_bs; t++)
                        m.k_val[lt_k_idx(b, h, ch, t)] = lt_k_value(kind, in, b, h, ch, t);
    }
    for (size_t seq = 0; seq + 1 < lt_update_begins.size(); seq++) {
        const size_t pages = static_cast<size_t>(lt_block_indices_begins[seq + 1] - lt_block_indices_begins[seq]);
        for (int32_t pos = lt_update_begins[seq]; pos < lt_update_begins[seq + 1]; pos++) {
            const size_t src = static_cast<size_t>(lt_updates[2 * pos]);
            const size_t dst = static_cast<size_t>(lt_updates[2 * pos + 1]);
            if (src / lt_bs >= pages || dst / lt_bs >= pages)
                continue;
            const size_t sb = static_cast<size_t>(lt_block_indices[lt_block_indices_begins[seq] + src / lt_bs]), ss = src % lt_bs;
            const size_t db = static_cast<size_t>(lt_block_indices[lt_block_indices_begins[seq] + dst / lt_bs]), ds = dst % lt_bs;
            const bool requantize = by_channel && (sb != db || !lt_int4(kind));
            for (size_t h = 0; h < lt_heads; h++) {
                for (size_t d = 0; d < lt_v; d++)
                    m.exact.v[lt_v_idx(db, h, ds, d)] = m.exact.v[lt_v_idx(sb, h, ss, d)];
                if (lt_compressed(kind)) {
                    m.exact.v_comp[lt_v_comp_idx(db, h, ds)] = m.exact.v_comp[lt_v_comp_idx(sb, h, ss)];
                    m.exact.v_comp[lt_v_comp_idx(db, h, ds) + 1] = m.exact.v_comp[lt_v_comp_idx(sb, h, ss) + 1];
                }
                if (!by_channel) {
                    for (size_t ch = 0; ch < lt_k; ch++)
                        m.exact.k[lt_k_idx(db, h, ch, ds)] = m.exact.k[lt_k_idx(sb, h, ch, ss)];
                    if (lt_compressed(kind)) {
                        m.exact.k_comp[lt_k_comp_idx(kind, db, h, ds)] = m.exact.k_comp[lt_k_comp_idx(kind, sb, h, ss)];
                        m.exact.k_comp[lt_k_comp_idx(kind, db, h, ds) + 1] = m.exact.k_comp[lt_k_comp_idx(kind, sb, h, ss) + 1];
                    }
                    continue;
                }
                for (size_t ch = 0; ch < lt_k; ch++) {
                    m.k_val[lt_k_idx(db, h, ch, ds)] = m.k_val[lt_k_idx(sb, h, ch, ss)];
                    m.k_err[lt_k_idx(db, h, ch, ds)] = m.k_err[lt_k_idx(sb, h, ch, ss)];
                    if (!requantize) {
                        m.exact.k[lt_k_idx(db, h, ch, ds)] = m.exact.k[lt_k_idx(sb, h, ch, ss)];
                        continue;
                    }
                    float min_v = std::numeric_limits<float>::max(), max_v = std::numeric_limits<float>::lowest();
                    for (size_t t = 0; t < lt_bs; t++) {
                        min_v = std::min(min_v, m.k_val[lt_k_idx(db, h, ch, t)]);
                        max_v = std::max(max_v, m.k_val[lt_k_idx(db, h, ch, t)]);
                    }
                    const float step = lt_requant_step(kind, min_v, max_v);
                    for (size_t t = 0; t < lt_bs; t++)
                        m.k_err[lt_k_idx(db, h, ch, t)] += 0.6f * step;
                }
                if (requantize)
                    m.requantized[db * lt_heads + h] = true;
            }
        }
    }
    return m;
}

size_t lt_mismatches(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); i++)
        n += a[i] != b[i];
    return n + (a.size() != b.size());
}

class pa_kv_reorder_layout_test : public ::testing::TestWithParam<lt_case> {};

}  // namespace

TEST_P(pa_kv_reorder_layout_test, matches_reference) {
    const auto p = GetParam();

    auto in = lt_random(p.kind, 20261101);
    std::vector<uint8_t> kb, vb;
    lt_transfer(p.kind, true, in, kb, vb);
    lt_run(p.kind, p.dynamic_indices, kb, vb);
    lt_cache out;
    lt_transfer(p.kind, false, out, kb, vb);
    const auto ref = lt_reference(p.kind, in);

    // The reorder did run: seq 0's first move puts token 0 of page 0 at token 1 of page 2.
    for (size_t h = 0; h < lt_heads; h++)
        for (size_t d = 0; d < lt_v; d++)
            ASSERT_EQ(out.v[lt_v_idx(2, h, 1, d)], in.v[lt_v_idx(0, h, 0, d)]);
    ASSERT_NE(lt_mismatches(out.k, in.k), 0u);

    EXPECT_EQ(lt_mismatches(out.v, ref.exact.v), 0u) << "V codes";
    EXPECT_EQ(lt_mismatches(out.v_comp, ref.exact.v_comp), 0u) << "V (inv_scale, zp)";
    if (!lt_by_channel(p.kind)) {
        EXPECT_EQ(lt_mismatches(out.k, ref.exact.k), 0u) << "K codes";
        EXPECT_EQ(lt_mismatches(out.k_comp, ref.exact.k_comp), 0u) << "K (inv_scale, zp)";
        return;
    }

    // BY_CHANNEL K: a page no requantization touched keeps its bytes; elsewhere every value is the model's within the
    // error its requantizations allow (k_err), plus the f16 rounding of the dequantization.
    size_t checked = 0;
    for (size_t b = 0; b < lt_blocks; b++) {
        for (size_t h = 0; h < lt_heads; h++) {
            const bool requantized = ref.requantized[b * lt_heads + h];
            for (size_t ch = 0; ch < lt_k; ch++) {
                const size_t ci = lt_k_comp_idx(p.kind, b, h, ch);
                if (!requantized) {
                    ASSERT_EQ(out.k_comp[ci], ref.exact.k_comp[ci]) << "page " << b << " head " << h << " channel " << ch;
                    ASSERT_EQ(out.k_comp[ci + 1], ref.exact.k_comp[ci + 1]) << "page " << b << " head " << h << " channel " << ch;
                }
                for (size_t t = 0; t < lt_bs; t++) {
                    const size_t i = lt_k_idx(b, h, ch, t);
                    if (!requantized) {
                        ASSERT_EQ(out.k[i], ref.exact.k[i]) << "page " << b << " head " << h << " channel " << ch << " token " << t;
                        continue;
                    }
                    const float tol = ref.k_err[i] + 2e-3f * std::max(1.0f, std::fabs(ref.k_val[i]));
                    ASSERT_NEAR(lt_k_value(p.kind, out, b, h, ch, t), ref.k_val[i], tol)
                        << "page " << b << " head " << h << " channel " << ch << " token " << t;
                    checked++;
                }
            }
        }
    }
    ASSERT_GT(checked, 0u);
}

INSTANTIATE_TEST_SUITE_P(smoke_pa_kv_reorder_layout,
                         pa_kv_reorder_layout_test,
                         ::testing::Values(lt_case{lt_kind::f16, false},
                                           lt_case{lt_kind::i8_by_token, false},
                                           lt_case{lt_kind::i8_by_channel, false},
                                           lt_case{lt_kind::u4_by_channel, false},
                                           lt_case{lt_kind::f16, true},
                                           lt_case{lt_kind::i8_by_token, true},
                                           lt_case{lt_kind::i8_by_channel, true},
                                           lt_case{lt_kind::u4_by_channel, true}));
