// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for cache serialization round-trip of:
//   1) kv_cache stages including the zp_concat stage (commit: "GPU: fix std::bad_function_call")
//   2) MoE prefill execution flags (commit: "GPU: serialize MoE prefill execution flags")
//
// These tests replicate the serialization logic of internal structures
// (stages_helper, moe prefill flags) without requiring a full model compilation,
// verifying that the save/load contract is correct.

#include "test_utils.h"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

using namespace cldnn;
using namespace ::tests;

// --------------------------------------------------------------------------
// kv_cache stages_helper round-trip tests
// --------------------------------------------------------------------------
// Mirrors the kv_stage enum and stages_helper from kv_cache.cpp
namespace {
enum class kv_stage_test : uint8_t { scatter_update, concat, beam_table, dq, scale_concat, zp_concat };

struct stages_helper_test {
    std::vector<kv_stage_test> stages;

    void save(BinaryOutputBuffer& ob) const {
        ob << stages.size();
        for (const auto& stage : stages) {
            ob << static_cast<uint8_t>(stage);
        }
    }

    void load(BinaryInputBuffer& ib) {
        size_t stages_size = 0;
        ib >> stages_size;
        stages.resize(stages_size);
        for (auto& stage : stages) {
            uint8_t stage_ = 0;
            ib >> stage_;
            stage = static_cast<kv_stage_test>(stage_);
        }
    }

    std::optional<size_t> try_get_index(kv_stage_test stage) const noexcept {
        if (const auto it = std::find(stages.begin(), stages.end(), stage); it != stages.end()) {
            return static_cast<size_t>(std::distance(stages.begin(), it));
        }
        return {};
    }
};
}  // namespace

// Test: KV-cache stages including zp_concat survive a binary round-trip.
// This covers the scenario where a compressed KV-cache with zero-point inputs
// has stages {concat, beam_table, dq, scale_concat, zp_concat}.
// Before the fix, the load() path did not restore the dispatch function for
// zp_concat, causing std::bad_function_call on shape change.
TEST(cache_serialization, kv_cache_stages_with_zp_concat_round_trip) {
    auto& engine = get_test_engine();

    // Simulate a compressed KV-cache with indirect + zero-points
    stages_helper_test original;
    original.stages.push_back(kv_stage_test::concat);
    original.stages.push_back(kv_stage_test::beam_table);
    original.stages.push_back(kv_stage_test::dq);
    original.stages.push_back(kv_stage_test::scale_concat);
    original.stages.push_back(kv_stage_test::zp_concat);

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    stages_helper_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.stages.size(), original.stages.size());
    for (size_t i = 0; i < original.stages.size(); ++i) {
        ASSERT_EQ(static_cast<uint8_t>(loaded.stages[i]),
                  static_cast<uint8_t>(original.stages[i]))
            << "Stage mismatch at index " << i;
    }

    // Verify zp_concat stage is present and at the correct index
    auto zp_idx = loaded.try_get_index(kv_stage_test::zp_concat);
    ASSERT_TRUE(zp_idx.has_value()) << "zp_concat stage lost during serialization";
    ASSERT_EQ(*zp_idx, 4u);

    // Verify all stages are at expected positions
    ASSERT_TRUE(loaded.try_get_index(kv_stage_test::concat).has_value());
    ASSERT_TRUE(loaded.try_get_index(kv_stage_test::beam_table).has_value());
    ASSERT_TRUE(loaded.try_get_index(kv_stage_test::dq).has_value());
    ASSERT_TRUE(loaded.try_get_index(kv_stage_test::scale_concat).has_value());
}

// Test: KV-cache stages without zp_concat (non-ZP compressed cache).
TEST(cache_serialization, kv_cache_stages_without_zp_concat_round_trip) {
    auto& engine = get_test_engine();

    stages_helper_test original;
    original.stages.push_back(kv_stage_test::concat);
    original.stages.push_back(kv_stage_test::beam_table);
    original.stages.push_back(kv_stage_test::dq);
    original.stages.push_back(kv_stage_test::scale_concat);

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    stages_helper_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.stages.size(), 4u);
    ASSERT_FALSE(loaded.try_get_index(kv_stage_test::zp_concat).has_value());
    ASSERT_TRUE(loaded.try_get_index(kv_stage_test::scale_concat).has_value());
}

// Test: scatter_update-only KV-cache stages (non-indirect, non-compressed).
TEST(cache_serialization, kv_cache_stages_scatter_only_round_trip) {
    auto& engine = get_test_engine();

    stages_helper_test original;
    original.stages.push_back(kv_stage_test::scatter_update);

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    stages_helper_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.stages.size(), 1u);
    ASSERT_TRUE(loaded.try_get_index(kv_stage_test::scatter_update).has_value());
}

// --------------------------------------------------------------------------
// MoE prefill execution flags round-trip tests
// --------------------------------------------------------------------------
// Mirrors the save/load logic for the three boolean flags in
// moe_3gemm_swiglu_opt_impl: use_micro_gemm_prefill,
// use_gpu_mask_gen_prefill, use_grouped_gemm_prefill.
// Before the fix, these flags were not serialized, causing mismatched
// execution paths and buffer counts after cache load.

namespace {
struct moe_prefill_flags_test {
    bool use_micro_gemm_prefill = false;
    bool use_gpu_mask_gen_prefill = false;
    bool use_grouped_gemm_prefill = false;

    void save(BinaryOutputBuffer& ob) const {
        ob << use_micro_gemm_prefill;
        ob << use_gpu_mask_gen_prefill;
        ob << use_grouped_gemm_prefill;
    }

    void load(BinaryInputBuffer& ib) {
        ib >> use_micro_gemm_prefill;
        ib >> use_gpu_mask_gen_prefill;
        ib >> use_grouped_gemm_prefill;
    }
};
}  // namespace

// Test: micro_gemm path — only micro_gemm flag set
TEST(cache_serialization, moe_prefill_flags_micro_gemm_round_trip) {
    auto& engine = get_test_engine();

    moe_prefill_flags_test original;
    original.use_micro_gemm_prefill = true;
    original.use_gpu_mask_gen_prefill = true;
    original.use_grouped_gemm_prefill = false;

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    moe_prefill_flags_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.use_micro_gemm_prefill, true);
    ASSERT_EQ(loaded.use_gpu_mask_gen_prefill, true);
    ASSERT_EQ(loaded.use_grouped_gemm_prefill, false);
}

// Test: grouped_gemm path — only grouped_gemm flag set
TEST(cache_serialization, moe_prefill_flags_grouped_gemm_round_trip) {
    auto& engine = get_test_engine();

    moe_prefill_flags_test original;
    original.use_micro_gemm_prefill = false;
    original.use_gpu_mask_gen_prefill = false;
    original.use_grouped_gemm_prefill = true;

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    moe_prefill_flags_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.use_micro_gemm_prefill, false);
    ASSERT_EQ(loaded.use_gpu_mask_gen_prefill, false);
    ASSERT_EQ(loaded.use_grouped_gemm_prefill, true);
}

// Test: fallback path — all flags false (per-expert onednn loop)
TEST(cache_serialization, moe_prefill_flags_fallback_round_trip) {
    auto& engine = get_test_engine();

    moe_prefill_flags_test original;
    original.use_micro_gemm_prefill = false;
    original.use_gpu_mask_gen_prefill = false;
    original.use_grouped_gemm_prefill = false;

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    moe_prefill_flags_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.use_micro_gemm_prefill, false);
    ASSERT_EQ(loaded.use_gpu_mask_gen_prefill, false);
    ASSERT_EQ(loaded.use_grouped_gemm_prefill, false);
}

// Test: all flags true — should round-trip correctly even if this
// combination is not used in practice
TEST(cache_serialization, moe_prefill_flags_all_true_round_trip) {
    auto& engine = get_test_engine();

    moe_prefill_flags_test original;
    original.use_micro_gemm_prefill = true;
    original.use_gpu_mask_gen_prefill = true;
    original.use_grouped_gemm_prefill = true;

    membuf mem_buf;
    {
        std::ostream out_mem(&mem_buf);
        BinaryOutputBuffer ob(out_mem);
        original.save(ob);
    }

    moe_prefill_flags_test loaded;
    {
        std::istream in_mem(&mem_buf);
        BinaryInputBuffer ib(in_mem, engine);
        loaded.load(ib);
    }

    ASSERT_EQ(loaded.use_micro_gemm_prefill, true);
    ASSERT_EQ(loaded.use_gpu_mask_gen_prefill, true);
    ASSERT_EQ(loaded.use_grouped_gemm_prefill, true);
}

// --------------------------------------------------------------------------
// data::load_weights — data_size / output_layout cross-validation
// --------------------------------------------------------------------------
// Regression test for a heap buffer overflow in data::load_weights.
//
// output_layout (which sizes the destination buffer) and data_size (how many
// bytes are read into that buffer) are deserialized independently from the
// cache blob. A corrupted/malicious blob can declare a data_size far larger
// than the layout, and on the host-accessible path the bytes are read straight
// into a buffer allocated solely from output_layout
// (ib >> make_data(mem->buffer_ptr(), data_size)), overflowing it.
namespace {
// Build a blob in exactly the byte order that data::load_weights expects, with
// a data_size that lies about how much payload follows the layout.
membuf make_oversized_data_blob(const layout& output_layout, size_t malicious_data_size, allocation_type alloc_type) {
    membuf mem_buf;
    std::ostream out_mem(&mem_buf);
    BinaryOutputBuffer ob(out_mem);

    ob << output_layout;
    ob << make_data(&alloc_type, sizeof(alloc_type));
    ob << make_data(&malicious_data_size, sizeof(size_t));

    bool weightless_caching = false;  // take the plain (non-weightless) path
    ob << weightless_caching;

    // Mirror the reader's alignment padding so stream offsets stay in sync.
    if (!ob.is_encrypted() && !ob.is_offset_sub_buffer_aligned()) {
        std::vector<uint8_t> pad(ob.get_bytes_to_sub_buffer_boundary(), 0);
        ob << make_data(pad.data(), pad.size());
    }

    // Actually provide malicious_data_size bytes of payload so the vulnerable
    // read() copies all of them into the (much smaller) destination buffer.
    std::vector<uint8_t> payload(malicious_data_size, 0xAA);
    ob << make_data(payload.data(), payload.size());

    return mem_buf;
}
}  // namespace

TEST(cache_serialization, load_weights_rejects_data_size_exceeding_layout) {
    auto& engine = get_test_engine();

    // Pick a host-accessible allocation type the test device actually supports,
    // otherwise load_weights would take the device-copy path / fail to allocate.
    allocation_type alloc_type = allocation_type::usm_host;
    if (!engine.supports_allocation(allocation_type::usm_host)) {
        if (engine.supports_allocation(allocation_type::usm_shared)) {
            alloc_type = allocation_type::usm_shared;
        } else {
            GTEST_SKIP() << "Test device does not support host-accessible USM allocation";
        }
    }

    // Victim buffer: 4 bytes. Attacker claims 64 KB of weight data.
    layout small_layout({1, 1, 1, 4}, data_types::u8, format::bfyx);
    const size_t malicious_data_size = 64 * 1024;
    ASSERT_GT(malicious_data_size, small_layout.bytes_count());

    membuf mem_buf = make_oversized_data_blob(small_layout, malicious_data_size, alloc_type);

    std::istream in_mem(&mem_buf);
    BinaryInputBuffer ib(in_mem, engine);

    cldnn::data data_prim;
    // weights_memory / model_tensor_base are null: non-weightless, non-zero-copy
    // path, so mem is allocated from output_layout and the oversized read would
    // overflow it. The cross-check must reject the blob first.
    ASSERT_THROW(data_prim.load_weights(ib, /*weights_memory=*/nullptr, /*model_tensor_base=*/nullptr), ov::AssertFailure);
}