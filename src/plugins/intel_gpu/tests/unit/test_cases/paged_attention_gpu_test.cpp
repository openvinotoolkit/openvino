// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gpu_test.h"

class paged_attention_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(paged_attention_test, basic) {
    auto p = GetParam();

    execute(p, p.run_reference);
}

class xattention_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(xattention_test, basic) {
    auto p = GetParam();
    if (!check_cm_available())
        GTEST_SKIP() << "CM JIT support is required for XAttention tests, and the device must be Xe1 or later";

    execute(p, p.run_reference);
}

class xattention_invalid_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(xattention_invalid_test, throws_on_invalid_xattention_inputs) {
    auto p = GetParam();
    if (!check_cm_available())
        GTEST_SKIP() << "CM JIT support is required for XAttention tests, and the device must be Xe1 or later";

    std::string expected_error_substr;
    if (!p.xattention_block_size.has_value() || p.xattention_block_size->empty()) {
        expected_error_substr = "XAttention block size input";
    } else if (!p.xattention_threshold.has_value() || p.xattention_threshold->empty()) {
        expected_error_substr = "XAttention threshold input";
    } else {
        expected_error_substr = "XAttention";
    }

    try {
        execute(p, false);
        FAIL() << "Expected exception containing: " << expected_error_substr;
    } catch (const ov::Exception& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(expected_error_substr), std::string::npos)
            << "Unexpected exception message: " << msg;
    } catch (const std::exception& e) {
        FAIL() << "Expected ov::Exception, got std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception, got unknown exception type";
    }
}

class adaptive_rkv_diversity_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(adaptive_rkv_diversity_test, basic) {
    auto p = GetParam();

    execute(p);
}

class qq_bias_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(qq_bias_test, basic) {
    auto p = GetParam();

    execute(p);
}


static paged_attention_test_params disable_reference_compare(paged_attention_test_params p) {
    p.run_reference = false;
    return p;
}

static std::vector<int32_t> gen_tokens_ids_test_data(size_t seq_len, int num_images, size_t avg_img_len) {
    std::vector<int32_t> v(seq_len, 0);
    size_t gap = seq_len / (num_images + 1);
    for (int img = 0; img < num_images; img++) {
        size_t center = gap * (img + 1);
        size_t half = avg_img_len / 2;
        size_t start = (center > half) ? center - half : 0;
        size_t end = std::min(seq_len, center + half);
        for (size_t i = start; i < end; i++)
            v[i] = 1;
    }


    return v;
}


// Test class to verify FlashAttnV2 multi-token sink correction is actually active.
// Runs the same prompt twice with FA_V2 enabled: once with non-zero sinks, once with zero sinks.
// Asserts outputs differ, proving the sink correction in sdpa_opt.cl executes.
// This test is platform-independent (works on any GPU with OpenCL support).
class paged_attention_sink_v2_effect_test : public PagedAttentionTest<paged_attention_test_params> {};

TEST_P(paged_attention_sink_v2_effect_test, verify_sink_changes_v2_output) {
    auto p = GetParam();
    p.has_sink_input = true;
    p.force_flashattn_v2 = true;

    // Run FlashAttnV2 multi-token path with non-zero sinks
    std::vector<ov::float16> nonzero_sinks(p.num_heads);
    for (int h = 0; h < p.num_heads; h++)
        nonzero_sinks[h] = ov::float16(-1.0f + 0.3f * h);

    rg.set_seed(GET_SUITE_NAME);
    p.sink_values = nonzero_sinks;
    execute(p, false);
    auto output_with_sinks = get_output_data();

    // Run FlashAttnV2 multi-token path with zero sinks (sink code still runs but has no effect)
    std::vector<ov::float16> zero_sinks(p.num_heads, ov::float16(0.0f));

    rg.set_seed(GET_SUITE_NAME);
    p.sink_values = zero_sinks;
    execute(p, false);
    auto output_no_sinks = get_output_data();

    ASSERT_EQ(output_with_sinks.size(), output_no_sinks.size());

    // Verify outputs differ — proving the V2 sink correction code is active
    bool found_difference = false;
    for (size_t i = 0; i < output_with_sinks.size(); i++) {
        if (std::abs(static_cast<float>(output_with_sinks[i]) - static_cast<float>(output_no_sinks[i])) > 1e-4f) {
            found_difference = true;
            break;
        }
    }
    ASSERT_TRUE(found_difference)
        << "FlashAttnV2 multi-token output is identical with and without sinks — sink code may not be executing";
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_sink_v2_effect, paged_attention_sink_v2_effect_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Multi-token (prompt) cases that exercise the FlashAttnV2 online-softmax path with sinks.
    // These directly test the HAS_SINK_INPUT code at line 2552 in sdpa_opt.cl.
    // Platform-independent: forces FA_V2 regardless of HW micro-kernel availability.
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false },
    paged_attention_test_params{ {{128, 0}}, 4, 4, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false },
    // GQA prompt
    paged_attention_test_params{ {{64, 0}}, 8, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false },
}));

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention, paged_attention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    /* with scores output, use SnapKV */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores output */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 16, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 128, 96, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 48, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 96, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 32, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 48, 96, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 48, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 16, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* without scores output, dynamic input query paddings */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores, per_block rotation */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 48, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 48, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores, per_token rotation */
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 128, 192, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* without scores output, dynamic input query paddings, KV-cache compression */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token

    /* with scores, per_block rotation, KV-cache compression */
    paged_attention_test_params{ {{1, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{1, 34}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    /* with scores, per_token rotation, KV-cache compression */
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* With sliding windows */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 6, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 6, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{512, 0}}, 2, 2, 64, 32, 16, 20, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{512, 0}}, 2, 2, 64, 32, 16, 20, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, 8, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token + 2nd token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 48, 64, 16, 128, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, 4, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{5, 10}}, 2, 2, 64, 64, 16, 2, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{248, 16}}, 2, 2, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{34, 0}}, 2, 2, 32, 32, 16, 2, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 1st token
    paged_attention_test_params{ {{1, 1008}}, 32, 32, 128, 128, 16, 6, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token
    paged_attention_test_params{ {{6, 20}}, 2, 2, 128, 128, 16, 8, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1, 288}}, 64, 8, 64, 64, 16, 128, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching, GQA
    paged_attention_test_params{ {{84, 2}}, 32, 32, 128, 128, 16, 16, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 492}}, 32, 32, 32, 32, 16, 32, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 492}}, 16, 16, 64, 64, 16, 64, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 492}}, 8, 8, 128, 128, 16, 128, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{2, 30}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{232, 24}}, 2, 2, 512, 512, 16, 32, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 592}}, 32, 32, 128, 128, 16, 64, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 692}}, 32, 32, 128, 128, 16, 128, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 792}}, 32, 32, 128, 128, 16, 256, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: prefix caching
    paged_attention_test_params{ {{1, 34}, {2, 20}, {10, 34}}, 2, 2, 64, 64, 16, 10, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // mixed: 2nd token + 1st token + part of 1st token
    /* Per-Block rotation cases with  Per-Channel quantization*/
    paged_attention_test_params{ {{16, 32}}, 2, 2, 32, 32, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{8, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{256, 56}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{48, 1024}}, 8, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{2, 34}, {2, 515}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{4, 34}, {4, 515}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{6, 34}, {25, 86}, {10, 64}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate
    paged_attention_test_params{ {{8, 34}, {25, 86}, {10, 64}}, 2, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per block rotate

    /* Per-Token rotation cases with  Per-Channel quantization*/
    paged_attention_test_params{ {{8, 38}}, 16, 16, 256, 256, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{34, 34}}, 2, 2, 32, 32, 16, 2, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{2, 1008}}, 32, 32, 128, 128, 16, 6, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{6, 40}}, 2, 2, 128, 128, 16, 8, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{254, 60}}, 32, 8, 128, 128, 16, 10, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{84, 256}}, 32, 32, 128, 128, 16, 16, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{40, 96}, {30, 50}}, 2, 2, 64, 64, 16, 8, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate
    paged_attention_test_params{ {{128, 130}, {256, 60}}, 2, 2, 48, 64, 16, 128, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, ENABLE_FA_V2, false, 0, {}, false }, // 2nd token, per token rotate

    /* INT4 KV-cache compression (u4+BY_CHANNEL only; BY_TOKEN is not supported for 4-bit) */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 1st token u4+BY_CHANNEL
    paged_attention_test_params{ {{256, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 1st token long u4+BY_CHANNEL
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 1st token + 1st token u4+BY_CHANNEL
    paged_attention_test_params{ {{1, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token u4+BY_CHANNEL
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token + 2nd token u4+BY_CHANNEL
    paged_attention_test_params{ {{1, 300}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token, past_len>partition_size u4+BY_CHANNEL
    /* u4 MIXED mode (generate+prefill in same batch) */
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // mixed u4+BY_CHANNEL
    /* u4 with larger head size */
    paged_attention_test_params{ {{1, 34}}, 2, 2, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token, head_size=128 u4+BY_CHANNEL
    // Test optimization logic (exercises USE_DUAL_NIBBLE_V_OPT when v_head_size/2 % SUBGROUP_SIZE == 0)
    paged_attention_test_params{ {{1, 300}}, 2, 2, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token, past_len>partition_size, head_size=128 u4 (dual-nibble + multi-partition)
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd+2nd token batch, head_size=128 u4 (dual-nibble)
    paged_attention_test_params{ {{10, 0}}, 2, 2, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 1st token, head_size=128 u4 (dual-nibble prefill)
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // mixed mode, head_size=128 u4 (dual-nibble)

    /* u4 with GQA (multi query groups) */
    paged_attention_test_params{ {{1, 34}}, 8, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token, GQA 8/2 u4+BY_CHANNEL
    // Test optimization logic (exercises USE_DUAL_NIBBLE_V_OPT when v_head_size/2 % SUBGROUP_SIZE == 0)
    paged_attention_test_params{ {{1, 34}}, 24, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token, GQA 24/8 head_size=128 u4 (LLaMA-3.2-3B config, dual-nibble)

    paged_attention_test_params{ {{1, 4200}}, 8, 2, 64, 64, 16, 8, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // GQA 8/2 + scores + sw=8, kv_group_size=4 -> QUERIES_PER_WI=4
    paged_attention_test_params{ {{1, 4200}}, 4, 2, 64, 64, 16, 8, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // GQA 4/2 + scores + sw=8, kv_group_size=2 -> QUERIES_PER_WI=2
    paged_attention_test_params{ {{1, 5000}}, 16, 4, 128, 128, 16, 16, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }, // GQA 16/4 + scores + sw=16, k_head_size=128, kv_group_size=4
}));

INSTANTIATE_TEST_SUITE_P(smoke_cm_xattention, xattention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    //////////////////////////////////////////////////////////////////////////////// single-seq + bypass xattention ////////////////////////////////////////////////////////////////////////
    // single-seq prefill + uncompressed cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/0]
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/1]
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/2]

    // single-seq generate + uncompressed cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/3]
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/4]
    paged_attention_test_params{ {{1, 10}},   2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // generate_only [basic/5]

    paged_attention_test_params{ {{1, 300}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // generate_only, past_len>partition [basic/6]
    paged_attention_test_params{ {{1, 1023}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/7]
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/8]
    paged_attention_test_params{ {{1, 127}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/9]
    paged_attention_test_params{ {{1, 128}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/10]
    paged_attention_test_params{ {{1, 129}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/11]
    paged_attention_test_params{ {{1, 32}},   28, 28, 128, 128, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/12]

    // single-seq prefill + by_token i8 cache
    paged_attention_test_params{ {{32, 0}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/13]
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/14]
    paged_attention_test_params{ {{2048, 0}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/15]
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/16]
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/17]
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/18]

    // single-seq generate + by_token i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/19]
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd toke [basic/20]
    paged_attention_test_params{ {{1, 10}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // generate_only [basic/21]
    paged_attention_test_params{ {{1, 300}},  2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // generate_only, past_len>partition [basic/22]
    paged_attention_test_params{ {{1, 1023}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/23]
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/24]

    // single-seq prefill + by_channel i8 cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/25]
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/26]
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/27]

    // single-seq generate + by_channel i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/28]
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/29]
    paged_attention_test_params{ {{1, 1023}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/30]
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/31]

    //////////////////////////////////////////////////////////////////////////////// single-seq + xattention ////////////////////////////////////////////////////////////////////////
    // single-seq prefill + uncompressed cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/32]
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/33] - Skipped on Xe1, see TEST_P
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/34] - Skipped on Xe1, see TEST_P

    // single-seq generate + uncompressed cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/35]
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/36]
    paged_attention_test_params{ {{1, 10}},   2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // generate_only [basic/37]

    paged_attention_test_params{ {{1, 300}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // generate_only, past_len>partition [basic/38]
    paged_attention_test_params{ {{1, 1023}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/39]
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/40]
    paged_attention_test_params{ {{1, 127}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/41]
    paged_attention_test_params{ {{1, 128}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/42]
    paged_attention_test_params{ {{1, 129}},  2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/43]
    paged_attention_test_params{ {{1, 32}},   28, 28, 128, 128, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/44]

    // single-seq prefill + by_token i8 cache
    paged_attention_test_params{ {{32, 0}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/45]
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/46]
    paged_attention_test_params{ {{2048, 0}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/47]
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/48]
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/49]
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/50]

    // single-seq generate + by_token i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/51]
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd toke [basic/52]
    paged_attention_test_params{ {{1, 10}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // generate_only [basic/53]
    paged_attention_test_params{ {{1, 300}},  2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // generate_only, past_len>partition [basic/54]
    paged_attention_test_params{ {{1, 1023}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/55]
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/56]

    // single-seq prefill + by_channel i8 cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/57]
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/58]
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/59]

    // single-seq generate + by_channel i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/60]
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/61]
    paged_attention_test_params{ {{1, 1023}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/62]
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/63]

    //////////////////////////////////////////////////////////////////////////////// diversary head_size and head_ratio + bypass xattention ////////////////////////////////////////////////////////////////////////
    /////////////////// head_size 96 as phi-3-mini-128k-instruct //////////////////////////////
    // single-seq prefill + uncompressed cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 96, 96, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/64]
    // single-seq generate + uncompressed cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 96, 96, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/65]

    // single-seq prefill + by_token i8 cache
    paged_attention_test_params{ {{32, 0}},   2, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/66]
    // single-seq generate + by_token i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/67]
                
    // single-seq prefill + by_channel i8 cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/68]
    // single-seq generate + by_channel i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/69]

    /////////////////// head_ratio 16:1 as minicpm4 //////////////////////////////
    // single-seq prefill + uncompressed cache
    paged_attention_test_params{ {{32, 0}},   32, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/70]
    // single-seq generate + uncompressed cache
    paged_attention_test_params{ {{1, 31}},   32, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/71]

    // single-seq prefill + by_token i8 cache
    paged_attention_test_params{ {{32, 0}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/72]
    // single-seq generate + by_token i8 cache
    paged_attention_test_params{ {{1, 31}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/73]
                
    // single-seq prefill + by_channel i8 cache
    paged_attention_test_params{ {{32, 0}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 1st token [basic/74]
    // single-seq generate + by_channel i8 cache
    paged_attention_test_params{ {{1, 31}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} }, // 2nd token [basic/75]

    //////////////////////////////////////////////////////////////////////////////// diversary head_size and head_ratio + enable xattention ////////////////////////////////////////////////////////////////////////
    /////////////////// head_size 96 as phi-3-mini-128k-instruct //////////////////////////////
    // single-seq prefill + uncompressed cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 96, 96, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/76]
    // single-seq generate + uncompressed cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 96, 96, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/77]

    // single-seq prefill + by_token i8 cache
    paged_attention_test_params{ {{32, 0}},   2, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/78]
    // single-seq generate + by_token i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/79]
                
    // single-seq prefill + by_channel i8 cache
    paged_attention_test_params{ {{32, 0}},   4, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/80]
    // single-seq generate + by_channel i8 cache
    paged_attention_test_params{ {{1, 31}},   2, 2, 96, 96, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/81]

    /////////////////// head_ratio 16:1 as minicpm4 //////////////////////////////
    // single-seq prefill + uncompressed cache
    paged_attention_test_params{ {{32, 0}},   32, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/82]
    // single-seq generate + uncompressed cache
    paged_attention_test_params{ {{1, 31}},   32, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/83]

    // single-seq prefill + by_token i8 cache
    paged_attention_test_params{ {{32, 0}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/84]
    // single-seq generate + by_token i8 cache
    paged_attention_test_params{ {{1, 31}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/85]
                
    // single-seq prefill + by_channel i8 cache
    paged_attention_test_params{ {{32, 0}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 1st token [basic/86]
    // single-seq generate + by_channel i8 cache
    paged_attention_test_params{ {{1, 31}},   32, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} }, // 2nd token [basic/87]

    //////////////////////////////////////////////////////////////////////////////// multi-seq + bypass xattention ////////////////////////////////////////////////////////////////////////
    // multi-seq prefill-only
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f}, std::vector<int>{128, 128} }, // multi-subsequence prefill [basic/88]
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f}, std::vector<int>{128, 128} }, // multi-subsequence prefill [basic/89]
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f}, std::vector<int>{128, 128} }, // multi-subsequence prefill [basic/90]

    // multi-seq generate-only
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f}, std::vector<int>{128, 128} }, // multi-subsequence generate [basic/91]
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f}, std::vector<int>{128, 128} }, // multi-subsequence generate [basic/92]
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f}, std::vector<int>{128, 128} }, // multi-subsequence generate [basic/93]

    // multi-seq mixed
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f, 100.f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/94]
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f, 100.f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/95]
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f, 100.f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/96]

    paged_attention_test_params{ {{1, 1024 + 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f, 100.f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/97]
    paged_attention_test_params{ {{1, 1024 + 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f, 100.f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/98]
    paged_attention_test_params{ {{1, 1024 + 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.f, 100.f, 100.f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/99]

    //////////////////////////////////////////////////////////////////////////////// multi-seq + enable xattention ////////////////////////////////////////////////////////////////////////
    // multi-seq prefill
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128} }, // multi-subsequence generate [basic/100]
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128} }, // multi-subsequence generate [basic/101]
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128} }, // multi-subsequence generate [basic/102]

    // multi-seq generate
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128} }, // multi-subsequence mixed [basic/103]
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128} }, // multi-subsequence mixed [basic/104]
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128} }, // multi-subsequence mixed [basic/105]

    // multi-seq mixed
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/106]
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/107]
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/108]

    paged_attention_test_params{ {{1, 1024 + 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/109]
    paged_attention_test_params{ {{1, 1024 + 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/110]
    paged_attention_test_params{ {{1, 1024 + 34}, {25, 0}, {10, 34}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence mixed [basic/111]

    // Option B: maximally-divergent past_tail to expose by-channel dispatch under-count (basic/112)
    // past_len=3 => past_tail=3%16=3; cur_tokens 5,7,9 => sub-blocks ceil((3+5)/16)=1, ceil((3+7)/16)=1, ceil((3+9)/16)=1 = 3 sub-blocks
    // Simple ceil_div(total_tokens=21, 16) would give 2, under-dispatching by 1 sub-block
    paged_attention_test_params{ {{5, 3}, {7, 3}, {9, 3}}, 4, 2, 64, 64, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f, 0.9f}, std::vector<int>{128, 128, 128} }, // multi-subsequence by-channel divergent [basic/112]
}));

INSTANTIATE_TEST_SUITE_P(smoke_cm_xattention_block_size, xattention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Valid values
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128} },
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{256} },
    // Any other value
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{1000} },
}));

INSTANTIATE_TEST_SUITE_P(smoke_cm_xattention_block_size_invalid, xattention_invalid_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Abnormal: user does not provide xattention_block_size values
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::nullopt },
    // Abnormal: user provides empty xattention_block_size tensor
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{} },
    // Abnormal: user provides invalid number of block sizes (should be 1 or equal to batch size)
    paged_attention_test_params{ {{16, 0}, {16, 0}, {16, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f}, std::vector<int>{128, 128} },
}));

INSTANTIATE_TEST_SUITE_P(smoke_cm_xattention_threshold_invalid, xattention_invalid_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Abnormal: user does not provide xattention_threshold values
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::nullopt, std::vector<int>{128} },
    // Abnormal: user provides empty xattention_threshold tensor
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{}, std::vector<int>{128} },
    // Abnormal: user provides invalid number of threshold values (should be 1 or equal to batch size)
    paged_attention_test_params{ {{16, 0}, {16, 0}, {16, 0}}, 2, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{0.9f, 0.9f}, std::vector<int>{128, 128, 128} },
}));

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_rkv, adaptive_rkv_diversity_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Small evictable_size tests (uniform across sequences)
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },
    paged_attention_test_params{ {{128, 0}}, 4, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },

    // Edge cases - empty evictable_size (kernel skip test)
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 0, {0} },    // Empty - skip kernel
    paged_attention_test_params{ {{32, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {16} },  // Minimal valid size (block-aligned)

    // Multi-sequence tests - uniform evictable_sizes
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },            // 2 sequences: same size
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },  // 3 sequences: same size

    // Multi-sequence tests - varying evictable_sizes per sequence
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48} },                              // 2 sequences: different evictable sizes
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48, 32} },                // 3 sequences: different sizes
    paged_attention_test_params{ {{128, 0}, {128, 0}, {128, 0}, {128, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {16, 32, 48, 32} },  // 4 sequences: varied sizes

    // With KV cache compression - BY_TOKEN
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },
    paged_attention_test_params{ {{128, 0}}, 4, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48} },
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48, 32} },

    // With KV cache compression - BY_CHANNEL
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },
    paged_attention_test_params{ {{128, 0}}, 4, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48} },
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48, 32} },

    // Large evictable_size tests
    paged_attention_test_params{ {{192, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 48, {96} },
    paged_attention_test_params{ {{256, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 64, {128} },
    paged_attention_test_params{ {{512, 0}}, 4, 2, 128, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 128, {256} },

    // GQA configurations
    paged_attention_test_params{ {{128, 0}}, 8, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },     // heads_num=8, start+evict=96
    paged_attention_test_params{ {{160, 0}}, 16, 4, 128, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {48} },  // heads_num=16, start+evict=80

    // Different head sizes
    paged_attention_test_params{ {{128, 0}}, 2, 2, 128, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },  // head_size=128, start+evict=96
    paged_attention_test_params{ {{160, 0}}, 4, 4, 256, 256, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {48} },  // head_size=256, start+evict=80
}));

INSTANTIATE_TEST_SUITE_P(smoke_qq_bias, qq_bias_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // basic tests with 1 sequence
    paged_attention_test_params{ {{4, 32}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic ,true, ENABLE_QQ_BIAS },
    paged_attention_test_params{ {{4, 32}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, true, ENABLE_QQ_BIAS },
    paged_attention_test_params{ {{4, 32}}, 2, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, true, ENABLE_QQ_BIAS },

    // multi sequences tests
    paged_attention_test_params{ {{4, 32}, {128, 0}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, true, QueryToQueryAttentionDescriptor{{{{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}}}, {0, 16, 16}} },
    paged_attention_test_params{ {{128, 0}, {4, 32}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, true, QueryToQueryAttentionDescriptor{{{{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}}}, {0, 0, 16}} },
    paged_attention_test_params{ {{4, 20}, {4, 32}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, true, QueryToQueryAttentionDescriptor{{{{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}, {1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}}}, {0, 16, 32}} },

    // multi sequence with different qq bias patterns
    paged_attention_test_params{ {{4, 20}, {2, 32}, {4, 25}}, 2, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, DISABLE_DIVERSITY, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, true, QueryToQueryAttentionDescriptor{{{{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}, {1, 0, 1, 1}, {1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}}}, {0, 16, 20, 36}} },
}));

// Performance-focused tests with larger sequence lengths, single/multi-subsequences, and CM v.s. OCL/micro path (which is triggered with xattention ON/OFF).
// These tests are not validating correctness (outputs are not checked), but are intended to be run in a performance benchmarking mode to evaluate kernel performance and behavior of the paged attention implementation.
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_paged_attention_perf_ocl, paged_attention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // prefill-only
    disable_reference_compare(paged_attention_test_params{ {{4096, 0}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 4*1024}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 7*1024}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 15*1024}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 23*1024}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 31*1024}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),

    // generate-only
    disable_reference_compare(paged_attention_test_params{ {{1, 4096}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION,   DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{1, 4*4096}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{1, 8*4096}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
    disable_reference_compare(paged_attention_test_params{ {{1, 16*4096}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),

    // mixed prefill+generate
    disable_reference_compare(paged_attention_test_params{ {{1, 1*4096}, {1, 1*1024}, {4096, 1024}}, 32, 8, 128, 128, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false }),
}));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_paged_attention_perf_cm, xattention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // prefill-only
    disable_reference_compare(paged_attention_test_params{ {{4096, 0}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 4*1024}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 7*1024}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 15*1024}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 23*1024}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{4096, 31*1024}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),

    // generate-only
    disable_reference_compare(paged_attention_test_params{ {{1, 4096}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{1, 4*4096}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{1, 8*4096}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),
    disable_reference_compare(paged_attention_test_params{ {{1, 16*4096}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{256} }),

    // mixed prefill+generate
    disable_reference_compare(paged_attention_test_params{ {{1, 1*4096}, {1, 1*1024}, {4096, 1024}}, 32, 8, 128, 128, 256, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f, 100.0f, 100.0f}, std::vector<int>{256, 256, 256} }),
}));

static constexpr int TOKEN_IDS_SEQ_LEN = 8192;
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_paged_attention_perf_token_ids_ocl, paged_attention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    disable_reference_compare(paged_attention_test_params{ {{TOKEN_IDS_SEQ_LEN, 0}}, 1, 1, 128, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, false, {}, true, gen_tokens_ids_test_data(TOKEN_IDS_SEQ_LEN, 3, TOKEN_IDS_SEQ_LEN/4) }),
    disable_reference_compare(paged_attention_test_params{ {{TOKEN_IDS_SEQ_LEN, 0}}, 1, 1, 128, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2, false, 0, {}, false, std::nullopt, std::nullopt, ov::element::dynamic, false, {}, true, gen_tokens_ids_test_data(TOKEN_IDS_SEQ_LEN, 1, TOKEN_IDS_SEQ_LEN) }),
}));

// Shared KV-cache test: simulates writer (write_kv_cache=true) + reader (write_kv_cache=false)
// sharing the same KV cache. Verifies the reader produces correct attention output by comparing
// against a reference SDPA computed with the writer's K,V data.
class shared_kv_cache_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(shared_kv_cache_test, reader_output_matches_reference) {
    auto p = GetParam();

    if (p.has_xattention && !check_cm_available())
        GTEST_SKIP() << "CM JIT support is required for XAttention tests";

    // --- Step 1: Create writer PAM and run writer PA (write_kv_cache=true) to fill cache ---
    PagedAttentionManager pam(rg,
                              tests::get_test_engine(),
                              tests::get_test_stream(),
                              p.subsequences,
                              p.num_heads,
                              p.num_kv_heads,
                              p.k_head_size,
                              p.v_head_size,
                              p.block_size,
                              p.sliding_window_size,
                              p.kv_cache_compression,
                              p.key_cache_quant_mode,
                              false,
                              p.has_xattention,
                              p.rotation_config);

    if (p.has_xattention) {
        pam.xattention_block_size.clear();
        if (p.xattention_block_size.has_value())
            pam.xattention_block_size = p.xattention_block_size.value();
        pam.xattention_threshold.clear();
        if (p.xattention_threshold.has_value()) {
            for (const float t : p.xattention_threshold.value())
                pam.xattention_threshold.emplace_back(static_cast<ov::float16>(t));
        }
        pam.xattention_stride.assign(p.subsequences.size(), 16);
    }

    auto writer_query_mem = pam.get_query_memory();
    auto key_mem = pam.get_key_memory();
    auto value_mem = pam.get_value_memory();
    auto key_cache_mem = p.has_xattention ? pam.get_key_cache_memory_cm() : pam.get_key_cache_memory();
    auto value_cache_mem = pam.get_value_cache_memory();
    auto past_lens_mem = pam.get_past_lens_memory();
    auto subsequence_begins_mem = pam.get_subsequence_begins_memory();
    auto block_indices_mem = pam.get_block_indices_memory();
    auto block_indices_begins_mem = pam.get_block_indices_begins_memory();
    auto scale_mem = pam.get_scale_memory();
    auto sliding_window_mem = pam.get_sliding_window_memory();
    auto alibi_mem = pam.get_alibi_memory();
    auto max_context_len_mem = pam.get_max_context_len_memory();
    auto score_aggregation_mem = pam.get_score_aggregation();
    auto rotated_block_indices_mem = pam.get_rotated_block_indices_memory();
    auto rotation_deltas_mem = pam.get_rotation_deltas_memory();
    auto rotation_trig_lut_mem = pam.get_rotation_trig_lut_memory();
    auto xattention_threshold_mem = pam.get_xattention_threshold_memory();
    auto xattention_block_size_mem = pam.get_xattention_block_size_memory();
    auto xattention_stride_mem = pam.get_xattention_stride_memory();
    auto sinks_mem = pam.get_sinks_memory();
    auto adaptive_rkv_start_size_mem = pam.get_adaptive_rkv_start_size_memory();
    auto adaptive_rkv_evictable_sizes_mem = pam.get_adaptive_rkv_evictable_sizes_memory();
    auto adaptive_rkv_diversity_block_set_indices_mem = pam.get_adaptive_rkv_diversity_block_set_indices_memory();
    auto adaptive_rkv_diversity_block_set_indices_begins_mem = pam.get_adaptive_rkv_diversity_block_set_indices_begins_memory();
    auto token_type_ids_mem = pam.get_token_type_ids_memory();
    auto qq_bias_mem = pam.get_qq_bias_memory();
    auto qq_bias_begins_mem = pam.get_qq_bias_begins_memory();

    auto query_layout = writer_query_mem->get_layout();
    auto key_layout = key_mem->get_layout();
    auto value_layout = value_mem->get_layout();
    auto key_cache_layout = key_cache_mem->get_layout();
    auto value_cache_layout = value_cache_mem->get_layout();

    query_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.k_head_size });
    key_layout.set_partial_shape(ov::PartialShape{ -1, p.num_kv_heads * p.k_head_size });
    value_layout.set_partial_shape(ov::PartialShape{ -1, p.num_kv_heads * p.v_head_size });
    { auto ps = key_cache_layout.get_partial_shape(); ps[0] = -1; key_cache_layout.set_partial_shape(ps); }
    { auto ps = value_cache_layout.get_partial_shape(); ps[0] = -1; value_cache_layout.set_partial_shape(ps); }

    auto build_pa_network = [&](bool write_kv_cache) {
        std::vector<cldnn::input_info> pa_inputs = {
            cldnn::input_info("query"), cldnn::input_info("key"), cldnn::input_info("value"),
            cldnn::input_info("key_cache"), cldnn::input_info("value_cache"),
            cldnn::input_info("past_lens"), cldnn::input_info("subsequence_begins"),
            cldnn::input_info("block_indices"), cldnn::input_info("block_indices_begins"),
            cldnn::input_info("scale"), cldnn::input_info("sliding_window"), cldnn::input_info("alibi"),
            cldnn::input_info("max_context_len"), cldnn::input_info("score_aggregation_window"),
            cldnn::input_info("rotated_block_indices"), cldnn::input_info("rotation_deltas"),
            cldnn::input_info("rotation_trig_lut_modified"),
            cldnn::input_info("xattention_threshold"), cldnn::input_info("xattention_block_size"),
            cldnn::input_info("xattention_stride"), cldnn::input_info("sinks"),
            cldnn::input_info("adaptive_rkv_start_size"), cldnn::input_info("adaptive_rkv_evictable_sizes"),
            cldnn::input_info("adaptive_rkv_diversity_block_set_indices"), cldnn::input_info("adaptive_rkv_diversity_block_set_indices_begins"),
            cldnn::input_info("token_type_ids"), cldnn::input_info("qq_bias"), cldnn::input_info("qq_bias_begins")
        };

        auto pa_prim = cldnn::paged_attention("paged_attention", pa_inputs);
        pa_prim.k_head_size = p.k_head_size;
        pa_prim.v_head_size = p.v_head_size;
        pa_prim.kv_heads_num = p.num_kv_heads;
        pa_prim.heads_num = p.num_heads;
        pa_prim.scale_val = pam.get_default_scale();
        pa_prim.has_alibi = false;
        pa_prim.num_outputs = 1;
        pa_prim.sliding_window = p.sliding_window_size;
        pa_prim.is_key_by_channel = (p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL);
        pa_prim.has_xattention = p.has_xattention;
        pa_prim.write_kv_cache = write_kv_cache;

        cldnn::topology topo;
        topo.add(
            cldnn::input_layout("query", query_layout),
            cldnn::input_layout("key", key_layout),
            cldnn::input_layout("value", value_layout),
            cldnn::input_layout("key_cache", key_cache_layout),
            cldnn::input_layout("value_cache", value_cache_layout),
            cldnn::input_layout("past_lens", past_lens_mem->get_layout()),
            cldnn::input_layout("subsequence_begins", subsequence_begins_mem->get_layout()),
            cldnn::input_layout("block_indices", block_indices_mem->get_layout()),
            cldnn::input_layout("block_indices_begins", block_indices_begins_mem->get_layout()),
            cldnn::input_layout("scale", scale_mem->get_layout()),
            cldnn::input_layout("sliding_window", sliding_window_mem->get_layout()),
            cldnn::input_layout("alibi", alibi_mem->get_layout()),
            cldnn::input_layout("max_context_len", max_context_len_mem->get_layout()),
            cldnn::input_layout("score_aggregation_window", score_aggregation_mem->get_layout()),
            pa_prim,
            cldnn::reorder("output", cldnn::input_info("paged_attention", 0), cldnn::format::bfyx, cldnn::data_types::f16)
        );
        topo.add(cldnn::input_layout("rotated_block_indices", rotated_block_indices_mem->get_layout()));
        topo.add(cldnn::input_layout("rotation_deltas", rotation_deltas_mem->get_layout()));
        topo.add(cldnn::input_layout("rotation_trig_lut", rotation_trig_lut_mem->get_layout()));
        topo.add(cldnn::activation("rotation_trig_lut_modified", cldnn::input_info("rotation_trig_lut"), cldnn::activation_func::none));
        topo.add(cldnn::input_layout("xattention_threshold", xattention_threshold_mem->get_layout()));
        topo.add(cldnn::input_layout("xattention_block_size", xattention_block_size_mem->get_layout()));
        topo.add(cldnn::input_layout("xattention_stride", xattention_stride_mem->get_layout()));
        topo.add(cldnn::input_layout("sinks", sinks_mem->get_layout()));
        topo.add(cldnn::input_layout("adaptive_rkv_start_size", adaptive_rkv_start_size_mem->get_layout()));
        topo.add(cldnn::input_layout("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem->get_layout()));
        topo.add(cldnn::input_layout("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem->get_layout()));
        topo.add(cldnn::input_layout("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem->get_layout()));
        topo.add(cldnn::input_layout("token_type_ids", token_type_ids_mem->get_layout()));
        topo.add(cldnn::input_layout("qq_bias", qq_bias_mem->get_layout()));
        topo.add(cldnn::input_layout("qq_bias_begins", qq_bias_begins_mem->get_layout()));

        ov::intel_gpu::ExecutionConfig config = tests::get_test_default_config(tests::get_test_engine());
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::internal::key_cache_quant_mode(p.key_cache_quant_mode));
        return tests::get_network(tests::get_test_engine(), topo, config, tests::get_test_stream_ptr(), false);
    };

    auto set_network_inputs = [&](cldnn::network::ptr& net, cldnn::memory::ptr query_mem_input) {
        net->set_input_data("query", query_mem_input);
        net->set_input_data("key", key_mem);
        net->set_input_data("value", value_mem);
        net->set_input_data("key_cache", key_cache_mem);
        net->set_input_data("value_cache", value_cache_mem);
        net->set_input_data("past_lens", past_lens_mem);
        net->set_input_data("subsequence_begins", subsequence_begins_mem);
        net->set_input_data("block_indices", block_indices_mem);
        net->set_input_data("block_indices_begins", block_indices_begins_mem);
        net->set_input_data("scale", scale_mem);
        net->set_input_data("sliding_window", sliding_window_mem);
        net->set_input_data("alibi", alibi_mem);
        net->set_input_data("max_context_len", max_context_len_mem);
        net->set_input_data("score_aggregation_window", score_aggregation_mem);
        net->set_input_data("rotated_block_indices", rotated_block_indices_mem);
        net->set_input_data("rotation_deltas", rotation_deltas_mem);
        net->set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
        net->set_input_data("xattention_threshold", xattention_threshold_mem);
        net->set_input_data("xattention_block_size", xattention_block_size_mem);
        net->set_input_data("xattention_stride", xattention_stride_mem);
        net->set_input_data("sinks", sinks_mem);
        net->set_input_data("adaptive_rkv_start_size", adaptive_rkv_start_size_mem);
        net->set_input_data("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem);
        net->set_input_data("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem);
        net->set_input_data("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem);
        net->set_input_data("token_type_ids", token_type_ids_mem);
        net->set_input_data("qq_bias", qq_bias_mem);
        net->set_input_data("qq_bias_begins", qq_bias_begins_mem);
    };

    // Execute writer PA to fill the KV cache
    auto writer_network = build_pa_network(true);
    set_network_inputs(writer_network, writer_query_mem);
    writer_network->execute();

    // --- Step 2: Fill K,V with garbage to verify write_kv_cache=false skips cache write ---
    {
        cldnn::mem_lock<ov::float16> key_lock(key_mem, tests::get_test_stream());
        std::fill(key_lock.begin(), key_lock.end(), ov::float16(0.0f));
    }
    {
        cldnn::mem_lock<ov::float16> value_lock(value_mem, tests::get_test_stream());
        std::fill(value_lock.begin(), value_lock.end(), ov::float16(0.0f));
    }

    // Generate a fresh query for the reader (different from writer's query)
    std::vector<std::vector<ov::float16>> reader_query_data;
    for (const auto& subsequence_desc : p.subsequences) {
        size_t elements = static_cast<size_t>(p.num_heads) * subsequence_desc.num_tokens * p.k_head_size;
        reader_query_data.push_back(rg.generate_random_1d<ov::float16>(elements, -1, 1));
    }

    int total_tokens = 0;
    for (const auto& sd : p.subsequences)
        total_tokens += sd.num_tokens;

    auto reader_query_layout = cldnn::layout{ ov::PartialShape{ total_tokens, p.num_heads * p.k_head_size }, cldnn::data_types::f16, cldnn::format::bfyx };
    auto reader_query_mem = tests::get_test_engine().allocate_memory(reader_query_layout);
    {
        cldnn::mem_lock<ov::float16> lock(reader_query_mem, tests::get_test_stream());
        size_t offset = 0;
        for (size_t seq_idx = 0; seq_idx < p.subsequences.size(); seq_idx++) {
            size_t num_tokens = p.subsequences[seq_idx].num_tokens;
            size_t elements = num_tokens * p.num_heads * p.k_head_size;
            std::copy(reader_query_data[seq_idx].begin(),
                      reader_query_data[seq_idx].begin() + elements,
                      lock.begin() + offset);
            offset += elements;
        }
    }

    // --- Step 3: Execute reader PA (write_kv_cache=false) with the same cache ---
    // Input: Q(reader), K,V(zeros), kv_cache(filled by writer). Attention uses Q + kv_cache only.
    auto reader_network = build_pa_network(false);
    set_network_inputs(reader_network, reader_query_mem);
    auto reader_outputs = reader_network->execute();
    auto reader_output_mem = reader_outputs.at("output").get_memory();
    ASSERT_NE(reader_output_mem, nullptr);

    // --- Step 4: Compute reference SDPA ---
    // Input: Q(reader), K,V(writer's original data). Same data as what's in the cache.
    pam.query_data = reader_query_data;
    auto ref_data = PagedAttentionReference(pam).get_reference();
    const auto& ref_output = std::get<0>(ref_data);

    // --- Step 5: Compare reader output against reference ---
    ASSERT_EQ(reader_output_mem->count(), ref_output.size());
    cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> output_lock(reader_output_mem, tests::get_test_stream());
    for (size_t i = 0; i < ref_output.size(); i++) {
        ASSERT_NEAR(static_cast<float>(output_lock[i]), static_cast<float>(ref_output[i]), 7e-3f)
            << " at index=" << i;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_shared_kv_cache, shared_kv_cache_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Decode: 1 token, past=32 (cache already populated by writer)
    paged_attention_test_params{ {{1, 32}}, 4, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 },
    // Decode: 1 token, past=64 (longer context)
    paged_attention_test_params{ {{1, 64}}, 4, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 },
    // Multi-sequence decode
    paged_attention_test_params{ {{1, 32}, {1, 64}}, 4, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 },
}));

// CM (XAttention) path
INSTANTIATE_TEST_SUITE_P(smoke_shared_kv_cache_cm, shared_kv_cache_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Decode: 1 token, past=32
    paged_attention_test_params{ {{1, 32}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} },
    // Decode: 1 token, past=64
    paged_attention_test_params{ {{1, 64}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f}, std::vector<int>{128} },
    // Multi-sequence decode
    paged_attention_test_params{ {{1, 32}, {1, 64}}, 4, 2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, true, std::vector<float>{100.0f, 100.0f}, std::vector<int>{128, 128} },
}));

// Test to verify BY_CHANNEL requantize offset correctness for large head_dim (>=256).
// When head_dim > SUBGROUP_SIZE (16), the KV cache update kernel partitions the head dimension
// across work-groups. A bug in the head_size_offset calculation caused overlapping/missing channel
// writes for channels >= SUBGROUP_SIZE * NUM_K_HEAD_SIZE_PARTITIONS (e.g., 272-511 for head_dim=512),
// resulting in non-deterministic cache content.
// This test writes to the KV cache via PA, reads back the quantized cache, dequantizes, and
// verifies against the original input key data.
class kv_cache_by_channel_large_head_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(kv_cache_by_channel_large_head_test, verify_kv_cache_content) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam_ref = *this->pam;

    auto result = run_gpu_inference(pam_ref, p);

    PagedAttentionReference ref(pam_ref);

    for (size_t seq_idx = 0; seq_idx < p.subsequences.size(); seq_idx++) {
        const auto& sd = p.subsequences[seq_idx];
        const int total_tokens = sd.num_tokens + sd.past_len;

        auto cached_key = ref.read_key_from_cache(result.key_cache_mem, seq_idx, total_tokens);

        // Compare newly written tokens (positions past_len .. total_tokens-1) against original input
        for (int token_idx = sd.past_len; token_idx < total_tokens; token_idx++) {
            for (int head_idx = 0; head_idx < p.num_kv_heads; head_idx++) {
                for (int dim = 0; dim < p.k_head_size; dim++) {
                    size_t cache_offset = static_cast<size_t>(head_idx) * total_tokens * p.k_head_size +
                                          static_cast<size_t>(token_idx) * p.k_head_size + dim;
                    size_t input_offset = static_cast<size_t>(token_idx) * p.num_kv_heads * p.k_head_size +
                                          static_cast<size_t>(head_idx) * p.k_head_size + dim;

                    float cached_val = static_cast<float>(cached_key[cache_offset]);
                    float original_val = static_cast<float>(pam_ref.key_data[seq_idx][input_offset]);

                    // BY_CHANNEL i8 quantization tolerance: scale=(max-min)/255 per sub-block per channel
                    ASSERT_NEAR(cached_val, original_val, 25e-3f)
                        << " seq=" << seq_idx << " token=" << token_idx
                        << " head=" << head_idx << " dim=" << dim;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_kv_cache_by_channel_large_head, kv_cache_by_channel_large_head_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // head_dim=256, BY_CHANNEL i8: NUM_HEAD_SIZE_GROUPS=16, multiple partitions in generate
    paged_attention_test_params{ {{1, 10}}, 2, 2, 256, 256, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
    // head_dim=512, BY_CHANNEL i8: reproduces the original Gemma-4 VLM bug
    paged_attention_test_params{ {{1, 10}}, 2, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
    // head_dim=512, BY_CHANNEL i8: prefill (multi-token write)
    paged_attention_test_params{ {{10, 0}}, 2, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
    // head_dim=512, BY_CHANNEL i8: mixed (prefix-cached + new tokens)
    paged_attention_test_params{ {{4, 8}}, 2, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
    // head_dim=512, BY_CHANNEL u4: INT4 variant
    paged_attention_test_params{ {{1, 10}}, 2, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, std::nullopt, std::nullopt, ov::element::u4 },
    // head_dim=512, BY_CHANNEL u4: prefill
    paged_attention_test_params{ {{10, 0}}, 2, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, std::nullopt, std::nullopt, ov::element::u4 },
    // head_dim=256, BY_CHANNEL i8: GQA (8 heads, 2 KV heads)
    paged_attention_test_params{ {{1, 10}}, 8, 2, 256, 256, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
    // head_dim=512, BY_CHANNEL i8: GQA (8 heads, 2 KV heads)
    paged_attention_test_params{ {{1, 10}}, 8, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
    // head_dim=512, BY_CHANNEL i8: multi-sequence generate
    paged_attention_test_params{ {{1, 10}, {1, 14}}, 2, 2, 512, 512, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 },
}));

