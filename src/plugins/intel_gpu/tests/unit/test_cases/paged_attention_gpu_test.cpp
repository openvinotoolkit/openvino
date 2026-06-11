// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gpu_test.hpp"

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
    /* u4 with GQA (multi query groups) */
    paged_attention_test_params{ {{1, 34}}, 8, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, false, 0, {}, false, {}, {}, ov::element::u4 }, // 2nd token, GQA 8/2 u4+BY_CHANNEL

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

