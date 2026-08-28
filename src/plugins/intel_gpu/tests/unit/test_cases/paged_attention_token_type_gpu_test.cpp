// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gpu_test.h"
#include "test_utils/test_data/paged_attention_token_type_test_data.h"

struct paged_attention_token_type_test_params : public paged_attention_test_params {
    test::TestData token_type_test_data;
};

class paged_attention_token_type_test : public PagedAttentionTest<paged_attention_token_type_test_params> {
public:
    void apply_token_type_test_data(PagedAttentionManager& pam, const paged_attention_token_type_test_params& p, const test::TestData& data) {
        ASSERT_EQ(p.subsequences.size(), 1);
        ASSERT_EQ(p.subsequences[0].past_len, 0);

        const size_t seq_len = data.tokenTypes.size();
        const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.k_head_size);
        ASSERT_EQ(static_cast<size_t>(p.subsequences[0].num_tokens), seq_len);
        ASSERT_EQ(data.qData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.kData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.vData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.expectedOutput.size(), seq_len * hidden_dim);

        pam.query_data = {to_float16(data.qData)};
        pam.key_data = {to_float16(data.kData)};
        pam.value_data = {to_float16(data.vData)};
        pam.token_type_ids.assign(data.tokenTypes.begin(), data.tokenTypes.end());
    }

    void compare_token_type_output(cldnn::memory::ptr data_output_mem, const std::vector<float>& expected_output) {
        ASSERT_TRUE(data_output_mem);
        ASSERT_EQ(data_output_mem->count(), expected_output.size());
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(data_output_mem, tests::get_test_stream());
        constexpr float token_type_tolerance = 1e-2f;

        for (size_t i = 0; i < data_output_mem->count(); i++) {
            ASSERT_NEAR(static_cast<float>(mem_ptr[i]), expected_output[i], token_type_tolerance) << " at index=" << i;
        }
    }
};
TEST_P(paged_attention_token_type_test, basic) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    apply_token_type_test_data(pam, p, p.token_type_test_data);

    auto result = run_gpu_inference(pam, p);

    cldnn::memory::ptr output_data_mem = nullptr;
    cldnn::memory::ptr output_scores_mem = nullptr;
    cldnn::memory::ptr output_diversity_mem = nullptr;

    output_data_mem = result.outputs.at("output_data").get_memory();

    compare_token_type_output(output_data_mem, p.token_type_test_data.expectedOutput);
}

static paged_attention_token_type_test_params make_token_type_test_param(const test::TestData& data, bool disable_flashattn_v2) {
    paged_attention_token_type_test_params p;
    p.subsequences = {{static_cast<int>(data.tokenTypes.size()), 0}};
    p.num_heads = 1;
    p.num_kv_heads = 1;
    p.k_head_size = 32;
    p.v_head_size = 32;
    p.block_size = 16;
    p.sliding_window_size = data.slidingWindowSize;
    p.kv_cache_compression = DISABLE_CACHE_COMPRESSION;
    p.key_cache_quant_mode = ov::internal::CacheQuantMode::BY_TOKEN;
    p.dynamic_paddings = STATIC_INPUT_PAD;
    p.scores_mode = DISABLE_SCORES;
    p.rotation_config = DISABLE_ROTATION;
    p.disable_flashattn_v2 = disable_flashattn_v2;
    p.token_type_ids = std::vector<int>(data.tokenTypes.begin(), data.tokenTypes.end());
    p.token_type_test_data = data;
    return p;
}

static std::vector<paged_attention_token_type_test_params> make_token_type_test_params(const std::vector<test::TestData>& test_data) {
    std::vector<paged_attention_token_type_test_params> params;
    params.reserve(test_data.size() * 2);
    for (const auto& data : test_data) {
        params.push_back(make_token_type_test_param(data, ENABLE_FA_V2));
        params.push_back(make_token_type_test_param(data, DISABLE_FA_V2));
    }
    return params;
}

static std::string get_token_type_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) +
           (p.disable_flashattn_v2 == DISABLE_FA_V2 ? "_FlashAttnV2Disabled" : "_FlashAttnV2Enabled");
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type,
                         paged_attention_token_type_test,
                         ::testing::ValuesIn(make_token_type_test_params(test::PagedAttentionTokenTypeTestData::GetTestData())),
                         get_token_type_test_name);

#ifdef ENABLE_ONEDNN_FOR_GPU
// Verify that micro SDPA is used for PREFILL when token_type_ids is present,
// and produces correct results with bidirectional mask.
class paged_attention_token_type_micro_sdpa_prefill_test : public paged_attention_token_type_test {};

TEST_P(paged_attention_token_type_micro_sdpa_prefill_test, prefill_only) {
    auto& engine = tests::get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "Micro SDPA requires DPAS/XMX support";

    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    // Run micro SDPA path
    apply_token_type_test_data(pam, p, p.token_type_test_data);
    auto result = run_gpu_inference(pam, p);

    // Verify micro SDPA kernel was actually executed
    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    auto* impl = pa_inst->get_impl();
    ASSERT_NE(impl, nullptr);
    auto dump_info = impl->get_kernels_dump_info(*pa_inst->get_impl_params());
    EXPECT_TRUE(dump_info.get_entries().find("sdpa_micro") != std::string::npos)
        << "Expected micro SDPA kernel for PREFILL with token_type_ids, got: " << dump_info.get_entries();

    // Compare micro SDPA output against golden data
    cldnn::memory::ptr output_data_mem = result.outputs.at("output_data").get_memory();
    compare_token_type_output(output_data_mem, p.token_type_test_data.expectedOutput);
}

static std::vector<paged_attention_token_type_test_params> make_micro_sdpa_prefill_test_params() {
    auto test_data = test::PagedAttentionTokenTypeTestData::GetTestData();
    std::vector<paged_attention_token_type_test_params> params;
    for (const auto& data : test_data) {
        params.push_back(make_token_type_test_param(data, ENABLE_FA_V2));
    }
    return params;
}

static std::string get_micro_sdpa_prefill_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) + "_MicroSDPA_Prefill";
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_micro_sdpa_prefill,
                         paged_attention_token_type_micro_sdpa_prefill_test,
                         ::testing::ValuesIn(make_micro_sdpa_prefill_test_params()),
                         get_micro_sdpa_prefill_test_name);

// Micro SDPA in the MIXED stage while token_type_ids is present. MIXED used to fall back to
// paged_attention_opt__multi_tokens, which carries no token_type_ids handling of its own, so the
// fallback only ever cost performance.
//
// The PREFILL golden data is replayed as a chunked prefill:
//
//            |<----- past_len ----->|<---- num_tokens ---->|
//     k, v   | preloaded into cache | passed as input      |
//     q      | -                    | passed as input      |
//     golden | -                    | compared             |
//
// past_len is picked so the query chunk holds text tokens only. A bidirectional pair needs both the
// query and the key to be image tokens, so a text-only chunk is masked purely causally - which is
// what both MIXED kernels produce, neither of them implementing token_type_ids.
class paged_attention_token_type_micro_sdpa_mixed_test : public paged_attention_token_type_test {
public:
    void apply_mixed_test_data(PagedAttentionManager& pam, const paged_attention_token_type_test_params& p, const test::TestData& data) {
        ASSERT_EQ(p.subsequences.size(), 1);

        const size_t past_len = static_cast<size_t>(p.subsequences[0].past_len);
        const size_t num_tokens = static_cast<size_t>(p.subsequences[0].num_tokens);
        const size_t seq_len = data.tokenTypes.size();
        const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.k_head_size);

        ASSERT_GT(past_len, 0u);
        ASSERT_EQ(past_len + num_tokens, seq_len);
        ASSERT_EQ(data.qData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.kData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.vData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.expectedOutput.size(), seq_len * hidden_dim);

        // Key/Value cover the whole sequence: PagedAttentionManager copies the leading past_len
        // tokens into the KV cache and submits the rest through the key/value inputs.
        pam.key_data = {to_float16(data.kData)};
        pam.value_data = {to_float16(data.vData)};

        // Query only covers the scheduled chunk.
        const auto query_data = to_float16(data.qData);
        pam.query_data = {std::vector<ov::float16>(query_data.begin() + past_len * hidden_dim, query_data.end())};

        pam.token_type_ids.assign(data.tokenTypes.begin(), data.tokenTypes.end());
    }
};

TEST_P(paged_attention_token_type_micro_sdpa_mixed_test, mixed_stage) {
    const auto& device_info = tests::get_test_engine().get_device_info();
    if (!device_info.supports_immad || device_info.arch < cldnn::gpu_arch::xe_hpg)
        GTEST_SKIP() << "Micro SDPA requires DPAS/XMX support";
    if (device_info.arch == cldnn::gpu_arch::xe3p)
        GTEST_SKIP() << "Micro SDPA is disabled on xe3p";

    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    apply_mixed_test_data(pam, p, p.token_type_test_data);
    auto result = run_gpu_inference(pam, p);

    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    auto* impl = pa_inst->get_impl();
    ASSERT_NE(impl, nullptr);
    const auto kernel_entries = impl->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries();
    EXPECT_NE(kernel_entries.find("sdpa_micro"), std::string::npos) << "Expected micro SDPA kernel for MIXED with token_type_ids, got: " << kernel_entries;
    EXPECT_EQ(kernel_entries.find("paged_attention_opt__multi_tokens"), std::string::npos) << "MIXED fell back to the partition kernel: " << kernel_entries;

    // The golden data covers the whole sequence, but only the scheduled chunk is produced here.
    const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.v_head_size);
    const size_t cached_values = static_cast<size_t>(p.subsequences[0].past_len) * hidden_dim;
    const auto& golden_output = p.token_type_test_data.expectedOutput;
    const std::vector<float> expected_output(golden_output.begin() + cached_values, golden_output.end());

    cldnn::memory::ptr output_data_mem = result.outputs.at("output_data").get_memory();
    compare_token_type_output(output_data_mem, expected_output);
}

// Picks a split that leaves only text tokens in the scheduled chunk, and returns 0 when no usable
// split exists - either every token is an image token, or the tail would be short enough to be
// classified as GENERATE instead of MIXED.
static int find_text_only_mixed_split(const test::TestData& data) {
    const int seq_len = static_cast<int>(data.tokenTypes.size());
    int last_image_token = -1;
    for (int i = 0; i < seq_len; i++) {
        if (data.tokenTypes[i] == 1)
            last_image_token = i;
    }

    const int past_len = std::max(last_image_token + 1, seq_len / 2);
    const bool usable = past_len > 0 && seq_len - past_len >= 2;
    return usable ? past_len : 0;
}

static std::vector<paged_attention_token_type_test_params> make_micro_sdpa_mixed_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    for (const auto& data : test::PagedAttentionTokenTypeTestData::GetTestData()) {
        const int past_len = find_text_only_mixed_split(data);
        if (past_len == 0)
            continue;

        auto p = make_token_type_test_param(data, ENABLE_FA_V2);
        p.subsequences = {{static_cast<int>(data.tokenTypes.size()) - past_len, past_len}};
        params.push_back(p);
    }
    return params;
}

static std::string get_micro_sdpa_mixed_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) + "_Past" + std::to_string(p.subsequences[0].past_len) +
           "_MicroSDPA_Mixed";
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_micro_sdpa_mixed,
                         paged_attention_token_type_micro_sdpa_mixed_test,
                         ::testing::ValuesIn(make_micro_sdpa_mixed_test_params()),
                         get_micro_sdpa_mixed_test_name);
#endif
