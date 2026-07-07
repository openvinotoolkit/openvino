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

    // Run opt path (FA_V2 disabled) to get reference output
    auto p_ref = p;
    p_ref.disable_flashattn_v2 = DISABLE_FA_V2;
    apply_token_type_test_data(pam, p_ref, p_ref.token_type_test_data);
    auto ref_result = run_gpu_inference(pam, p_ref);
    cldnn::memory::ptr ref_output_mem = ref_result.outputs.at("output_data").get_memory();
    ASSERT_TRUE(ref_output_mem);
    std::vector<float> ref_output(ref_output_mem->count());
    {
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> ref_ptr(ref_output_mem, tests::get_test_stream());
        for (size_t i = 0; i < ref_output.size(); i++) {
            ref_output[i] = static_cast<float>(ref_ptr[i]);
        }
    }

    // Run micro SDPA path (FA_V2 enabled)
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

    // Compare micro SDPA output against opt path reference
    cldnn::memory::ptr output_data_mem = result.outputs.at("output_data").get_memory();
    compare_token_type_output(output_data_mem, ref_output);
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
#endif
