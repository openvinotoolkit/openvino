// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gpu_test.h"
#include "test_utils/test_data/paged_attention_token_type_test_data.h"

struct paged_attention_token_type_test_params : public paged_attention_test_params {
    test::TestData token_type_test_data;
    // Multi-subsequence tests: golden cases that make up the subsequences.
    std::vector<std::string> case_names;
    // MIXED multi-subsequence test: past_len of each subsequence.
    std::vector<int> case_past;
    // Head-size test: how many times the golden head (32) is repeated.
    int head_repeat = 1;
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

// Micro SDPA stays selected in the MIXED stage when token_type_ids is present.
//
// The PREFILL golden data is replayed as a chunked prefill:
//
//            |<----- past_len ----->|<---- num_tokens ---->|
//     k, v   | preloaded into cache | passed as input      |
//     q      | -                    | passed as input      |
//     golden | -                    | compared             |
//
// past_len is picked so the chunk has text tokens only, so the mask is plain causal.
// Image groups are tested by the mixed_image test.
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

        // token_type_ids: scheduled tokens only (no past).
        pam.token_type_ids.assign(data.tokenTypes.begin() + past_len, data.tokenTypes.end());
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
        p.token_type_ids = std::vector<int>(data.tokenTypes.begin() + past_len, data.tokenTypes.end());
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

// MIXED with image tokens in the scheduled chunk.
//
// The PREFILL golden is valid for the chunk only if the split is not inside an image group,
// so the splits are on group boundaries and the chunk has at least one image token.
static std::vector<paged_attention_token_type_test_params> make_micro_sdpa_mixed_image_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    //  - middle_segment, two_small_image_groups: a few image groups
    //  - casual_indirectly: single image tokens between text tokens (no group)
    //  - long_sequence: many groups, chunk longer than one tile
    //  - long_chain_of_1s: one long group at the start of the chunk
    //  - long_input_with_mixed_token_types (SW5): sliding window with many groups
    struct Case { const char* name; int past; };
    const auto all = test::PagedAttentionTokenTypeTestData::GetTestData();
    for (const Case& c : {Case{"middle_segment", 3},
                          Case{"casual_indirectly", 8},
                          Case{"long_sequence", 85},
                          Case{"long_chain_of_1s", 1},
                          Case{"two_small_image_groups", 12},
                          Case{"long_input_with_mixed_token_types", 99}}) {
        const auto it = std::find_if(all.begin(), all.end(), [&](const test::TestData& d) { return d.name == c.name; });
        OPENVINO_ASSERT(it != all.end(), "token type test data not found: ", c.name);
        const auto& data = *it;
        auto p = make_token_type_test_param(data, ENABLE_FA_V2);
        p.subsequences = {{static_cast<int>(data.tokenTypes.size()) - c.past, c.past}};
        p.token_type_ids = std::vector<int>(data.tokenTypes.begin() + c.past, data.tokenTypes.end());
        params.push_back(p);
    }
    return params;
}

static std::string get_micro_sdpa_mixed_image_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) + "_Past" + std::to_string(p.subsequences[0].past_len) +
           "_MicroSDPA_MixedImage";
}

class paged_attention_token_type_micro_sdpa_mixed_image_test : public paged_attention_token_type_micro_sdpa_mixed_test {};

TEST_P(paged_attention_token_type_micro_sdpa_mixed_image_test, mixed_stage_with_image_in_chunk) {
    // Same as mixed_stage, but the chunk has image tokens.
    const auto& device_info = tests::get_test_engine().get_device_info();
    if (!device_info.supports_immad || device_info.arch < cldnn::gpu_arch::xe_hpg)
        GTEST_SKIP() << "Micro SDPA requires DPAS/XMX support";

    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    apply_mixed_test_data(pam, p, p.token_type_test_data);
    auto result = run_gpu_inference(pam, p);

    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    const auto kernel_entries = pa_inst->get_impl()->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries();
    EXPECT_NE(kernel_entries.find("sdpa_micro"), std::string::npos) << "Expected micro SDPA kernel for MIXED, got: " << kernel_entries;

    const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.v_head_size);
    const size_t cached_values = static_cast<size_t>(p.subsequences[0].past_len) * hidden_dim;
    const auto& golden_output = p.token_type_test_data.expectedOutput;
    const std::vector<float> expected_output(golden_output.begin() + cached_values, golden_output.end());

    cldnn::memory::ptr output_data_mem = result.outputs.at("output_data").get_memory();
    compare_token_type_output(output_data_mem, expected_output);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_micro_sdpa_mixed_image,
                         paged_attention_token_type_micro_sdpa_mixed_image_test,
                         ::testing::ValuesIn(make_micro_sdpa_mixed_image_test_params()),
                         get_micro_sdpa_mixed_image_test_name);
#endif

// PREFILL with two subsequences.
//
// token_type_ids covers all subsequences, so subsequence B starts at subsequence_begins[B].
// A kernel that uses the index inside B reads A's values and the output no longer matches the golden.
class paged_attention_token_type_multi_subsequence_test : public paged_attention_token_type_test {
public:
    static const test::TestData& find_case(const std::vector<test::TestData>& all, const std::string& name) {
        for (const auto& d : all) {
            if (d.name == name)
                return d;
        }
        OPENVINO_THROW("token type test data not found: ", name);
    }
};

TEST_P(paged_attention_token_type_multi_subsequence_test, prefill_two_subsequences) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    const auto all = test::PagedAttentionTokenTypeTestData::GetTestData();
    ASSERT_EQ(p.case_names.size(), p.subsequences.size());

    pam.query_data.clear();
    pam.key_data.clear();
    pam.value_data.clear();
    pam.token_type_ids.clear();
    std::vector<float> expected;
    for (const auto& name : p.case_names) {
        const auto& d = find_case(all, name);
        ASSERT_EQ(d.slidingWindowSize, p.sliding_window_size);
        pam.query_data.push_back(to_float16(d.qData));
        pam.key_data.push_back(to_float16(d.kData));
        pam.value_data.push_back(to_float16(d.vData));
        pam.token_type_ids.insert(pam.token_type_ids.end(), d.tokenTypes.begin(), d.tokenTypes.end());
        expected.insert(expected.end(), d.expectedOutput.begin(), d.expectedOutput.end());
    }

    auto result = run_gpu_inference(pam, p);
    cldnn::memory::ptr output_data_mem = result.outputs.at("output_data").get_memory();

    compare_token_type_output(output_data_mem, expected);
}

static paged_attention_token_type_test_params make_multi_subsequence_param(const std::vector<std::string>& names, bool disable_fa_v2) {
    const auto all = test::PagedAttentionTokenTypeTestData::GetTestData();
    auto p = make_token_type_test_param(paged_attention_token_type_multi_subsequence_test::find_case(all, names.front()), disable_fa_v2);
    p.case_names = names;
    p.subsequences.clear();
    std::vector<int> ids;
    for (const auto& name : names) {
        const auto& d = paged_attention_token_type_multi_subsequence_test::find_case(all, name);
        p.subsequences.push_back({static_cast<int>(d.tokenTypes.size()), 0});
        ids.insert(ids.end(), d.tokenTypes.begin(), d.tokenTypes.end());
    }
    p.token_type_ids = ids;
    return p;
}

static std::vector<paged_attention_token_type_test_params> make_multi_subsequence_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    // The image group is in the second subsequence, and in the first.
    for (const auto& names : std::vector<std::vector<std::string>>{{"plain_casual", "middle_segment"}, {"middle_segment", "plain_casual"}}) {
        params.push_back(make_multi_subsequence_param(names, ENABLE_FA_V2));
    }
    return params;
}

static std::string get_multi_subsequence_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    std::string name;
    for (const auto& n : p.case_names) {
        name += (name.empty() ? "" : "_then_") + n;
    }
    return name + "_SW" + std::to_string(p.sliding_window_size) + (p.disable_flashattn_v2 == DISABLE_FA_V2 ? "_FlashAttnV2Disabled" : "_FlashAttnV2Enabled");
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_multi_subsequence,
                         paged_attention_token_type_multi_subsequence_test,
                         ::testing::ValuesIn(make_multi_subsequence_test_params()),
                         get_multi_subsequence_test_name);

// MIXED with two subsequences, each with image groups: one has past (past_len > 0), one does not.
// Both orders are used, so the image group has a zero and a non-zero subsequence_begin.
class paged_attention_token_type_mixed_multi_subsequence_test : public paged_attention_token_type_multi_subsequence_test {};

TEST_P(paged_attention_token_type_mixed_multi_subsequence_test, mixed_stage_two_subsequences) {
    auto p = GetParam();
#ifdef ENABLE_ONEDNN_FOR_GPU
    const auto& device_info = tests::get_test_engine().get_device_info();
    const bool micro_expected = device_info.supports_immad && device_info.arch >= cldnn::gpu_arch::xe_hpg;
#else
    const bool micro_expected = false;  // no micro SDPA without oneDNN
#endif

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    const auto all = test::PagedAttentionTokenTypeTestData::GetTestData();
    ASSERT_EQ(p.case_names.size(), p.subsequences.size());
    ASSERT_EQ(p.case_past.size(), p.subsequences.size());
    const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.k_head_size);

    pam.query_data.clear();
    pam.key_data.clear();
    pam.value_data.clear();
    pam.token_type_ids.clear();
    std::vector<float> expected;
    for (size_t i = 0; i < p.case_names.size(); i++) {
        const auto& d = find_case(all, p.case_names[i]);
        const size_t past = static_cast<size_t>(p.case_past[i]);
        ASSERT_EQ(d.slidingWindowSize, p.sliding_window_size);
        ASSERT_EQ(past + static_cast<size_t>(p.subsequences[i].num_tokens), d.tokenTypes.size());

        // Key/Value: whole sequence (past goes to the cache). Query: scheduled tokens only.
        const auto q = to_float16(d.qData);
        pam.query_data.push_back(std::vector<ov::float16>(q.begin() + past * hidden_dim, q.end()));
        pam.key_data.push_back(to_float16(d.kData));
        pam.value_data.push_back(to_float16(d.vData));
        pam.token_type_ids.insert(pam.token_type_ids.end(), d.tokenTypes.begin() + past, d.tokenTypes.end());
        expected.insert(expected.end(), d.expectedOutput.begin() + past * hidden_dim, d.expectedOutput.end());
    }

    auto result = run_gpu_inference(pam, p);

    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    const auto kernel_entries = pa_inst->get_impl()->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries();
    if (micro_expected) {
        EXPECT_NE(kernel_entries.find("sdpa_micro"), std::string::npos) << "Expected micro SDPA kernel for MIXED, got: " << kernel_entries;
    }

    compare_token_type_output(result.outputs.at("output_data").get_memory(), expected);
}

static paged_attention_token_type_test_params make_mixed_multi_subsequence_param(const std::vector<std::string>& names,
                                                                                 const std::vector<int>& past,
                                                                                 bool disable_fa_v2) {
    const auto all = test::PagedAttentionTokenTypeTestData::GetTestData();
    auto p = make_token_type_test_param(paged_attention_token_type_multi_subsequence_test::find_case(all, names.front()), disable_fa_v2);
    p.case_names = names;
    p.case_past = past;
    p.subsequences.clear();
    std::vector<int> ids;
    for (size_t i = 0; i < names.size(); i++) {
        const auto& d = paged_attention_token_type_multi_subsequence_test::find_case(all, names[i]);
        p.subsequences.push_back({static_cast<int>(d.tokenTypes.size()) - past[i], past[i]});
        ids.insert(ids.end(), d.tokenTypes.begin() + past[i], d.tokenTypes.end());
    }
    p.token_type_ids = ids;
    return p;
}

static std::vector<paged_attention_token_type_test_params> make_mixed_multi_subsequence_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    // Splits are on image-group boundaries.
    struct Case { std::vector<std::string> names; std::vector<int> past; };
    const std::vector<Case> cases = {
        // SW0: chunk with past, then a fresh prompt (and the reverse order)
        {{"middle_segment", "casual_indirectly"}, {3, 0}},
        {{"casual_indirectly", "middle_segment"}, {0, 3}},
        // SW5: same with a sliding window
        {{"two_small_image_groups", "short_seq_in_middle"}, {12, 0}},
    };
    for (const auto& c : cases) {
        params.push_back(make_mixed_multi_subsequence_param(c.names, c.past, ENABLE_FA_V2));
    }
    return params;
}

static std::string get_mixed_multi_subsequence_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    std::string name;
    for (size_t i = 0; i < p.case_names.size(); i++) {
        name += (name.empty() ? "" : "_then_") + p.case_names[i] + "_Past" + std::to_string(p.case_past[i]);
    }
    return name + "_SW" + std::to_string(p.sliding_window_size);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_mixed_multi_subsequence,
                         paged_attention_token_type_mixed_multi_subsequence_test,
                         ::testing::ValuesIn(make_mixed_multi_subsequence_test_params()),
                         get_mixed_multi_subsequence_test_name);

#ifdef ENABLE_ONEDNN_FOR_GPU
// MIXED micro SDPA with token_type_ids at head size 64/128/256.
//
// The golden head size is 32. Repeating the head n times (q' = q / sqrt(n), k' = k, v' = v) keeps the scores,
// and the output is the golden output repeated n times.
//
// This checks that the micro kernel still builds and is selected (no fallback to paged_attention_opt__multi_tokens).
class paged_attention_token_type_micro_sdpa_mixed_head_test : public paged_attention_token_type_micro_sdpa_mixed_test {};

TEST_P(paged_attention_token_type_micro_sdpa_mixed_head_test, mixed_stage_head_size) {
    const auto& device_info = tests::get_test_engine().get_device_info();
    if (!device_info.supports_immad || device_info.arch < cldnn::gpu_arch::xe_hpg)
        GTEST_SKIP() << "Micro SDPA requires DPAS/XMX support";

    auto p = GetParam();
    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    const auto& d = p.token_type_test_data;
    const size_t n = static_cast<size_t>(p.head_repeat);
    const size_t base_hidden = 32;
    const size_t past_len = static_cast<size_t>(p.subsequences[0].past_len);
    const size_t seq_len = d.tokenTypes.size();
    ASSERT_EQ(static_cast<size_t>(p.k_head_size), base_hidden * n);

    auto repeat = [&](const std::vector<float>& src, float mul) {
        std::vector<float> out;
        out.reserve(src.size() * n);
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t r = 0; r < n; r++) {
                for (size_t c = 0; c < base_hidden; c++) {
                    out.push_back(src[t * base_hidden + c] * mul);
                }
            }
        }
        return out;
    };

    const auto q = to_float16(repeat(d.qData, 1.0f / std::sqrt(static_cast<float>(n))));
    const size_t hidden = base_hidden * n;
    pam.query_data = {std::vector<ov::float16>(q.begin() + past_len * hidden, q.end())};
    pam.key_data = {to_float16(repeat(d.kData, 1.0f))};
    pam.value_data = {to_float16(repeat(d.vData, 1.0f))};
    pam.token_type_ids.assign(d.tokenTypes.begin() + past_len, d.tokenTypes.end());

    auto result = run_gpu_inference(pam, p);

    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    const auto kernel_entries = pa_inst->get_impl()->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries();
    EXPECT_NE(kernel_entries.find("sdpa_micro"), std::string::npos) << "head_size=" << p.k_head_size << ": micro SDPA not used for MIXED, got: " << kernel_entries;
    EXPECT_EQ(kernel_entries.find("paged_attention_opt__multi_tokens"), std::string::npos)
        << "head_size=" << p.k_head_size << ": MIXED fell back to the partition kernel: " << kernel_entries;

    const auto expected_full = repeat(d.expectedOutput, 1.0f);
    const std::vector<float> expected(expected_full.begin() + past_len * hidden, expected_full.end());
    compare_token_type_output(result.outputs.at("output_data").get_memory(), expected);
}

static std::vector<paged_attention_token_type_test_params> make_micro_sdpa_mixed_head_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    const auto all = test::PagedAttentionTokenTypeTestData::GetTestData();
    struct Case { const char* name; int past; std::vector<int> repeats; };
    // SW0: all head sizes. SW5: the largest only.
    for (const Case& c : {Case{"long_sequence", 85, {2, 4, 8}}, Case{"two_small_image_groups", 12, {8}}}) {
        const auto& d = paged_attention_token_type_multi_subsequence_test::find_case(all, c.name);
        for (const int n : c.repeats) {  // head_size 64, 128, 256
            auto p = make_token_type_test_param(d, ENABLE_FA_V2);
            p.k_head_size = 32 * n;
            p.v_head_size = 32 * n;
            p.head_repeat = n;
            p.subsequences = {{static_cast<int>(d.tokenTypes.size()) - c.past, c.past}};
            p.token_type_ids = std::vector<int>(d.tokenTypes.begin() + c.past, d.tokenTypes.end());
            params.push_back(p);
        }
    }
    return params;
}

static std::string get_micro_sdpa_mixed_head_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) + "_Past" + std::to_string(p.subsequences[0].past_len) + "_Head" +
           std::to_string(p.k_head_size);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_micro_sdpa_mixed_head,
                         paged_attention_token_type_micro_sdpa_mixed_head_test,
                         ::testing::ValuesIn(make_micro_sdpa_mixed_head_test_params()),
                         get_micro_sdpa_mixed_head_test_name);
#endif  // ENABLE_ONEDNN_FOR_GPU

// ---- Performance (DISABLED_; run with --gtest_also_run_disabled_tests) ----
//
// MIXED with token_type_ids, long past. Runs PA_TT_PERF_ITERS times (default 50) for kernel timing tools.
// Outputs are not checked.
struct pa_tt_perf_config {
    const char* name;
    int past_len;
    int chunk;
    int head_size;
    int num_heads;
    int num_kv_heads;
    bool with_images;   // image groups in the chunk (otherwise all zero)
};

class paged_attention_token_type_perf_test : public paged_attention_token_type_test {};

static std::vector<int> make_chunk_token_types(int chunk, bool with_images) {
    std::vector<int> v(chunk, 0);
    if (with_images) {
        // two image groups, about half of the chunk
        const int len = chunk / 4;
        for (int g = 0; g < 2; g++) {
            const int start = chunk / 8 + g * (chunk / 2);
            for (int i = start; i < std::min(chunk, start + len); i++)
                v[i] = 1;
        }
    }
    return v;
}

TEST_P(paged_attention_token_type_perf_test, mixed_token_type_perf) {
    auto p = GetParam();
    const auto& device_info = tests::get_test_engine().get_device_info();
    if (!device_info.supports_immad || device_info.arch < cldnn::gpu_arch::xe_hpg)
        GTEST_SKIP() << "Micro SDPA requires DPAS/XMX support";
    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    auto result = run_gpu_inference(pam, p);

    int iters = 50;
    if (const char* e = std::getenv("PA_TT_PERF_ITERS"))
        iters = std::max(1, std::atoi(e));
    for (int i = 0; i < iters; i++) {
        result.network->execute();
    }
    tests::get_test_stream().finish();

    auto pa_inst = result.network->get_primitive("paged_attention");
    ASSERT_NE(pa_inst, nullptr);
    std::cout << "[perf] kernels: " << pa_inst->get_impl()->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries() << " iters=" << iters << std::endl;
}

static std::vector<paged_attention_token_type_test_params> make_token_type_perf_params() {
    std::vector<paged_attention_token_type_test_params> params;
    const std::vector<pa_tt_perf_config> cfgs = {
        {"TextOnly_H128", 4096, 256, 128, 16, 8, false},
        {"Image_H128", 4096, 256, 128, 16, 8, true},
        {"TextOnly_H256", 4096, 256, 256, 8, 4, false},
        {"Image_H256", 4096, 256, 256, 8, 4, true},
        {"TextOnly_H128_Past16k", 16384, 256, 128, 16, 8, false},
    };
    for (const auto& c : cfgs) {
        paged_attention_token_type_test_params p;
        p.subsequences = {{c.chunk, c.past_len}};
        p.num_heads = c.num_heads;
        p.num_kv_heads = c.num_kv_heads;
        p.k_head_size = c.head_size;
        p.v_head_size = c.head_size;
        p.block_size = 16;
        p.sliding_window_size = 0;
        p.kv_cache_compression = DISABLE_CACHE_COMPRESSION;
        p.key_cache_quant_mode = ov::internal::CacheQuantMode::BY_TOKEN;
        p.dynamic_paddings = STATIC_INPUT_PAD;
        p.scores_mode = DISABLE_SCORES;
        p.rotation_config = DISABLE_ROTATION;
        p.disable_flashattn_v2 = ENABLE_FA_V2;
        p.run_reference = false;
        p.token_type_ids = make_chunk_token_types(c.chunk, c.with_images);
        p.token_type_test_data.name = c.name;
        p.case_names = {c.name};
        params.push_back(p);
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_paged_attention_token_type_perf,
                         paged_attention_token_type_perf_test,
                         ::testing::ValuesIn(make_token_type_perf_params()),
                         [](const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) { return obj.param.case_names.front(); });
