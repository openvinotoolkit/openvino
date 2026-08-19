// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gpu_test.h"
#include "test_utils/test_data/paged_attention_token_type_test_data.h"

struct paged_attention_token_type_test_params : public paged_attention_test_params {
    test::TestData token_type_test_data;
};

constexpr float token_type_tolerance = 1e-2f;

// The kernel-name guard the MIXED suites share: the bidirectional mask for a non-PREFILL stage exists
// in sdpa_ocl only (sdpa_micro.cl gates its block on IS_PREFILL, and pa_multi_token /
// paged_attention_opt.cl has no token_type_ids support at all). Check what ACTUALLY ran instead of
// re-deriving the generator choice -- that covers TEST_USE_SDPA_OCL=0, non-immad devices and builds
// without micro kernels in one place. Free function so both fixture instantiations can call it.
static bool ran_sdpa_ocl(const cldnn::network::ptr& network, std::string& entries_out) {
    auto pa_inst = network->get_primitive("paged_attention");
    EXPECT_NE(pa_inst, nullptr);
    if (pa_inst == nullptr)
        return false;
    auto* impl = pa_inst->get_impl();
    EXPECT_NE(impl, nullptr);
    if (impl == nullptr)
        return false;
    entries_out = impl->get_kernels_dump_info(*pa_inst->get_impl_params()).get_entries();
    return entries_out.find("sdpa_ocl") != std::string::npos;
}

class paged_attention_token_type_test : public PagedAttentionTest<paged_attention_token_type_test_params> {
public:
    // Loads one golden sequence of length L into the manager, split as past = [0, past_len) +
    // new = [past_len, L). past_len == 0 is the PREFILL case the golden data was generated for;
    // past_len > 0 turns the same data into a MIXED step, see pick_mixed_split() for when that is
    // still described by the same golden output.
    void apply_token_type_test_data(PagedAttentionManager& pam, const paged_attention_token_type_test_params& p, const test::TestData& data) {
        ASSERT_EQ(p.subsequences.size(), 1);
        const size_t past_len = static_cast<size_t>(p.subsequences[0].past_len);

        const size_t seq_len = data.tokenTypes.size();
        const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.k_head_size);
        ASSERT_EQ(static_cast<size_t>(p.subsequences[0].num_tokens) + past_len, seq_len);
        ASSERT_EQ(data.qData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.kData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.vData.size(), seq_len * hidden_dim);
        ASSERT_EQ(data.expectedOutput.size(), seq_len * hidden_dim);

        // Q holds the NEW tokens only, while K/V keep all L rows: PagedAttentionManager writes
        // [0, past_len) of key_data/value_data into the KV cache and passes the remainder as the K/V
        // inputs (get_QKV_memory's skip_past_len). token_type_ids is "[B_token]", i.e. new tokens too.
        const auto q_all = to_float16(data.qData);
        pam.query_data = {std::vector<ov::float16>(q_all.begin() + past_len * hidden_dim, q_all.end())};
        pam.key_data = {to_float16(data.kData)};
        pam.value_data = {to_float16(data.vData)};
        pam.token_type_ids.assign(data.tokenTypes.begin() + past_len, data.tokenTypes.end());
    }

    // expected_offset skips the golden rows belonging to the cached prefix; the GPU only outputs the
    // new tokens.
    void compare_token_type_output(cldnn::memory::ptr data_output_mem, const std::vector<float>& expected_output, size_t expected_offset = 0) {
        ASSERT_TRUE(data_output_mem);
        ASSERT_EQ(data_output_mem->count(), expected_output.size() - expected_offset);
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(data_output_mem, tests::get_test_stream());

        for (size_t i = 0; i < data_output_mem->count(); i++) {
            ASSERT_NEAR(static_cast<float>(mem_ptr[i]), expected_output[expected_offset + i], token_type_tolerance) << " at index=" << i;
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

// ------------------------------------------------------------------------- the reference's yardstick
// PagedAttentionReference models bidirectional image groups itself (see
// get_mask_mem_combined_multi_head), and the MIXED suite below is measured against it. So the model
// needs its own check: reproduce the checked-in golden output for the PREFILL case it was generated
// for. Without this, the reference-based MIXED coverage would be our code grading our code.
// No PagedAttention network runs here -- only the reference topology.
class paged_attention_token_type_reference_test : public paged_attention_token_type_test {};

TEST_P(paged_attention_token_type_reference_test, reference_matches_golden) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    apply_token_type_test_data(pam, p, p.token_type_test_data);

    const auto ref_data = std::get<0>(PagedAttentionReference(pam).get_reference());
    const auto& expected = p.token_type_test_data.expectedOutput;
    ASSERT_EQ(ref_data.size(), expected.size());
    for (size_t i = 0; i < ref_data.size(); i++) {
        ASSERT_NEAR(static_cast<float>(ref_data[i]), expected[i], token_type_tolerance) << " at index=" << i;
    }
}

static std::vector<paged_attention_token_type_test_params> make_single_variant_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    for (const auto& data : test::PagedAttentionTokenTypeTestData::GetTestData())
        params.push_back(make_token_type_test_param(data, ENABLE_FA_V2));
    return params;
}

static std::string get_token_type_data_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_reference,
                         paged_attention_token_type_reference_test,
                         ::testing::ValuesIn(make_single_variant_test_params()),
                         get_token_type_data_name);

// --------------------------------------------------------------- MIXED over the same golden data
// Re-runs each golden sequence as a MIXED step: past = [0, P) served from the KV cache, new = [P, L)
// as the query. This is the coordinate shift the sdpa_ocl change is about -- token_type_ids stays in
// LOCAL (new-token) coordinates while keys count from the start of the cached context -- and it needs
// NO new golden data.
//
// Why the PREFILL golden output is still the right answer for the new rows: the reference builds image
// groups over the NEW tokens alone, so MIXED and PREFILL disagree only about a group that STRADDLES P
// -- MIXED truncates it to [P, group_end), PREFILL keeps [group_begin, group_end). The keys that drops
// are all < P <= query, i.e. already inside the causal term, so the two allowed sets coincide whenever
//   (a) there is no sliding window -- the causal term alone reaches back to key 0 -- or
//   (b) P sits at a group boundary.
// pick_mixed_split() picks a P satisfying one of those, preferring the middle of the sequence so both
// halves carry image tokens. A sequence that admits neither (one group covering everything, with a
// window) is skipped here; the straddling case is covered by the reference-based suite instead.
static int pick_mixed_split(const std::vector<int32_t>& token_types, int sliding_window_size) {
    const int seq_len = static_cast<int>(token_types.size());
    auto usable = [&](int split) {
        // >= 2 new tokens, or the stage is GENERATE rather than MIXED.
        if (split <= 0 || seq_len - split < 2)
            return false;
        return sliding_window_size == 0 || token_types[split - 1] != 1 || token_types[split] != 1;
    };

    const int mid = seq_len / 2;
    for (int d = 0; d <= seq_len; d++) {
        if (usable(mid - d))
            return mid - d;
        if (usable(mid + d))
            return mid + d;
    }
    return 0;
}

class paged_attention_token_type_mixed_test : public paged_attention_token_type_test {};

TEST_P(paged_attention_token_type_mixed_test, matches_prefill_golden) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    apply_token_type_test_data(pam, p, p.token_type_test_data);

    auto result = run_gpu_inference(pam, p);

    std::string entries;
    if (!ran_sdpa_ocl(result.network, entries)) {
        GTEST_SKIP() << "the MIXED bidirectional mask is implemented in sdpa_ocl only, got: " << entries;
    }

    const size_t hidden_dim = static_cast<size_t>(p.num_heads) * static_cast<size_t>(p.k_head_size);
    compare_token_type_output(result.outputs.at("output_data").get_memory(),
                              p.token_type_test_data.expectedOutput,
                              static_cast<size_t>(p.subsequences[0].past_len) * hidden_dim);
}

static std::vector<paged_attention_token_type_test_params> make_token_type_mixed_test_params() {
    std::vector<paged_attention_token_type_test_params> params;
    for (const auto& data : test::PagedAttentionTokenTypeTestData::GetTestData()) {
        const int past_len = pick_mixed_split(data.tokenTypes, data.slidingWindowSize);
        if (past_len == 0)
            continue;

        auto p = make_token_type_test_param(data, ENABLE_FA_V2);
        p.subsequences = {{static_cast<int>(data.tokenTypes.size()) - past_len, past_len}};
        p.token_type_ids = std::vector<int>(data.tokenTypes.begin() + past_len, data.tokenTypes.end());
        params.push_back(p);
    }
    return params;
}

static std::string get_token_type_mixed_test_name(const testing::TestParamInfo<paged_attention_token_type_test_params>& obj) {
    const auto& p = obj.param;
    return p.token_type_test_data.name + "_SW" + std::to_string(p.sliding_window_size) + "_past" + std::to_string(p.subsequences[0].past_len);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_token_type_mixed,
                         paged_attention_token_type_mixed_test,
                         ::testing::ValuesIn(make_token_type_mixed_test_params()),
                         get_token_type_mixed_test_name);

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
    // Either DPAS prefill generator is acceptable: which one runs is picked by TEST_USE_SDPA_OCL
    // (sdpa_ocl by default, sdpa_micro when it is 0) and both implement the bidirectional mask.
    const auto entries = dump_info.get_entries();
    EXPECT_TRUE(entries.find("sdpa_ocl") != std::string::npos || entries.find("sdpa_micro") != std::string::npos)
        << "Expected a DPAS SDPA kernel for PREFILL with token_type_ids, got: " << entries;

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

// ------------------------------------------------------- token_type_ids present but EMPTY ([B_token | 0])
// has_token_type_ids is decided at COMPILE time from an input shape that may still be dynamic (see
// plugin/ops/paged_attention.cpp), but the op contract allows the tensor itself to be EMPTY at runtime,
// meaning "no image tokens" -- intel_cpu gates on getShape().hasZeroDims() and the reference on
// count > 0. The GPU paged attention impl is shape-agnostic, so it cannot drop the feature per shape;
// sdpa_ocl instead receives the runtime element count as a scalar and skips every token_type_ids read
// when it is 0. Without that gate the kernel reads a zero-sized allocation for [0, B_token) and returns
// silently wrong output.
//
// The harness backs the empty tensor with an all-ONES [B_token] allocation whose layout is shrunk to
// [0] (see PagedAttentionManager::get_token_type_ids_memory), so a kernel that reads it anyway sees
// "every token is an image token" and cannot pass by luck. Expected output is therefore the plain
// causal one, which is what PagedAttentionReference already computes.
//
// Covers PREFILL (past_len == 0) and MIXED (past_len > 0). The gate itself is stage-independent -- the
// scalar comes from update_dispatch_data() -- but MIXED is where reading the buffer anyway is most
// damaging, because the all-ones poison then makes the whole new-token region one image group and
// un-masks its future keys.
class paged_attention_empty_token_type_ids_test : public PagedAttentionTest<paged_attention_test_params> {};

TEST_P(paged_attention_empty_token_type_ids_test, plain_causal) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    auto result = run_gpu_inference(pam, p);

    std::string entries;
    if (!ran_sdpa_ocl(result.network, entries)) {
        GTEST_SKIP() << "the empty token_type_ids gate is implemented in sdpa_ocl only, got: " << entries;
    }

    auto output_data_mem = result.outputs.at("output_data").get_memory();
    auto ref_data = PagedAttentionReference(pam).get_reference(result.key_cache_mem);
    compare(output_data_mem, nullptr, nullptr, ref_data);
}

static paged_attention_test_params make_empty_token_type_ids_param(SubsequenceDescriptor subsequence,
                                                                   int num_heads,
                                                                   int head_size,
                                                                   int sliding_window_size) {
    paged_attention_test_params p;
    p.subsequences = {subsequence};
    p.num_heads = num_heads;
    p.num_kv_heads = num_heads;
    p.k_head_size = head_size;
    p.v_head_size = head_size;
    p.block_size = 16;
    p.sliding_window_size = sliding_window_size;
    p.kv_cache_compression = DISABLE_CACHE_COMPRESSION;
    p.key_cache_quant_mode = ov::internal::CacheQuantMode::BY_TOKEN;
    p.dynamic_paddings = STATIC_INPUT_PAD;
    p.scores_mode = DISABLE_SCORES;
    p.rotation_config = DISABLE_ROTATION;
    p.disable_flashattn_v2 = ENABLE_FA_V2;
    p.empty_token_type_ids = true;
    return p;
}

static std::string get_empty_token_type_ids_test_name(const testing::TestParamInfo<paged_attention_test_params>& obj) {
    const auto& p = obj.param;
    return "tokens" + std::to_string(p.subsequences[0].num_tokens) + "_past" + std::to_string(p.subsequences[0].past_len) + "_h" +
           std::to_string(p.num_heads) + "_d" + std::to_string(p.k_head_size) + "_SW" + std::to_string(p.sliding_window_size);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention_empty_token_type_ids,
                         paged_attention_empty_token_type_ids_test,
                         ::testing::ValuesIn(std::vector<paged_attention_test_params>{
                             // causal_k extension site, representative shape
                             make_empty_token_type_ids_param({128, 0}, 2, 64, 0),
                             // window_k_begin extension site
                             make_empty_token_type_ids_param({128, 0}, 2, 64, 32),
                             // the unit-test config of the suites above: head 32, 1 head, i.e. the only
                             // one where kq_query_blocks == 2 and bidir_group_begin/end is a real array
                             make_empty_token_type_ids_param({40, 0}, 1, 32, 0),
                             make_empty_token_type_ids_param({40, 0}, 1, 32, 16),
                             // fewer queries than one query block
                             make_empty_token_type_ids_param({10, 0}, 2, 64, 0),
                             // MIXED: same sites again with past_len > 0, where the LOCAL/KEY coordinate
                             // shift is live. past_len is deliberately both block-aligned and not.
                             make_empty_token_type_ids_param({128, 64}, 2, 64, 0),
                             make_empty_token_type_ids_param({128, 64}, 2, 64, 32),
                             make_empty_token_type_ids_param({40, 37}, 1, 32, 0),
                             make_empty_token_type_ids_param({40, 37}, 1, 32, 16),
                             // window start still inside the cached prefix, i.e. window_begin_local <= 0
                             make_empty_token_type_ids_param({40, 37}, 1, 32, 64),
                             make_empty_token_type_ids_param({10, 100}, 2, 64, 0),
                         }),
                         get_empty_token_type_ids_test_name);

// ------------------------------------------------------- MIXED bidirectional attention vs. reference
// What the golden data cannot express: an image group that reaches the FIRST new token, i.e. one that
// would straddle the past/new boundary in the full sequence. The reference resolves that the same way
// the op does -- groups are built over the new tokens alone, so the group simply starts at local 0 --
// and with a sliding window that is observably different from the PREFILL answer. Also sweeps the
// shape axes the golden data fixes at 1 head / head_size 32 / single subsequence.
class paged_attention_token_type_bidir_ref_test : public PagedAttentionTest<paged_attention_test_params> {};

TEST_P(paged_attention_token_type_bidir_ref_test, matches_reference) {
    auto p = GetParam();

    ASSERT_TRUE(this->pam.has_value());
    auto& pam = *this->pam;

    auto result = run_gpu_inference(pam, p);

    std::string entries;
    if (!ran_sdpa_ocl(result.network, entries)) {
        GTEST_SKIP() << "the bidirectional mask outside PREFILL is implemented in sdpa_ocl only, got: " << entries;
    }

    auto ref_data = PagedAttentionReference(pam).get_reference(result.key_cache_mem);
    compare(result.outputs.at("output_data").get_memory(), nullptr, nullptr, ref_data);
}

// "[B_token]" image-token pattern: a run of `group_len` ones every `period` NEW tokens, the first one
// starting at `first`. first == 0 is the case of interest -- a group whose begin is the very first new
// token, which is what a chunked prefill splitting an image in half produces.
static std::vector<int> make_image_groups(int b_token, int first, int group_len, int period) {
    std::vector<int> token_type_ids(b_token, 0);
    for (int start = first; start < b_token; start += period)
        for (int i = start; i < std::min(start + group_len, b_token); i++)
            token_type_ids[i] = 1;
    return token_type_ids;
}

static paged_attention_test_params make_bidir_ref_param(std::vector<SubsequenceDescriptor> subsequences,
                                                        int num_heads,
                                                        int num_kv_heads,
                                                        int head_size,
                                                        int sliding_window_size,
                                                        std::vector<int> token_type_ids) {
    paged_attention_test_params p;
    p.subsequences = std::move(subsequences);
    p.num_heads = num_heads;
    p.num_kv_heads = num_kv_heads;
    p.k_head_size = head_size;
    p.v_head_size = head_size;
    p.block_size = 16;
    p.sliding_window_size = sliding_window_size;
    p.kv_cache_compression = DISABLE_CACHE_COMPRESSION;
    p.key_cache_quant_mode = ov::internal::CacheQuantMode::BY_TOKEN;
    p.dynamic_paddings = STATIC_INPUT_PAD;
    p.scores_mode = DISABLE_SCORES;
    p.rotation_config = DISABLE_ROTATION;
    p.disable_flashattn_v2 = ENABLE_FA_V2;
    p.token_type_ids = std::move(token_type_ids);
    return p;
}

static std::string get_bidir_ref_test_name(const testing::TestParamInfo<paged_attention_test_params>& obj) {
    const auto& p = obj.param;
    std::string name = "seq";
    for (const auto& subsequence : p.subsequences)
        name += std::to_string(subsequence.num_tokens) + "p" + std::to_string(subsequence.past_len) + "_";
    name += "h" + std::to_string(p.num_heads) + "kv" + std::to_string(p.num_kv_heads) + "_d" + std::to_string(p.k_head_size) + "_SW" +
            std::to_string(p.sliding_window_size);
    // Several cases share a shape and differ only in the image-token pattern, which does not fit in a
    // test name; the list index disambiguates them and points straight at the offending entry.
    return name + "_c" + std::to_string(obj.index);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_paged_attention_token_type_bidir_ref,
    paged_attention_token_type_bidir_ref_test,
    ::testing::ValuesIn(std::vector<paged_attention_test_params>{
        // --- the straddling group: it begins at new token 0, so PREFILL golden data cannot describe it
        make_bidir_ref_param({{64, 48}}, 2, 2, 64, 0, make_image_groups(64, 0, 20, 32)),
        make_bidir_ref_param({{64, 48}}, 2, 2, 64, 16, make_image_groups(64, 0, 20, 32)),
        // window shorter than the group, so window_k_begin has to be pulled back into it
        make_bidir_ref_param({{64, 48}}, 2, 2, 64, 8, make_image_groups(64, 0, 40, 64)),
        // --- group running to the very last new token: exercises the causal_k extension clamp at q
        make_bidir_ref_param({{64, 48}}, 2, 2, 64, 16, make_image_groups(64, 40, 24, 64)),
        // --- every new token is an image token
        make_bidir_ref_param({{48, 80}}, 2, 2, 64, 16, std::vector<int>(48, 1)),
        // --- shape sweep: head sizes, GQA, non-block-aligned past_len, no window
        make_bidir_ref_param({{40, 37}}, 1, 1, 32, 0, make_image_groups(40, 0, 12, 20)),
        make_bidir_ref_param({{40, 37}}, 1, 1, 32, 16, make_image_groups(40, 5, 12, 20)),
        make_bidir_ref_param({{72, 55}}, 8, 2, 128, 32, make_image_groups(72, 3, 18, 30)),
        make_bidir_ref_param({{72, 55}}, 8, 2, 128, 0, make_image_groups(72, 3, 18, 30)),
        // --- several subsequences: token_type_ids is one flat [B_token] buffer, so the kernel's
        //     `token_type_ids += subsequence_begin` bump has to be right for every one of them
        make_bidir_ref_param({{20, 33}, {36, 0}, {12, 64}}, 2, 2, 64, 0, make_image_groups(68, 0, 9, 14)),
        make_bidir_ref_param({{20, 33}, {36, 0}, {12, 64}}, 2, 2, 64, 16, make_image_groups(68, 2, 9, 14)),
        // --- PREFILL controls on the same shapes: proves these cases fail for a MIXED-specific reason
        make_bidir_ref_param({{112, 0}}, 2, 2, 64, 16, make_image_groups(112, 0, 20, 32)),
        make_bidir_ref_param({{68, 0}}, 2, 2, 64, 0, make_image_groups(68, 0, 9, 14)),
    }),
    get_bidir_ref_test_name);
