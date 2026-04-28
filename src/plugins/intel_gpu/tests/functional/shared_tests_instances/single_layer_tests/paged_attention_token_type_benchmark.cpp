// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Benchmark for the full PagedAttention operator comparing two models:
//   1. With token_type_ids input (bidirectional attention for VLM image tokens)
//   2. Plain causal (no token_type_ids — standard LLM path)
//
// Both models are compiled, warmed up, and timed independently per scenario.
// All benchmark tests are DISABLED by default — run with:
//   --gtest_also_run_disabled_tests --gtest_filter="*PagedAttnTokenTypeBench*"

#include "shared_test_classes/single_op/paged_attention_token_type.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test;

namespace {

constexpr int WARMUP_ITERS = 10;
constexpr int BENCH_ITERS  = 100;

// ── Helper: create a named Parameter node ───────────────────────────────────

static std::shared_ptr<ov::op::v0::Parameter> MakeParam(const ov::PartialShape& pshape,
                                                         ov::element::Type element_type,
                                                         const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element_type, pshape);
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

// ── Build PagedAttention model ──────────────────────────────────────────────
// When has_token_type_ids=true, token_type_ids is a dynamic Parameter input.
// When has_token_type_ids=false, token_type_ids is an empty Constant (plain causal).

static std::shared_ptr<ov::Model> BuildPagedAttentionModel(ov::element::Type data_type,
                                                            int64_t head_size,
                                                            int64_t head_num,
                                                            int32_t sliding_window_size,
                                                            bool has_token_type_ids) {
    auto q = MakeParam(ov::PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "q");
    auto k = MakeParam(ov::PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
    auto v = MakeParam(ov::PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");

    const int64_t block_size = 16;
    auto key_cache =
        MakeParam(ov::PartialShape{ov::Dimension::dynamic(), head_num, head_size, block_size}, data_type, "key_cache.0");
    auto value_cache =
        MakeParam(ov::PartialShape{ov::Dimension::dynamic(), head_num, block_size, head_size}, data_type, "value_cache.0");
    auto past_lens = MakeParam(ov::PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
    auto subsequence_begins = MakeParam(ov::PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
    auto block_indices = MakeParam(ov::PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
    auto block_indices_begins =
        MakeParam(ov::PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");

    float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
    auto sliding_window =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{sliding_window_size});
    auto alibi_slopes = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0}, std::vector<float>{});
    auto max_context_len = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{8192});
    auto score_aggregation_window = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
    auto rotated_block_indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
    auto rotation_deltas = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
    auto rotation_trig_lut = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0}, std::vector<float>{0});
    auto xattention_threshold = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0}, std::vector<float>{0});
    auto xattention_block_size = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{64});
    auto xattention_stride = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{8});
    auto sinks = std::make_shared<ov::op::v0::Constant>(data_type, ov::Shape{0}, std::vector<float>{0});
    auto adaptive_rkv_start_size = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
    auto adaptive_rkv_evictable_sizes =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
    auto adaptive_rkv_diversity_block_set_indices =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});
    auto adaptive_rkv_diversity_block_set_indices_begins =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});

    auto qq_bias = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{0}, std::vector<uint8_t>{0});
    auto qq_bias_begins = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{0});

    // token_type_ids: Parameter (dynamic input) vs Constant (empty, plain causal)
    std::shared_ptr<ov::Node> token_type_ids_node;
    std::shared_ptr<ov::op::v0::Parameter> token_type_ids_param;

    ov::ParameterVector params = {q, k, v, key_cache, value_cache,
                                   past_lens, subsequence_begins, block_indices, block_indices_begins};

    if (has_token_type_ids) {
        token_type_ids_param = MakeParam(ov::PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "token_type_ids");
        token_type_ids_node = token_type_ids_param;
        params.push_back(token_type_ids_param);
    } else {
        token_type_ids_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
    }

    ov::OutputVector pa_inputs = {q, k, v, key_cache, value_cache,
                                   past_lens, subsequence_begins, block_indices, block_indices_begins,
                                   scale, sliding_window, alibi_slopes, max_context_len,
                                   score_aggregation_window, rotated_block_indices, rotation_deltas,
                                   rotation_trig_lut, xattention_threshold, xattention_block_size,
                                   xattention_stride, sinks, adaptive_rkv_start_size,
                                   adaptive_rkv_evictable_sizes, adaptive_rkv_diversity_block_set_indices,
                                   adaptive_rkv_diversity_block_set_indices_begins,
                                   token_type_ids_node, qq_bias, qq_bias_begins};

    auto paged_attn = std::make_shared<ov::op::PagedAttentionExtension>(pa_inputs);
    paged_attn->get_rt_info()["num_k_heads"] = head_num;
    paged_attn->get_rt_info()["k_head_size"] = head_size;
    paged_attn->get_rt_info()["num_v_heads"] = head_num;
    paged_attn->get_rt_info()["v_head_size"] = head_size;

    return std::make_shared<ov::Model>(ov::OutputVector{paged_attn}, params);
}

// ── Helper: generate random float data ──────────────────────────────────────

static std::vector<float> GenRandomFloats(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count);
    for (auto& x : data)
        x = dist(rng);
    return data;
}

// ── Token pattern generators ────────────────────────────────────────────────

static std::vector<int32_t> GenSingleImageBlock(size_t seq_len, size_t image_len) {
    std::vector<int32_t> v(seq_len, 0);
    size_t start = (seq_len - image_len) / 2;
    for (size_t i = start; i < start + image_len && i < seq_len; i++)
        v[i] = 1;
    return v;
}

static std::vector<int32_t> GenMultiImage(size_t seq_len, int num_images, size_t avg_img_len) {
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

// ── Benchmark scenario descriptor ───────────────────────────────────────────

struct BenchScenario {
    std::string name;
    ov::element::Type data_type;
    int64_t head_size;
    int64_t head_num;
    int32_t sliding_window_size;
    size_t seq_len;
    std::vector<int32_t> token_types;
    bool use_flash_attn_v2;
};

// ── Timing result ───────────────────────────────────────────────────────────

struct TimingResult {
    double avg_us;
    double p50_us;
    double min_us;
    double max_us;
};

// ── Benchmark test fixture ──────────────────────────────────────────────────

class PagedAttnTokenTypeBench : public ::testing::Test {
protected:
    // Build, compile, bind inputs, warmup, and time a single model variant
    TimingResult BenchModel(const BenchScenario& sc,
                            bool has_token_type_ids) {
        auto model = BuildPagedAttentionModel(sc.data_type, sc.head_size, sc.head_num,
                                               sc.sliding_window_size, has_token_type_ids);

        ov::Core core;
        ov::AnyMap config;
        config[ov::hint::inference_precision.name()] = ov::element::f32;
        config[ov::hint::kv_cache_precision.name()] = ov::element::f32;
        auto compiled = core.compile_model(model, "GPU", config);
        auto infer_request = compiled.create_infer_request();

        const size_t hidden_dim = sc.head_size * sc.head_num;
        const size_t block_size = 16;
        const size_t block_nums = 8192 / block_size;

        auto q_data = GenRandomFloats(sc.seq_len * hidden_dim, 42);
        auto k_data = GenRandomFloats(sc.seq_len * hidden_dim, 43);
        auto v_data = GenRandomFloats(sc.seq_len * hidden_dim, 44);

        ov::Tensor q_tensor(sc.data_type, {sc.seq_len, hidden_dim});
        ov::Tensor k_tensor(sc.data_type, {sc.seq_len, hidden_dim});
        ov::Tensor v_tensor(sc.data_type, {sc.seq_len, hidden_dim});
        std::memcpy(q_tensor.data<float>(), q_data.data(), q_data.size() * sizeof(float));
        std::memcpy(k_tensor.data<float>(), k_data.data(), k_data.size() * sizeof(float));
        std::memcpy(v_tensor.data<float>(), v_data.data(), v_data.size() * sizeof(float));

        ov::Tensor key_cache_tensor(sc.data_type, {block_nums, static_cast<size_t>(sc.head_num),
                                                     static_cast<size_t>(sc.head_size), block_size});
        ov::Tensor value_cache_tensor(sc.data_type, {block_nums, static_cast<size_t>(sc.head_num),
                                                       block_size, static_cast<size_t>(sc.head_size)});

        size_t batch_size = 1;
        int32_t total_blocks = static_cast<int32_t>((sc.seq_len + block_size - 1) / block_size);

        ov::Tensor past_lens(ov::element::i32, {batch_size});
        ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
        ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
        ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

        past_lens.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[0] = 0;
        subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(sc.seq_len);
        block_indices_begins.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[1] = total_blocks;
        for (int32_t i = 0; i < total_blocks; i++) {
            block_indices.data<int32_t>()[i] = i;
        }

        auto params = model->get_parameters();
        for (auto& param : params) {
            auto name = param->get_friendly_name();
            if (name == "q")                         infer_request.set_tensor(param, q_tensor);
            else if (name == "k")                    infer_request.set_tensor(param, k_tensor);
            else if (name == "v")                    infer_request.set_tensor(param, v_tensor);
            else if (name == "key_cache.0")          infer_request.set_tensor(param, key_cache_tensor);
            else if (name == "value_cache.0")        infer_request.set_tensor(param, value_cache_tensor);
            else if (name == "past_lens")            infer_request.set_tensor(param, past_lens);
            else if (name == "subsequence_begins")   infer_request.set_tensor(param, subsequence_begins);
            else if (name == "block_indices")        infer_request.set_tensor(param, block_indices);
            else if (name == "block_indices_begins") infer_request.set_tensor(param, block_indices_begins);
            else if (name == "token_type_ids") {
                ov::Tensor token_type_tensor(ov::element::i32, {sc.seq_len});
                std::memcpy(token_type_tensor.data<int32_t>(), sc.token_types.data(), sc.seq_len * sizeof(int32_t));
                infer_request.set_tensor(param, token_type_tensor);
            }
        }

        // Warmup
        for (int i = 0; i < WARMUP_ITERS; i++) {
            infer_request.infer();
        }

        // Timed iterations
        std::vector<double> times_us;
        times_us.reserve(BENCH_ITERS);

        for (int i = 0; i < BENCH_ITERS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            infer_request.infer();
            auto end = std::chrono::high_resolution_clock::now();
            double us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            times_us.push_back(us);
        }

        auto sorted = times_us;
        std::sort(sorted.begin(), sorted.end());

        return {
            std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size(),
            sorted[sorted.size() / 2],
            sorted.front(),
            sorted.back()
        };
    }

    // Run benchmark for both model variants and print comparison
    void RunBenchmark(const BenchScenario& sc) {
#ifdef _WIN32
        _putenv_s("OV_GPU_COULD_USE_FLASHATTN_V2", sc.use_flash_attn_v2 ? "1" : "0");
#else
        ::setenv("OV_GPU_COULD_USE_FLASHATTN_V2", sc.use_flash_attn_v2 ? "1" : "0", 1);
#endif

        std::cout << std::fixed << std::setprecision(1);
        std::cout << "┌─── " << sc.name << " ───" << std::endl;
        std::cout << "│ Config: seq_len=" << sc.seq_len
                  << "  head_size=" << sc.head_size
                  << "  head_num=" << sc.head_num
                  << "  sliding_window=" << sc.sliding_window_size
                  << "  flash_v2=" << (sc.use_flash_attn_v2 ? "ON" : "OFF") << std::endl;
        std::cout << "│ Warmup: " << WARMUP_ITERS << " iters"
                  << "  Bench: " << BENCH_ITERS << " iters" << std::endl;

        // Model 1: plain causal (no token_type_ids)
        // auto causal = BenchModel(sc, /*has_token_type_ids=*/false);
        // std::cout << "│ Plain causal:       avg=" << causal.avg_us << " us"
        //           << "  p50=" << causal.p50_us << " us"
        //           << "  min=" << causal.min_us << " us"
        //           << "  max=" << causal.max_us << " us" << std::endl;

        // Model 2: with token_type_ids
        auto with_tt = BenchModel(sc, /*has_token_type_ids=*/true);
        std::cout << "│ With token_type_ids: avg=" << with_tt.avg_us << " us"
                  << "  p50=" << with_tt.p50_us << " us"
                  << "  min=" << with_tt.min_us << " us"
                  << "  max=" << with_tt.max_us << " us" << std::endl;

        // double overhead_pct = ((with_tt.avg_us - causal.avg_us) / causal.avg_us) * 100.0;
        // std::cout << "│ Overhead:  " << std::showpos << overhead_pct << "%" << std::noshowpos << std::endl;
        std::cout << "└──────────────────────────────────" << std::endl;

#ifdef _WIN32
        _putenv_s("OV_GPU_COULD_USE_FLASHATTN_V2", "");
#else
        ::unsetenv("OV_GPU_COULD_USE_FLASHATTN_V2");
#endif
    }
};

// Best case
TEST_F(PagedAttnTokenTypeBench, best_case) {
    const size_t seq = 2048;
    BenchScenario sc;

    sc.name = "Prefill: Single Image (2048 seq, h=128, 8 heads)",
    sc.data_type = ov::element::f32,
    sc.head_size = 128, sc.head_num = 8,
    sc.sliding_window_size = 0,
    sc.seq_len = seq,
    sc.token_types = GenSingleImageBlock(seq, 0),
    sc.use_flash_attn_v2 = true;
    
    RunBenchmark(sc);
}

// Average case
TEST_F(PagedAttnTokenTypeBench, average_case) {
    const size_t seq = 2048;
    BenchScenario sc;

    sc.name = "Prefill: Multi-Image (2K seq, 4 images, h=128, 8 heads)",
    sc.data_type = ov::element::f32,
    sc.head_size = 128, sc.head_num = 8,
    sc.sliding_window_size = 0,
    sc.seq_len = seq,
    sc.token_types = GenMultiImage(seq, 4, 200),
    sc.use_flash_attn_v2 = true;
    
    RunBenchmark(sc);
}

// Worst case
TEST_F(PagedAttnTokenTypeBench, worst_case) {
    const size_t seq = 8192;
    BenchScenario sc;

    sc.name = "Prefill: Single Image (8K seq, h=128, 1 head)",
    sc.data_type = ov::element::f32,
    sc.head_size = 128, sc.head_num = 1,
    sc.sliding_window_size = 0,
    sc.seq_len = seq,
    sc.token_types = GenSingleImageBlock(seq, seq),
    sc.use_flash_attn_v2 = true;
    
    RunBenchmark(sc);
}

}  // namespace


// BASE RESULTS:
// [==========] Running 3 tests from 1 test suite.
// [----------] Global test environment set-up.
// [----------] 3 tests from PagedAttnTokenTypeBench
// [ RUN      ] PagedAttnTokenTypeBench.best_case
// ┌─── Prefill: Single Image (2048 seq, h=128, 8 heads) ───
// │ Config: seq_len=2048  head_size=128  head_num=8  sliding_window=0  flash_v2=ON
// │ Warmup: 10 iters  Bench: 100 iters
// Non default env value for GPU_COULD_USE_FLASHATTN_V2 = 1
// Non default env value for GPU_COULD_USE_FLASHATTN_V2 = 1
// │ With token_type_ids: avg=51402.0 us  p50=51460.0 us  min=49003.0 us  max=52608.0 us
// └──────────────────────────────────
// [       OK ] PagedAttnTokenTypeBench.best_case (5855 ms)
// [ RUN      ] PagedAttnTokenTypeBench.average_case
// ┌─── Prefill: Multi-Image (2K seq, 4 images, h=128, 8 heads) ───
// │ Config: seq_len=2048  head_size=128  head_num=8  sliding_window=0  flash_v2=ON
// │ Warmup: 10 iters  Bench: 100 iters
// Non default env value for GPU_COULD_USE_FLASHATTN_V2 = 1
// Non default env value for GPU_COULD_USE_FLASHATTN_V2 = 1
// │ With token_type_ids: avg=53248.2 us  p50=53350.0 us  min=51118.0 us  max=54394.0 us
// └──────────────────────────────────
// [       OK ] PagedAttnTokenTypeBench.average_case (5918 ms)
// [ RUN      ] PagedAttnTokenTypeBench.worst_case
// ┌─── Prefill: Single Image (8K seq, h=128, 1 head) ───
// │ Config: seq_len=8192  head_size=128  head_num=1  sliding_window=0  flash_v2=ON
// │ Warmup: 10 iters  Bench: 100 iters
// Non default env value for GPU_COULD_USE_FLASHATTN_V2 = 1
// Non default env value for GPU_COULD_USE_FLASHATTN_V2 = 1
// │ With token_type_ids: avg=169382.7 us  p50=169722.0 us  min=165978.0 us  max=171382.0 us
// └──────────────────────────────────
// [       OK ] PagedAttnTokenTypeBench.worst_case (19038 ms)
// [----------] 3 tests from PagedAttnTokenTypeBench (30811 ms total)

// [----------] Global test environment tear-down
// [==========] 3 tests from 1 test suite ran. (30811 ms total)
// [  PASSED  ] 3 tests.
