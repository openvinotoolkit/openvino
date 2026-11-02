// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

// GroupQueryAttention with a sliding window (local_window_size) executed end to end against the TEMPLATE plugin. With
// a windowed KV cache (sliding_window_cache), causal attention and no scale / bias / sink, the GPU
// GroupQueryAttentionDecomposition drops the explicit mask and gives SDPA is_causal + sliding_window, so the kernel has
// to apply the window itself (sdpa_ocl dropped it: its plain-SDPA jit never defined SLIDING_WINDOW_SIZE). The other two
// paths keep the explicit causal + window mask (SDPA is_causal = false) and are the controls: the windowed cache with an
// explicit scale, and a full-length cache.
namespace {

enum class Path { NATIVE_WINDOW, WINDOWED_CACHE_MASK, FULL_CACHE_MASK };

struct GQAWindowParams {
    size_t head_size;
    Path path;
};

constexpr size_t num_heads = 8;
constexpr size_t kv_num_heads = 2;
constexpr int64_t window = 16;
constexpr size_t capacity = 24;  // windowed cache rows (>= window); evicts in blocks of capacity - window + 1

bool windowed_cache(const GQAWindowParams& p) {
    return p.path != Path::FULL_CACHE_MASK;
}

std::shared_ptr<ov::Model> make_gqa_model(const GQAWindowParams& p) {
    const auto f16 = ov::element::f16;
    const auto D = static_cast<int64_t>(p.head_size);
    const auto H = static_cast<int64_t>(num_heads);
    const auto KVH = static_cast<int64_t>(kv_num_heads);
    const auto cache_len = windowed_cache(p) ? ov::Dimension(static_cast<int64_t>(capacity)) : ov::Dimension::dynamic();

    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, H, -1, D});
    auto key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, KVH, -1, D});
    auto value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, KVH, -1, D});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, KVH, cache_len, D});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, KVH, cache_len, D});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    ov::OutputVector inputs(14);
    inputs[0] = query;
    inputs[1] = key;
    inputs[2] = value;
    inputs[3] = past_key;
    inputs[4] = past_value;
    inputs[5] = seqlens_k;
    inputs[6] = total_sequence_length;
    for (size_t i = 7; i <= 13; ++i) {
        inputs[i] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    }
    // An explicit scale (the default value) keeps the explicit mask on the windowed cache.
    const float scale = p.path == Path::WINDOWED_CACHE_MASK ? 1.0f / std::sqrt(static_cast<float>(p.head_size)) : 0.0f;
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs,
                                                                       H,
                                                                       KVH,
                                                                       scale,
                                                                       false,
                                                                       false,
                                                                       0,
                                                                       ov::op::internal::GroupQueryAttentionQuantType::NONE,
                                                                       ov::op::internal::GroupQueryAttentionQuantType::NONE,
                                                                       window,
                                                                       windowed_cache(p));
    ov::ResultVector results;
    for (size_t i = 0; i < gqa->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(i)));
    }
    return std::make_shared<ov::Model>(results, ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length});
}

ov::Tensor random_tensor(const ov::Shape& shape, std::mt19937& gen) {
    ov::Tensor tensor(ov::element::f16, shape);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = ov::float16(dist(gen));
    }
    return tensor;
}

class GQASlidingWindowTest : public ::testing::TestWithParam<GQAWindowParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<GQAWindowParams>& info) {
        const auto& p = info.param;
        const char* names[] = {"windowedCache_nativeWindow", "windowedCache_explicitMask", "fullCache_explicitMask"};
        return "D" + std::to_string(p.head_size) + "_" + names[static_cast<int>(p.path)];
    }
};

TEST_P(GQASlidingWindowTest, MatchesTemplate) {
    const auto& p = GetParam();
    auto core = ov::test::utils::PluginCache::get().core();
    const auto model = make_gqa_model(p);
    auto gpu_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto ref_model = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    auto gpu = gpu_model.create_infer_request();
    auto ref = ref_model.create_infer_request();

    std::mt19937 gen(static_cast<uint32_t>(p.head_size * 10 + static_cast<size_t>(p.path)));
    // {past, current}: a prefill longer than the window and the capacity, two decodes past the window, a chunk on a
    // past. Every step has keys older than the window for some query.
    const std::vector<std::pair<size_t, size_t>> steps = {{0, 40}, {40, 1}, {41, 1}, {42, 20}};
    for (const auto& [past, current] : steps) {
        SCOPED_TRACE("past " + std::to_string(past) + ", current " + std::to_string(current));
        const size_t D = p.head_size;
        // The windowed cache is always `capacity` rows; the decomposition derives the resident rows from seqlens_k.
        const size_t past_rows = windowed_cache(p) ? capacity : past;
        std::vector<ov::Tensor> inputs = {
            random_tensor({1, num_heads, current, D}, gen),
            random_tensor({1, kv_num_heads, current, D}, gen),
            random_tensor({1, kv_num_heads, current, D}, gen),
            random_tensor({1, kv_num_heads, past_rows, D}, gen),
            random_tensor({1, kv_num_heads, past_rows, D}, gen),
            ov::Tensor(ov::element::i32, ov::Shape{1}),
            ov::Tensor(ov::element::i32, ov::Shape{}),
        };
        inputs[5].data<int32_t>()[0] = static_cast<int32_t>(past + current - 1);
        inputs[6].data<int32_t>()[0] = static_cast<int32_t>(past + current);
        for (size_t i = 0; i < inputs.size(); ++i) {
            gpu.set_input_tensor(i, inputs[i]);
            ref.set_input_tensor(i, inputs[i]);
        }
        gpu.infer();
        ref.infer();

        for (size_t o = 0; o < model->get_output_size(); ++o) {
            const auto expected = ref.get_output_tensor(o);
            const auto actual = gpu.get_output_tensor(o);
            ASSERT_EQ(expected.get_shape(), actual.get_shape()) << "output " << o;
            ASSERT_EQ(expected.get_element_type(), actual.get_element_type()) << "output " << o;
            // Present caches (outputs 1, 2) are copied rows: exact. Attention output: f16 compute.
            const float tolerance = o == 0 ? 0.02f : 0.0f;
            const auto* e = expected.data<const ov::float16>();
            const auto* a = actual.data<const ov::float16>();
            size_t mismatches = 0;
            float max_diff = 0.0f;
            size_t first_mismatch = 0;
            for (size_t i = 0; i < expected.get_size(); ++i) {
                const float diff = std::fabs(static_cast<float>(e[i]) - static_cast<float>(a[i]));
                if (!(diff <= tolerance)) {  // NaN counts as a mismatch
                    if (mismatches++ == 0) {
                        first_mismatch = i;
                    }
                }
                if (diff > max_diff || std::isnan(diff)) {
                    max_diff = diff;
                }
            }
            EXPECT_EQ(mismatches, 0u) << "output " << o << ": " << mismatches << " of " << expected.get_size() << " elements off by more than " << tolerance
                                      << ", max diff " << max_diff << ", first at " << first_mismatch << " (expected " << static_cast<float>(e[first_mismatch])
                                      << ", actual " << static_cast<float>(a[first_mismatch]) << ")";
        }
    }
}

std::vector<GQAWindowParams> make_params() {
    std::vector<GQAWindowParams> params;
    for (const size_t head_size : {64, 128}) {
        for (const auto path : {Path::NATIVE_WINDOW, Path::WINDOWED_CACHE_MASK, Path::FULL_CACHE_MASK}) {
            params.push_back({head_size, path});
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_GQASlidingWindow, GQASlidingWindowTest, ::testing::ValuesIn(make_params()), GQASlidingWindowTest::get_test_case_name);

}  // namespace
