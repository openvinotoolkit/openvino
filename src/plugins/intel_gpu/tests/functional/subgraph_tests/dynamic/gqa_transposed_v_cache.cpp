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
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/core.hpp"

// GroupQueryAttention executed end to end (ONNX com.microsoft GroupQueryAttention -> the GPU GroupQueryAttentionDecomposition
// -> StatelessKV + SDPA or the explicit attention chain), against the TEMPLATE plugin. transpose_v: the model keeps its V cache
// as [B, kv_heads, head_size, S] (past_value -> Transpose{0,1,3,2} -> GQA -> Transpose{0,1,3,2} -> present_value). The
// decomposition used to hand that layout to SDPA as value order {0,1,3,2}, which sdpa_ocl and sdpa_opt read as if the head dim
// were contiguous (wrong output), and the explicit attention / quantized KV paths kept StatelessKV -> Transpose -> Result,
// which the runtime rejects. The plain [B, kv_heads, S, head_size] layout is the control.
namespace {

using QKOutput = ov::op::internal::GroupQueryAttentionQKOutput;
using QuantType = ov::op::internal::GroupQueryAttentionQuantType;

enum class Attention { SDPA, SOFTCAP, QK_OUTPUT, I8_KV };

struct GQAVCacheParams {
    size_t head_size;
    bool transpose_v;
    Attention attention;
};

constexpr size_t num_heads = 8;
constexpr size_t kv_num_heads = 2;

std::shared_ptr<ov::Model> make_gqa_model(const GQAVCacheParams& p) {
    const auto f16 = ov::element::f16;
    const bool i8_kv = p.attention == Attention::I8_KV;
    const auto cache_type = i8_kv ? ov::element::i8 : f16;
    const auto D = static_cast<int64_t>(p.head_size);
    const auto H = static_cast<int64_t>(num_heads);
    const auto KVH = static_cast<int64_t>(kv_num_heads);

    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, H, -1, D});
    auto key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, KVH, -1, D});
    auto value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, KVH, -1, D});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(cache_type, ov::PartialShape{1, KVH, -1, D});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(cache_type, p.transpose_v ? ov::PartialShape{1, KVH, D, -1} : ov::PartialShape{1, KVH, -1, D});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    const auto transpose_v_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});
    ov::OutputVector inputs(14);
    inputs[0] = query;
    inputs[1] = key;
    inputs[2] = value;
    inputs[3] = past_key;
    inputs[4] = p.transpose_v ? std::make_shared<ov::op::v1::Transpose>(past_value, transpose_v_order)->output(0) : past_value->output(0);
    inputs[5] = seqlens_k;
    inputs[6] = total_sequence_length;
    for (size_t i = 7; i <= 13; ++i) {
        inputs[i] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    }
    if (i8_kv) {
        // Per-channel scales, different per channel so a K/V channel mix-up shows; |V| <= 1 stays well inside i8.
        std::vector<float> scales(kv_num_heads * p.head_size);
        for (size_t c = 0; c < scales.size(); ++c) {
            scales[c] = 0.01f * (1.0f + 0.1f * static_cast<float>(c % 7));
        }
        inputs[12] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{scales.size()}, scales);
        inputs[13] = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{scales.size()}, scales);
    }

    const auto quant = i8_kv ? QuantType::PER_CHANNEL : QuantType::NONE;
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs,
                                                                       H,
                                                                       KVH,
                                                                       0.0f,
                                                                       false,
                                                                       false,
                                                                       i8_kv ? 8 : 0,
                                                                       quant,
                                                                       quant,
                                                                       -1,
                                                                       false,
                                                                       false,
                                                                       true,
                                                                       p.attention == Attention::SOFTCAP ? 30.0f : 0.0f,
                                                                       1e-6f,
                                                                       p.attention == Attention::QK_OUTPUT ? QKOutput::AFTER_SOFTMAX : QKOutput::NONE);

    ov::ResultVector results;
    for (size_t i = 0; i < gqa->get_output_size(); ++i) {
        ov::Output<ov::Node> output = gqa->output(i);
        if (p.transpose_v && i == 2) {
            output = std::make_shared<ov::op::v1::Transpose>(output, transpose_v_order);
        }
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length});
}

ov::Tensor random_tensor(const ov::element::Type& type, const ov::Shape& shape, std::mt19937& gen) {
    ov::Tensor tensor(type, shape);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    if (type == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            data[i] = ov::float16(dist(gen));
        }
    } else {
        // i8 cache entries
        std::uniform_int_distribution<int> idist(-100, 100);
        auto* data = tensor.data<int8_t>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            data[i] = static_cast<int8_t>(idist(gen));
        }
    }
    return tensor;
}

float element(const ov::Tensor& tensor, size_t i) {
    if (tensor.get_element_type() == ov::element::f16) {
        return static_cast<float>(tensor.data<const ov::float16>()[i]);
    }
    if (tensor.get_element_type() == ov::element::i8) {
        return static_cast<float>(tensor.data<const int8_t>()[i]);
    }
    return tensor.data<const float>()[i];
}

class GQATransposedVCacheTest : public ::testing::TestWithParam<GQAVCacheParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<GQAVCacheParams>& info) {
        const auto& p = info.param;
        const char* names[] = {"sdpa", "softcap", "qk_output", "i8_kv"};
        return "D" + std::to_string(p.head_size) + (p.transpose_v ? "_transposedV_" : "_plainV_") + names[static_cast<int>(p.attention)];
    }
};

TEST_P(GQATransposedVCacheTest, MatchesTemplate) {
    const auto& p = GetParam();
    auto core = ov::test::utils::PluginCache::get().core();
    const auto model = make_gqa_model(p);
    auto gpu_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto ref_model = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    auto gpu = gpu_model.create_infer_request();
    auto ref = ref_model.create_infer_request();

    const bool i8_kv = p.attention == Attention::I8_KV;
    const auto cache_type = i8_kv ? ov::element::i8 : ov::element::f16;
    std::mt19937 gen(static_cast<uint32_t>(p.head_size * 10 + static_cast<size_t>(p.attention) * 2 + (p.transpose_v ? 1 : 0)));
    // {past, current}: prefill, decode, chunked prefill on a past.
    const std::vector<std::pair<size_t, size_t>> steps = {{0, 17}, {17, 1}, {18, 40}};
    for (const auto& [past, current] : steps) {
        SCOPED_TRACE("past " + std::to_string(past) + ", current " + std::to_string(current));
        const size_t D = p.head_size;
        std::vector<ov::Tensor> inputs = {
            random_tensor(ov::element::f16, {1, num_heads, current, D}, gen),
            random_tensor(ov::element::f16, {1, kv_num_heads, current, D}, gen),
            random_tensor(ov::element::f16, {1, kv_num_heads, current, D}, gen),
            random_tensor(cache_type, {1, kv_num_heads, past, D}, gen),
            random_tensor(cache_type, p.transpose_v ? ov::Shape{1, kv_num_heads, D, past} : ov::Shape{1, kv_num_heads, past, D}, gen),
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
            // Present caches (outputs 1, 2) are the past bytes plus the appended tokens: exact for f16, and for i8 the current
            // tokens' quantization may round one step apart (f16 vs f32 math). Attention output / QK scores: f16 compute.
            const bool cache_output = o == 1 || o == 2;
            const float tolerance = cache_output ? (i8_kv ? 1.0f : 0.0f) : (i8_kv ? 0.03f : 0.02f);
            size_t mismatches = 0;
            float max_diff = 0.0f;
            size_t first_mismatch = 0;
            for (size_t i = 0; i < expected.get_size(); ++i) {
                const float diff = std::fabs(element(expected, i) - element(actual, i));
                if (!(diff <= tolerance)) {  // NaN counts as a mismatch
                    if (mismatches++ == 0) {
                        first_mismatch = i;
                    }
                }
                if (diff > max_diff || std::isnan(diff)) {
                    max_diff = diff;
                }
            }
            EXPECT_EQ(mismatches, 0u) << "output " << o << ": " << mismatches << " of " << expected.get_size() << " elements off by more than "
                                      << tolerance << ", max diff " << max_diff << ", first at " << first_mismatch << " (expected "
                                      << element(expected, first_mismatch) << ", actual " << element(actual, first_mismatch) << ")";
        }
    }
}

std::vector<GQAVCacheParams> make_params() {
    std::vector<GQAVCacheParams> params;
    for (const size_t head_size : {64, 128}) {
        for (const bool transpose_v : {true, false}) {
            for (const auto attention : {Attention::SDPA, Attention::SOFTCAP, Attention::QK_OUTPUT, Attention::I8_KV}) {
                params.push_back({head_size, transpose_v, attention});
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_GQATransposedVCache,
                         GQATransposedVCacheTest,
                         ::testing::ValuesIn(make_params()),
                         GQATransposedVCacheTest::get_test_case_name);

}  // namespace
