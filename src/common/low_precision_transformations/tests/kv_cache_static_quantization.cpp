// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/llm_builders.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/kv_cache_concat.hpp"
#include "low_precision/move_fake_convert_up_through_kv_cache_concat.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/op_conversions/fake_convert_decomposition.hpp"

using namespace ov;
using namespace ov::op;
using namespace testing;
using namespace ov::test::utils;

enum class SDPAType {
    Standard,
    WithMask,
    WithMaskAndScale,
};

using KVCacheStaticQuantizationParams = std::tuple<ov::element::Type,  // original precision
                                                   ov::element::Type,  // low precision
                                                   SDPAType,
                                                   size_t,  // num_groups
                                                   bool,    // stateful
                                                   float,   // scale
                                                   float>;  // shift

class KVCacheStaticQuantization : public TransformationTestsF,
                                  public WithParamInterface<KVCacheStaticQuantizationParams> {
public:
    KVCacheStaticQuantization() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<KVCacheStaticQuantizationParams>& obj) {
        auto [original_precision, low_precision, sdpa_type, num_groups, stateful, scale, shift] = obj.param;
        std::ostringstream result;
        result << "original_precision=" << original_precision << "_"
               << "low_precision=" << low_precision << "_"
               << "sdpa_type=" << static_cast<int>(sdpa_type) << "_"
               << "num_groups=" << num_groups << "_"
               << "stateful=" << stateful << "_"
               << "scale=" << scale << "_"
               << "shift=" << shift;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> get_model(ov::element::Type original_precision,
                                         ov::element::Type low_precision,
                                         SDPAType sdpa_type,
                                         size_t num_groups,
                                         bool stateful,
                                         float scale,
                                         float shift);

    std::shared_ptr<ov::Model> get_model_ref(ov::element::Type original_precision,
                                             ov::element::Type low_precision,
                                             SDPAType sdpa_type,
                                             size_t num_groups,
                                             bool stateful,
                                             float scale,
                                             float shift);

    void SetUp() override {
        TransformationTestsF::SetUp();
        auto [original_precision, low_precision, sdpa_type, num_groups, stateful, scale, shift] = GetParam();
        model = get_model(original_precision, low_precision, sdpa_type, num_groups, stateful, scale, shift);
        model_ref = get_model_ref(original_precision, low_precision, sdpa_type, num_groups, stateful, scale, shift);

        using namespace ov::pass::low_precision;
        // Note: below is provided the minimal set of transformations required for kv cache static quantization.
        LayerTransformation::Params params;
        params.defaultPrecisions = {low_precision};
        manager.register_pass<MoveFakeConvertUpThroughKVCacheConcat>();
        auto graph_rewrite = manager.register_pass<ov::pass::GraphRewrite>();
        graph_rewrite->add_matcher<ov::pass::FakeConvertDecomposition>();
        graph_rewrite->add_matcher<ConcatTransformation>(params);
        // This transformation works only for stateful model
        graph_rewrite->add_matcher<KVCacheConcat>(model);
    }

private:
    // these params don't affect transformations behavior, so constant values can be used
    const ov::Dimension batch = 1, n_heads = 15, k_features = 64, v_features = 64;
    const std::vector<int64_t> qkv_order{0, 1, 2, 3};
};

TEST_P(KVCacheStaticQuantization, CompareWithRefImpl) {}

inline std::shared_ptr<ov::Node> create_sdpa(SDPAType sdpa_type,
                                             std::shared_ptr<ov::Node> q,
                                             std::shared_ptr<ov::Node> k,
                                             std::shared_ptr<ov::Node> v,
                                             ov::element::Type original_precision,
                                             ov::Dimension k_features,
                                             const std::vector<int64_t>& qkv_order) {
    std::shared_ptr<ov::Node> sdpa = nullptr;
    if (sdpa_type == SDPAType::WithMaskAndScale) {
        auto mask = make_attention_mask(q, k, original_precision, qkv_order);
        auto scale =
            v0::Constant::create(original_precision, ov::Shape{}, {1.0f / std::sqrt(k_features.get_max_length())});
        sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, v, mask, scale, true);
    } else if (sdpa_type == SDPAType::WithMask) {
        auto mask = make_attention_mask(q, k, original_precision, qkv_order);
        sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, v, mask, true);
    } else {
        OPENVINO_ASSERT(sdpa_type == SDPAType::Standard, "Unsupported SDPA type: ", static_cast<int>(sdpa_type));
        sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, v, true);
    }
    sdpa->set_friendly_name("sdpa");
    return sdpa;
}

std::shared_ptr<ov::Model> KVCacheStaticQuantization::get_model(ov::element::Type original_precision,
                                                                ov::element::Type low_precision,
                                                                SDPAType sdpa_type,
                                                                size_t num_groups,
                                                                bool stateful,
                                                                float scale,
                                                                float shift) {
    auto params = form_sdpa_params(batch,
                                   n_heads,
                                   k_features,
                                   v_features,
                                   original_precision,
                                   original_precision,
                                   qkv_order,
                                   num_groups);

    std::shared_ptr<ov::Node> concat_k_input = params[1];
    std::shared_ptr<ov::Node> concat_v_input = params[2];
    if (stateful) {
        auto in_beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
        in_beam_idx->set_friendly_name("beam_idx");
        params.push_back(in_beam_idx);

        concat_k_input = make_kv_rearrange(concat_k_input, in_beam_idx, qkv_order[0]);
        concat_v_input = make_kv_rearrange(concat_v_input, in_beam_idx, qkv_order[0]);
    }

    int64_t concat_axis = qkv_order[2];
    auto concat_k = std::make_shared<v0::Concat>(ov::OutputVector{concat_k_input, params[3]}, concat_axis);
    auto concat_v = std::make_shared<v0::Concat>(ov::OutputVector{concat_v_input, params[4]}, concat_axis);

    auto present_k = std::make_shared<v0::Result>(concat_k);
    present_k->set_friendly_name("present_k");

    auto present_v = std::make_shared<v0::Result>(concat_v);
    present_v->set_friendly_name("present_v");

    auto form_fake_convert = [&](std::shared_ptr<ov::Node> input) {
        auto fake_convert_scale = v0::Constant::create(original_precision, ov::Shape{}, {scale});
        auto fake_convert_shift = v0::Constant::create(original_precision, ov::Shape{}, {shift});
        return std::make_shared<v13::FakeConvert>(input, fake_convert_scale, fake_convert_shift, low_precision);
    };
    std::shared_ptr<ov::Node> q = params[0];
    std::shared_ptr<ov::Node> k = form_fake_convert(concat_k);
    std::shared_ptr<ov::Node> v = form_fake_convert(concat_v);

    if (num_groups > 1) {
        k = make_gqa(k, num_groups, n_heads, k_features, qkv_order);
        v = make_gqa(v, num_groups, n_heads, v_features, qkv_order);
    }

    auto sdpa = create_sdpa(sdpa_type, q, k, v, original_precision, k_features, qkv_order);
    auto sdpa_out = std::make_shared<v0::Result>(sdpa);
    sdpa_out->set_friendly_name("sdpa_out");

    ov::ResultVector results{sdpa_out, present_k, present_v};
    auto model = std::make_shared<ov::Model>(results, params, "LLM-KV-Cache-SDPA");

    if (stateful) {
        make_sdpa_model_stateful(model,
                                 params,
                                 present_k,
                                 present_v,
                                 original_precision,
                                 k_features,
                                 v_features,
                                 qkv_order);
        model->validate_nodes_and_infer_types();
    }
    return model;
}

std::shared_ptr<ov::Model> KVCacheStaticQuantization::get_model_ref(ov::element::Type original_precision,
                                                                    ov::element::Type low_precision,
                                                                    SDPAType sdpa_type,
                                                                    size_t num_groups,
                                                                    bool stateful,
                                                                    float scale,
                                                                    float shift) {
    auto params = form_sdpa_params(batch,
                                   n_heads,
                                   k_features,
                                   v_features,
                                   original_precision,
                                   stateful ? low_precision : original_precision,
                                   qkv_order,
                                   num_groups);
    auto fake_convert_downconvert = [&](const ov::Output<ov::Node>& input,
                                        float scale,
                                        float shift,
                                        bool skip_clamp = false) {
        auto downconvert = input;
        auto fake_convert_scale = v0::Constant::create(original_precision, ov::Shape{}, {scale});
        downconvert = std::make_shared<v1::Multiply>(input, fake_convert_scale);
        if (shift != 0.f) {
            auto fake_convert_shift = v0::Constant::create(original_precision, ov::Shape{}, {shift});
            downconvert = std::make_shared<v1::Subtract>(downconvert, fake_convert_shift);
        }
        if (!skip_clamp) {
            const auto [lower_bound, upper_bound] = [&]() {
                switch (low_precision) {
                case ov::element::f8e4m3:
                    return std::make_pair(static_cast<double>(std::numeric_limits<ov::float8_e4m3>::lowest()),
                                          static_cast<double>(std::numeric_limits<ov::float8_e4m3>::max()));
                case ov::element::f8e5m2:
                    return std::make_pair(static_cast<double>(std::numeric_limits<ov::float8_e5m2>::lowest()),
                                          static_cast<double>(std::numeric_limits<ov::float8_e5m2>::max()));
                default:
                    OPENVINO_THROW("Unsupported destination element type: ", low_precision);
                }
            }();
            downconvert = std::make_shared<v0::Clamp>(downconvert, lower_bound, upper_bound);
        }
        return std::make_shared<v0::Convert>(downconvert, low_precision);
    };

    auto fake_convert_upconvert =
        [original_precision](const ov::Output<ov::Node>& input, float scale, float shift) -> std::shared_ptr<ov::Node> {
        auto upconvert = input;
        upconvert = std::make_shared<v0::Convert>(input, original_precision);
        if (shift != 0.f) {
            auto fake_convert_shift = v0::Constant::create(original_precision, ov::Shape{}, {shift});
            upconvert = std::make_shared<v1::Subtract>(upconvert, fake_convert_shift);
        }
        auto fake_convert_scale = v0::Constant::create(original_precision, ov::Shape{}, {scale});
        return std::make_shared<v1::Multiply>(upconvert, fake_convert_scale);
    };

    // Create upconvert→downconvert ("reverse" fake convert: low precision on input-output, fp32 inside)
    auto create_reverse_fake_convert = [&](std::shared_ptr<ov::Node> rearranged_input) -> std::shared_ptr<ov::Node> {
        auto upconvert_result = fake_convert_upconvert(rearranged_input, 1.0f / scale, -shift);
        return fake_convert_downconvert(upconvert_result, scale, shift, true);
    };

    int64_t concat_axis = qkv_order[2];
    ov::OutputVector concat_k_inputs, concat_v_inputs;
    if (stateful) {
        auto in_beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
        in_beam_idx->set_friendly_name("beam_idx");
        params.push_back(in_beam_idx);

        auto rearranged_k = make_kv_rearrange(params[1], in_beam_idx, qkv_order[0]);
        auto rearranged_v = make_kv_rearrange(params[2], in_beam_idx, qkv_order[0]);

        auto concat_k_input = create_reverse_fake_convert(rearranged_k);
        auto concat_v_input = create_reverse_fake_convert(rearranged_v);
        // In case of stateful model, the whole KV cache subgraph has low precision after LPT,
        // so no fownconvert subgraph is needed.
        concat_k_inputs = {concat_k_input, fake_convert_downconvert(params[3], scale, shift)};
        concat_v_inputs = {concat_v_input, fake_convert_downconvert(params[4], scale, shift)};
    } else {
        concat_k_inputs = {fake_convert_downconvert(params[1], scale, shift),
                           fake_convert_downconvert(params[3], scale, shift)};
        concat_v_inputs = {fake_convert_downconvert(params[2], scale, shift),
                           fake_convert_downconvert(params[4], scale, shift)};
    }

    auto concat_k = std::make_shared<v0::Concat>(concat_k_inputs, concat_axis);
    auto concat_v = std::make_shared<v0::Concat>(concat_v_inputs, concat_axis);

    auto present_k = std::make_shared<v0::Result>(stateful ? create_reverse_fake_convert(concat_k)
                                                           // Note: upconvert part contains opposite scale and shift
                                                           : fake_convert_upconvert(concat_k, 1.f / scale, -shift));
    present_k->set_friendly_name("present_k");

    auto present_v = std::make_shared<v0::Result>(stateful ? create_reverse_fake_convert(concat_v)
                                                           // Note: upconvert part contains opposite scale and shift
                                                           : fake_convert_upconvert(concat_v, 1.f / scale, -shift));
    present_v->set_friendly_name("present_v");

    std::shared_ptr<ov::Node> q = params[0];
    std::shared_ptr<ov::Node> k = fake_convert_upconvert(concat_k, 1.f / scale, -shift);
    std::shared_ptr<ov::Node> v = fake_convert_upconvert(concat_v, 1.f / scale, -shift);

    if (num_groups > 1) {
        k = make_gqa(k, num_groups, n_heads, k_features, qkv_order);
        v = make_gqa(v, num_groups, n_heads, v_features, qkv_order);
    }

    auto sdpa = create_sdpa(sdpa_type, q, k, v, original_precision, k_features, qkv_order);
    auto sdpa_out = std::make_shared<v0::Result>(sdpa);
    sdpa_out->set_friendly_name("sdpa_out");

    ov::ResultVector results{sdpa_out, present_k, present_v};
    auto model = std::make_shared<ov::Model>(results, params, "LLM-KV-Cache-SDPA");

    if (stateful) {
        make_sdpa_model_stateful(model,
                                 params,
                                 present_k,
                                 present_v,
                                 original_precision,
                                 k_features,
                                 v_features,
                                 qkv_order);
        model->validate_nodes_and_infer_types();
    }
    return model;
}

namespace {
const ov::element::TypeVector low_precision_types = {
    ov::element::f8e4m3,
    ov::element::f8e5m2,
};

const std::vector<SDPAType> sdpa_types = {
    SDPAType::Standard,
    SDPAType::WithMask,
    SDPAType::WithMaskAndScale,
};

const std::vector<float> scales = {4.f};
const std::vector<float> shifts = {0.f, 12.f};

INSTANTIATE_TEST_SUITE_P(smoke_KVCacheStaticQuantization,
                         KVCacheStaticQuantization,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(low_precision_types),
                                            ::testing::ValuesIn(sdpa_types),
                                            ::testing::Values(1, 3),
                                            ::testing::Values(true, false),
                                            ::testing::ValuesIn(scales),
                                            ::testing::ValuesIn(shifts)),
                         KVCacheStaticQuantization::getTestCaseName);
}  // namespace