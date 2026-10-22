// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "test_utils.h"

using namespace ov;

namespace {

constexpr size_t head_cnt = 4;
constexpr size_t head_size = 128;

// Qwen3-like attention head: per-head RMSNorm -> RotateHalf RoPE, with cos/sin computed from
// position_ids x inv_freq in the inference precision.
std::shared_ptr<Model> build_qwen3_rms_rope_model() {
    auto position_ids = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{1, -1});
    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto position_ids_unsqueeze = std::make_shared<op::v0::Unsqueeze>(position_ids, unsqueeze_axis);
    auto position_ids_f16 = std::make_shared<op::v0::Convert>(position_ids_unsqueeze, element::f16);

    std::vector<float> inv_freq(head_size / 2);
    for (size_t i = 0; i < inv_freq.size(); ++i) {
        inv_freq[i] = std::pow(1000000.f, -2.f * static_cast<float>(i) / static_cast<float>(head_size));
    }
    auto inv_freq_const = op::v0::Constant::create(element::f16, Shape{1, head_size / 2, 1}, inv_freq);
    auto freqs = std::make_shared<op::v0::MatMul>(inv_freq_const, position_ids_f16);
    auto freqs_transpose = std::make_shared<op::v1::Transpose>(freqs, op::v0::Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto emb = std::make_shared<op::v0::Concat>(OutputVector{freqs_transpose, freqs_transpose}, -1);
    auto cos = std::make_shared<op::v0::Unsqueeze>(std::make_shared<op::v0::Cos>(emb), unsqueeze_axis);
    auto sin = std::make_shared<op::v0::Unsqueeze>(std::make_shared<op::v0::Sin>(emb), unsqueeze_axis);

    auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, head_cnt, -1, head_size});
    auto gamma = op::v0::Constant::create(element::f16, Shape{head_size}, {1.f});
    auto rms = std::make_shared<op::internal::RMS>(data, gamma, 1e-6);

    op::internal::RoPE::Config config;
    config.head_cnt = head_cnt;
    config.head_size = head_size;
    config.rotary_ndims = head_size;
    auto rope = std::make_shared<op::internal::RoPE>(OutputVector{rms, cos, sin}, config);

    return std::make_shared<Model>(OutputVector{rope}, ParameterVector{position_ids, data});
}

}  // namespace

// FuseRMSRoPE adds the RMS gamma as a 4th RoPE input, while IncreasePositionIdsPrecision matches
// a 3-input RoPE only. If the fusion runs first, the position MatMul and Sin/Cos stay in f16,
// which breaks attention for long contexts (large position ids).
TEST(FuseRMSRoPEPipeline, PositionIdsPrecisionIsIncreased) {
    auto& engine = tests::get_test_engine();
    auto context = std::make_shared<intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = build_qwen3_rms_rope_model();

    auto config = tests::get_test_default_config(engine);
    config.set_user_property(ov::hint::inference_precision(element::f16));
    config.set_property(ov::intel_gpu::enable_rms_rope_fusion(true));
    config.finalize(context.get(), model.get());
    intel_gpu::TransformationsPipeline pipeline(config, context);
    pipeline.apply(model);

    size_t rope_cnt = 0, sin_cos_cnt = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto rope = as_type_ptr<op::internal::RoPE>(node)) {
            ++rope_cnt;
            EXPECT_EQ(rope->get_input_size(), 4) << "FuseRMSRoPE is not applied";
        } else if (is_type<op::v0::Sin>(node) || is_type<op::v0::Cos>(node)) {
            ++sin_cos_cnt;
            EXPECT_EQ(node->get_output_element_type(0), element::f32) << *node << ": IncreasePositionIdsPrecision is not applied";
        }
    }
    EXPECT_EQ(rope_cnt, 1);
    EXPECT_EQ(sin_cos_cnt, 2);
}
