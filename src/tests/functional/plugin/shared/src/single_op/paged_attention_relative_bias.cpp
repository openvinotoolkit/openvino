// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention_relative_bias.hpp"

#include "shared_test_classes/single_op/paged_attention_prefill.hpp"

namespace ov {
namespace test {
namespace {

paged_attention_prefill::ModelOptions model_options(const PagedAttnRelativeBiasParams& params) {
    const auto& [inType, head_size, head_num, distances, sliding_window_size, batch_size, seq_len, device] = params;
    paged_attention_prefill::ModelOptions options{ov::element::f32,
                                                  static_cast<int64_t>(head_size),
                                                  static_cast<int64_t>(head_num),
                                                  static_cast<int32_t>(sliding_window_size)};
    options.cpu_layout = device == ov::test::utils::DEVICE_CPU;
    options.batch_size = batch_size;
    options.relative_bias_distances = static_cast<int64_t>(distances);
    return options;
}

}  // namespace

std::string PagedAttentionRelativeBiasTest::getTestCaseName(
    const testing::TestParamInfo<PagedAttnRelativeBiasParams>& obj) {
    const auto& [inType, head_size, head_num, distances, sliding_window_size, batch_size, seq_len, device] = obj.param;
    std::ostringstream result;
    result << "Prc=" << inType << "_";
    result << "HS=" << head_size << "_";
    result << "HN=" << head_num << "_";
    result << "D=" << distances << "_";
    result << "SW=" << sliding_window_size << "_";
    result << "BS=" << batch_size << "_";
    result << "SQ=" << seq_len << "_";
    result << "Device=" << device;

    return result.str();
}

void PagedAttentionRelativeBiasTest::SetUp() {
    const auto& [inType, head_size, head_num, distances, sliding_window_size, batch_size, seq_len, device] = GetParam();
    ASSERT_LE(seq_len, paged_attention_prefill::MAX_CONTEXT_LEN);
    configuration[ov::hint::inference_precision.name()] = ov::element::Type(inType);
    if (inType == ov::element::f16) {
        abs_threshold = 2e-2;
        rel_threshold = 2e-2;
    } else if (inType == ov::element::bf16) {
        abs_threshold = 6e-2;
        rel_threshold = 6e-2;
    }
    configuration[ov::hint::kv_cache_precision.name()] = ov::element::f32;
    targetDevice = device;

    init_input_shapes({InputShape{PartialShape::dynamic(1), {{batch_size * seq_len}}}});

    function = paged_attention_prefill::make_model(model_options(GetParam()));
}

void PagedAttentionRelativeBiasTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto batch_size = std::get<5>(GetParam());
    const auto seq_len = std::get<6>(GetParam());
    OPENVINO_ASSERT(!targetInputStaticShapes.empty() && targetInputStaticShapes[0] == ov::Shape{batch_size * seq_len},
                    "Unexpected total token count for the configured batch and sequence length");

    const auto tensors = paged_attention_prefill::make_prefill_inputs(model_options(GetParam()), seq_len);
    inputs.clear();
    for (const auto& param : function->get_parameters())
        inputs.insert({param, tensors.at(param->get_friendly_name())});
}

}  // namespace test
}  // namespace ov
