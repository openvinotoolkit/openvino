// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention_token_type.hpp"

#include <algorithm>
#include <random>

#include "shared_test_classes/single_op/paged_attention_prefill.hpp"

namespace ov {
namespace test {
namespace {

static ov::Tensor GenerateTokenTypeTensor(size_t seq_len) {
    ov::Tensor tensor(ov::element::i32, {seq_len});
    auto* token_types = tensor.data<int32_t>();
    std::fill(token_types, token_types + seq_len, 0);

    if (seq_len == 0) {
        return tensor;
    }

    std::mt19937 generator(static_cast<std::mt19937::result_type>(5489U + seq_len));
    const size_t max_group_count = 4;
    const size_t image_group_count = std::uniform_int_distribution<size_t>(1, max_group_count)(generator);

    std::vector<size_t> group_sizes(image_group_count, 1);
    std::vector<size_t> gaps(image_group_count + 1, 0);
    for (size_t i = 1; i < image_group_count; ++i) {
        gaps[i] = 1;
    }

    const size_t min_sequence_len = 2 * image_group_count - 1;
    OPENVINO_ASSERT(seq_len >= min_sequence_len,
                    "Sequence length must fit at least one token per image group and separator gap between groups. ",
                    "seq_len=",
                    seq_len,
                    ", image_group_count=",
                    image_group_count,
                    ", min_sequence_len=",
                    min_sequence_len);

    size_t remaining_tokens = seq_len - min_sequence_len;
    std::uniform_int_distribution<size_t> bucket_distribution(0, group_sizes.size() + gaps.size() - 1);
    while (remaining_tokens-- > 0) {
        const size_t bucket = bucket_distribution(generator);
        if (bucket < group_sizes.size()) {
            ++group_sizes[bucket];
        } else {
            ++gaps[bucket - group_sizes.size()];
        }
    }

    size_t token_position = gaps.front();
    for (size_t group_index = 0; group_index < image_group_count; ++group_index) {
        std::fill(token_types + token_position, token_types + token_position + group_sizes[group_index], 1);
        token_position += group_sizes[group_index] + gaps[group_index + 1];
    }

    return tensor;
}

paged_attention_prefill::ModelOptions model_options(const PagedAttnTokenTypeParams& params) {
    const auto& [inType, head_size, head_num, sliding_window_size, batch_size, seq_len, device] = params;
    paged_attention_prefill::ModelOptions options{ov::element::Type(inType),
                                                  static_cast<int64_t>(head_size),
                                                  static_cast<int64_t>(head_num),
                                                  static_cast<int32_t>(sliding_window_size)};
    options.batch_size = batch_size;
    options.token_type_ids = true;
    return options;
}

}  // namespace

std::string PagedAttentionTokenTypeTest::getTestCaseName(const testing::TestParamInfo<PagedAttnTokenTypeParams>& obj) {
    const auto& [inType, head_size, head_num, sliding_window_size, batch_size, seq_len, device] = obj.param;
    std::ostringstream result;
    result << "Prc=" << inType << "_";
    result << "HS=" << head_size << "_";
    result << "HN=" << head_num << "_";
    result << "SW=" << sliding_window_size << "_";
    result << "BS=" << batch_size << "_";
    result << "SQ=" << seq_len << "_";
    result << "Device=" << device;

    return result.str();
}

void PagedAttentionTokenTypeTest::SetUp() {
    const auto& [inType, head_size, head_num, sliding_window_size, batch_size, seq_len, device] = GetParam();
    // CPU plugin can be supported - it uses different key and value cache layout
    // plus uses different block size, which was not yet implemented in this
    // test class.
    ASSERT_EQ(device, ov::test::utils::DEVICE_GPU);
    ASSERT_LE(seq_len, paged_attention_prefill::MAX_CONTEXT_LEN);
    configuration[ov::hint::inference_precision.name()] = ov::element::f32;
    configuration[ov::hint::kv_cache_precision.name()] = ov::element::f32;
    targetDevice = device;

    init_input_shapes({InputShape{PartialShape::dynamic(1), {{batch_size * seq_len}}}});

    function = paged_attention_prefill::make_model(model_options(GetParam()));
}

void PagedAttentionTokenTypeTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto batch_size = std::get<4>(GetParam());
    const auto seq_len = std::get<5>(GetParam());
    OPENVINO_ASSERT(!targetInputStaticShapes.empty() && targetInputStaticShapes[0] == ov::Shape{batch_size * seq_len},
                    "Unexpected total token count for the configured batch and sequence length");

    auto tensors = paged_attention_prefill::make_prefill_inputs(model_options(GetParam()), seq_len);
    ov::Tensor token_types(ov::element::i32, {batch_size * seq_len});
    for (size_t batch = 0; batch < batch_size; ++batch) {
        const auto sequence = GenerateTokenTypeTensor(seq_len);
        std::copy_n(sequence.data<int32_t>(), seq_len, token_types.data<int32_t>() + batch * seq_len);
    }
    tensors["token_type_ids"] = token_types;

    inputs.clear();
    for (const auto& param : function->get_parameters())
        inputs.insert({param, tensors.at(param->get_friendly_name())});
}

}  // namespace test
}  // namespace ov
