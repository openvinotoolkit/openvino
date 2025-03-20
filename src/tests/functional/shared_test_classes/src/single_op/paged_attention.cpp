// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention.hpp"

#include "openvino/op/paged_attention.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
std::string PagedAttentionLayerTest::getTestCaseName(const testing::TestParamInfo<PagedAttentionParamsTuple>& obj) {
    std::vector<InputShape> input_shapes;
    PagedAttentionIntVectorsStruct i32_attr;
    PagedAttentionMiscInpStruct scalar_attr;
    std::optional<PagedAttentionRotationStruct> rotation_attr;
    ov::element::Type type;
    std::string targetDevice;
    std::tie(input_shapes, i32_attr, scalar_attr, rotation_attr, type, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : input_shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : input_shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "IT=" << type.get_type_name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void PagedAttentionLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    PagedAttentionIntVectorsStruct i32_attr;
    PagedAttentionMiscInpStruct misc_attrs;
    std::optional<PagedAttentionRotationStruct> rotation_attr;
    ov::element::Type type;
    std::tie(input_shapes, i32_attr, misc_attrs, rotation_attr, type, targetDevice) = this->GetParam();

    /*
    * [0]: query
    * shape: [batch_size_in_tokens, num_heads * head_size], type: f16
    * [1]: key
    * shape: [batch_size_in_tokens, num_kv_heads * head_size], type: f16
    * [2]: value 
    * shape: [batch_size_in_tokens, num_kv_heads * head_size], type: f16
    * [3]: key_cache
    * shape: [num_blocks, num_kv_heads, head_size, block_size], type: f16
    * [4]: value_cache
    * shape: [num_blocks, num_kv_heads, block_size, head_size], type: f16
    */
    init_input_shapes(input_shapes);
    ov::ParameterVector params;
    for (auto shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
    }

    /*
    * [5]: past_lens
    * shape: [batch_size_in_sequences], type: i32
    * [6]: subsequence_begins
    * shape: [batch_size_in_sequences + 1], type: i32
    * [7]: block_indices
    * Shape: [num_blocks], type: i32
    * [8]: block_indices_begins
    * Shape: [batch_size_in_sequences + 1], type: i32
    */
   auto past_lens = ov::op::v0::Constant::create(ov::element::i32, {i32_attr.past_lens.size()}, i32_attr.past_lens);
   auto subsequence_begins = ov::op::v0::Constant::create(ov::element::i32, {i32_attr.subsequence_begins.size()}, i32_attr.subsequence_begins);
   auto block_indices = ov::op::v0::Constant::create(ov::element::i32, {i32_attr.block_indices.size()}, i32_attr.block_indices);
   auto block_indices_begins = ov::op::v0::Constant::create(ov::element::i32, {i32_attr.block_indices_begins.size()}, i32_attr.block_indices_begins);

    /*
    * [9]: scale, optional
    * [10]: sliding_window, optional
    * [11]: alibi_slopes, optional
    * [12]: max_context_len
    * shape: [], type: i32
    */
    auto scale_const = ov::op::v0::Constant::create(ov::element::f32, {}, {misc_attrs.scale.value()});
    auto sliding_window_const = ov::op::v0::Constant::create(ov::element::i32, {}, {misc_attrs.sliding_window.value()});
    auto alibi_slopes_const = ov::op::v0::Constant::create(ov::element::f32, {misc_attrs.alibi_slopes.size()}, misc_attrs.alibi_slopes);
    auto max_context_len_const = ov::op::v0::Constant::create(ov::element::i32, {}, {misc_attrs.max_context_len});


    std::shared_ptr<ov::Node> paged_attn;
    if (!rotation_attr.has_value()) {
        paged_attn = std::make_shared<ov::op::PagedAttentionExtension>(
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            past_lens,
            subsequence_begins,
            block_indices,
            block_indices_begins,
            scale_const,
            sliding_window_const,
            alibi_slopes_const,
            max_context_len_const);
    } else {
    /*
    * [13]: rotated_block_indices​, optional​
    * shape: [num_rotated_blocks]​, type: i32
    * [14]: rotation_deltas​, optional​
    * shape: [num_rotated_blocks, BLOCK_SIZE]​ || [num_rotated_blocks, 1]​, type: i32
    * [15]: rotation_trig_lut​, optional​
    * shape: [max_num_batched_tokens / BLOCK_SIZE, head_size]​ || [max_num_batched_tokens, head_size], type: f16
    */
    auto rotated_block_indices = ov::op::v0::Constant::create(
        ov::element::i32,
        {rotation_attr.value().rotated_block_indices.size()},
        rotation_attr.value().rotated_block_indices);
    auto rotation_deltas = ov::op::v0::Constant::create(
        ov::element::i32,
        rotation_attr.value().rotation_deltas,
        std::vector<int>{0});
    auto rotation_trig_lut = ov::op::v0::Constant::create(
            ov::element::f32,
            rotation_attr.value().rotation_trig_lut,
            std::vector<float>{0});

    paged_attn = std::make_shared<ov::op::PagedAttentionExtension>(
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        past_lens,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        scale_const,
        sliding_window_const,
        alibi_slopes_const,
        max_context_len_const,
        rotated_block_indices,
        rotation_deltas,
        rotation_trig_lut);
    }
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(paged_attn->output(0))};
    function = std::make_shared<ov::Model>(results, params, "PagedAttentionInference");
}
} //  namespace test
} //  namespace ov
