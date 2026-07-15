// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention_token_type.hpp"

#include <algorithm>
#include <random>

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/reference/utils/paged_cache_manager_helper.hpp"

using namespace ov::op;

#define UNUSED(expr) (void)(expr);

namespace ov {
namespace test {
namespace helpers {

static constexpr size_t MAX_CONTEXT_LEN = 1024;
static constexpr size_t BLOCK_SIZE = 16;  //< Default for standard PA on GPU.

static std::shared_ptr<ov::op::v0::Parameter> MakeParam(const PartialShape& pshape,
                                                        element::Type element_type,
                                                        const std::string& name) {
    auto param = std::make_shared<v0::Parameter>(element_type, pshape);
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

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

static std::shared_ptr<ov::Model> PrepareModel(ov::element::Type data_type,
                                               ov::Dimension::value_type head_size,
                                               ov::Dimension::value_type head_num,
                                               int32_t sliding_window_size) {
    auto q = MakeParam(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "q");
    auto k = MakeParam(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
    auto v = MakeParam(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");

    // GPU plugin expects 4-dim cache with concrete element type
    // key_cache: [num_blocks, num_kv_heads, head_size, block_size]
    // value_cache: [num_blocks, num_kv_heads, block_size, head_size]
    const int64_t block_size = helpers::BLOCK_SIZE;
    auto key_cache =
        MakeParam(PartialShape{ov::Dimension::dynamic(), head_num, head_size, block_size}, data_type, "key_cache.0");
    auto value_cache =
        MakeParam(PartialShape{ov::Dimension::dynamic(), head_num, block_size, head_size}, data_type, "value_cache.0");
    auto past_lens = MakeParam(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
    auto subsequence_begins = MakeParam(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
    auto block_indices = MakeParam(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
    auto block_indices_begins =
        MakeParam(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");

    float scale_value = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scale = std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
    auto sliding_window =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{sliding_window_size});
    auto alibi_slopes = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
    auto max_context_len =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{MAX_CONTEXT_LEN});
    auto score_aggregation_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto rotated_block_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto rotation_deltas = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto rotation_trig_lut = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
    auto xattention_threshold = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
    auto xattention_block_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto xattention_stride = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto sinks = std::static_pointer_cast<v0::Constant>(ov::test::utils::make_constant(data_type, Shape{0}));
    auto adaptive_rkv_start_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto adaptive_rkv_evictable_sizes =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto adaptive_rkv_diversity_block_set_indices =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto adaptive_rkv_diversity_block_set_indices_begins =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});

    auto token_type_ids = MakeParam(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "token_type_ids");
    auto qq_bias = std::make_shared<v0::Constant>(ov::element::u8, Shape{0}, std::vector<uint8_t>{0});
    auto qq_bias_begins = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});

    ParameterVector params = {q,
                              k,
                              v,
                              key_cache,
                              value_cache,
                              past_lens,
                              subsequence_begins,
                              block_indices,
                              block_indices_begins,
                              token_type_ids};

    OutputVector pa_inputs = {q,
                              k,
                              v,
                              key_cache,
                              value_cache,
                              past_lens,
                              subsequence_begins,
                              block_indices,
                              block_indices_begins,
                              scale,
                              sliding_window,
                              alibi_slopes,
                              max_context_len,
                              score_aggregation_window,
                              rotated_block_indices,
                              rotation_deltas,
                              rotation_trig_lut,
                              xattention_threshold,
                              xattention_block_size,
                              xattention_stride,
                              sinks,
                              adaptive_rkv_start_size,
                              adaptive_rkv_evictable_sizes,
                              adaptive_rkv_diversity_block_set_indices,
                              adaptive_rkv_diversity_block_set_indices_begins,
                              token_type_ids,
                              qq_bias,
                              qq_bias_begins};

    OPENVINO_ASSERT(pa_inputs.size() == 28);

    auto paged_attn = std::make_shared<op::PagedAttentionExtension>(pa_inputs);
    paged_attn->get_rt_info()["num_k_heads"] = head_num;
    paged_attn->get_rt_info()["k_head_size"] = head_size;
    paged_attn->get_rt_info()["num_v_heads"] = head_num;
    paged_attn->get_rt_info()["v_head_size"] = head_size;

    // WARNING! Cache manger is needed only for template plugin and is attached
    // via transformations. BUT func tests disable all transformations for template plugin,
    // so it is needed to attach cache manager manually here...
    auto shared_handle = std::make_shared<ov::reference::paged_attention_cache::CacheManagerHandle>();
    *shared_handle = ov::reference::paged_attention_cache::make_cache_handle(data_type);

    OPENVINO_ASSERT(paged_attn->get_input_element_type(3) == data_type,
                    "AttachCacheManagerToPagedAttention: incompatible cache data types");

    ov::reference::paged_attention_cache::set_cache_manager(paged_attn.get(), *shared_handle);
    // ---
    return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
}

}  // namespace helpers

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
    ASSERT_LE(seq_len, helpers::MAX_CONTEXT_LEN);
    configuration[ov::hint::inference_precision.name()] = ov::element::f32;
    configuration[ov::hint::kv_cache_precision.name()] = ov::element::f32;
    targetDevice = device;

    init_input_shapes({InputShape{PartialShape::dynamic(1), {{batch_size * seq_len}}}});

    function = helpers::PrepareModel(inType, head_size, head_num, sliding_window_size);
}

void PagedAttentionTokenTypeTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& [inType, head_size, head_num, sliding_window_size, batch_size, seq_length, device] = this->GetParam();

    UNUSED(sliding_window_size);
    UNUSED(device);

    OPENVINO_ASSERT(!targetInputStaticShapes.empty() && targetInputStaticShapes[0].size() == 1,
                    "Expected a single 1-D shape representing total token count");
    OPENVINO_ASSERT(targetInputStaticShapes[0][0] == batch_size * seq_length,
                    "Unexpected total token count for the configured batch and sequence length");
    const size_t seq_len = seq_length;
    const size_t total_tokens = targetInputStaticShapes[0][0];
    const size_t hidden_dim = static_cast<size_t>(head_size) * static_cast<size_t>(head_num);

    using ov::test::utils::InputGenerateData;

    ov::Tensor q_tensor =
        ov::test::utils::create_and_fill_tensor(inType, {total_tokens, hidden_dim}, InputGenerateData(-1, 2, 32, 1));
    ov::Tensor k_tensor =
        ov::test::utils::create_and_fill_tensor(inType, {total_tokens, hidden_dim}, InputGenerateData(-1, 2, 32, 2));
    ov::Tensor v_tensor =
        ov::test::utils::create_and_fill_tensor(inType, {total_tokens, hidden_dim}, InputGenerateData(-1, 2, 32, 3));

    ov::Tensor token_type_tensor(ov::element::i32, {total_tokens});
    auto* token_types = token_type_tensor.data<int32_t>();
    for (size_t batch = 0; batch < batch_size; ++batch) {
        ov::Tensor sequence_token_types = helpers::GenerateTokenTypeTensor(seq_len);
        std::copy(sequence_token_types.data<int32_t>(),
                  sequence_token_types.data<int32_t>() + seq_len,
                  token_types + batch * seq_len);
    }

    // Cache tensors with known shapes (matching PrepareModel layout).
    const size_t block_size = helpers::BLOCK_SIZE;
    const size_t max_blocks_per_sequence = (helpers::MAX_CONTEXT_LEN + block_size - 1) / block_size;
    const size_t block_nums = batch_size * max_blocks_per_sequence;
    ov::Tensor key_cache_tensor = ov::test::utils::create_and_fill_tensor(
        inType,
        {block_nums, static_cast<size_t>(head_num), static_cast<size_t>(head_size), block_size},
        InputGenerateData(-1, 2, 32, 5));
    ov::Tensor value_cache_tensor = ov::test::utils::create_and_fill_tensor(
        inType,
        {block_nums, static_cast<size_t>(head_num), block_size, static_cast<size_t>(head_size)},
        InputGenerateData(-1, 2, 32, 6));

    const int32_t blocks_per_sequence = static_cast<int32_t>((seq_len + block_size - 1) / block_size);
    const int32_t total_blocks = static_cast<int32_t>(batch_size) * blocks_per_sequence;

    ov::Tensor past_lens(ov::element::i32, {batch_size});
    ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
    ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
    ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

    subsequence_begins.data<int32_t>()[0] = 0;
    block_indices_begins.data<int32_t>()[0] = 0;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        past_lens.data<int32_t>()[batch] = 0;
        subsequence_begins.data<int32_t>()[batch + 1] = static_cast<int32_t>((batch + 1) * seq_len);
        block_indices_begins.data<int32_t>()[batch + 1] = static_cast<int32_t>(batch + 1) * blocks_per_sequence;
    }
    for (int32_t i = 0; i < total_blocks; i++) {
        block_indices.data<int32_t>()[i] = i;
    }

    for (const auto& param : function->get_parameters()) {
        const auto& name = param->get_friendly_name();
        if (name == "q")
            inputs.insert({param, q_tensor});
        else if (name == "k")
            inputs.insert({param, k_tensor});
        else if (name == "v")
            inputs.insert({param, v_tensor});
        else if (name == "key_cache.0")
            inputs.insert({param, key_cache_tensor});
        else if (name == "value_cache.0")
            inputs.insert({param, value_cache_tensor});
        else if (name == "past_lens")
            inputs.insert({param, past_lens});
        else if (name == "subsequence_begins")
            inputs.insert({param, subsequence_begins});
        else if (name == "block_indices")
            inputs.insert({param, block_indices});
        else if (name == "block_indices_begins")
            inputs.insert({param, block_indices_begins});
        else if (name == "token_type_ids")
            inputs.insert({param, token_type_tensor});
    }
}
}  // namespace test
}  // namespace ov

#undef UNUSED