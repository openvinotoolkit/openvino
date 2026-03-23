// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_attention_token_type.hpp"

#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

using namespace ov::op;

namespace ov {
namespace test {
namespace helpers {
static std::vector<float> GetOutputAsFloatVec(const ov::Tensor& tensor) {
    std::vector<float> result(tensor.get_size());
    if (tensor.get_element_type() == ov::element::f32) {
        auto* p = tensor.data<float>();
        std::copy(p, p + tensor.get_size(), result.begin());
    } else if (tensor.get_element_type() == ov::element::f16) {
        auto* p = tensor.data<ov::float16>();
        for (size_t i = 0; i < tensor.get_size(); i++) {
            result[i] = static_cast<float>(p[i]);
        }
    }
    return result;
}

static std::shared_ptr<ov::op::v0::Parameter> MakeParam(const PartialShape& pshape,
                                                        element::Type element_type,
                                                        const std::string& name) {
    auto param = std::make_shared<v0::Parameter>(element_type, pshape);
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

static std::shared_ptr<ov::Model> PrepareModel(ov::element::Type data_type,
                                               ov::Dimension::value_type head_size,
                                               ov::Dimension::value_type head_num) {
    auto q = MakeParam(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "q");
    auto k = MakeParam(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
    auto v = MakeParam(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
    // GPU plugin expects 4-dim cache with concrete element type
    // key_cache: [num_blocks, num_kv_heads, head_size, block_size]
    // value_cache: [num_blocks, num_kv_heads, block_size, head_size]
    const int64_t block_size = 16;
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
    auto sliding_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto alibi_slopes = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
    auto max_context_len = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{1024});
    auto score_aggregation_window = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto rotated_block_indices = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto rotation_deltas = std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto rotation_trig_lut = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
    auto xattention_threshold = std::make_shared<v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{0});
    auto xattention_block_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{64});
    auto xattention_stride = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{8});
    auto sinks = std::static_pointer_cast<v0::Constant>(ov::test::utils::make_constant(data_type, Shape{0}));
    auto adaptive_rkv_start_size = std::make_shared<v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
    auto adaptive_rkv_evictable_sizes =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto adaptive_rkv_diversity_block_set_indices =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});
    auto adaptive_rkv_diversity_block_set_indices_begins =
        std::make_shared<v0::Constant>(ov::element::i32, Shape{0}, std::vector<int32_t>{0});

    auto token_type_ids = MakeParam(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "token_type_ids");

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
                              token_type_ids};

    OPENVINO_ASSERT(pa_inputs.size() == 26);

    auto paged_attn = std::make_shared<op::PagedAttentionExtension>(pa_inputs);
    paged_attn->get_rt_info()["num_k_heads"] = head_num;
    paged_attn->get_rt_info()["k_head_size"] = head_size;
    paged_attn->get_rt_info()["num_v_heads"] = head_num;
    paged_attn->get_rt_info()["v_head_size"] = head_size;

    return std::make_shared<ov::Model>(OutputVector{paged_attn}, params);
}

}  // namespace helpers

std::string PagedAttentionTokenTypeTest::getTestCaseName(const testing::TestParamInfo<PagedAttnTokenTypeParams>& obj) {
    const auto& [inType, head_size, head_num, pattern, device] = obj.param;
    std::ostringstream result;
    result << "Prc=" << inType << "_";
    result << "HS=" << head_size << "_";
    result << "HN=" << head_num << "_";
    result << "Pattern=" << pattern.name << "_";
    result << "Device=" << device;
    return result.str();
}

void PagedAttentionTokenTypeTest::SetUp() {
    const auto& [inType, head_size, head_num, pattern, device] = GetParam();
    configuration[ov::hint::inference_precision.name()] = ov::element::f32;
    configuration[ov::hint::kv_cache_precision.name()] = ov::element::f32;
    targetDevice = device;
    function = helpers::PrepareModel(inType, head_size, head_num);
    compile_model();
}

void PagedAttentionTokenTypeTest::RunAndValidate() {
    const auto& [inType, head_size, head_num, data, device] = this->GetParam();

    const size_t seq_len = data.tokenTypes.size();
    const size_t hidden_dim = head_size * head_num;

    ASSERT_EQ(data.qData.size(), seq_len * hidden_dim);
    ASSERT_EQ(data.kData.size(), seq_len * hidden_dim);
    ASSERT_EQ(data.vData.size(), seq_len * hidden_dim);

    ov::Tensor token_type_tensor(ov::element::i32, {seq_len});
    std::memcpy(token_type_tensor.data<int32_t>(), data.tokenTypes.data(), seq_len * sizeof(int32_t));

    ASSERT_TRUE(inType == ov::element::f32);
    ov::Tensor q_tensor(inType, {seq_len, hidden_dim});
    std::memcpy(q_tensor.data<float>(), data.qData.data(), seq_len * hidden_dim * sizeof(float));
    ov::Tensor k_tensor(inType, {seq_len, hidden_dim});
    std::memcpy(k_tensor.data<float>(), data.kData.data(), seq_len * hidden_dim * sizeof(float));
    ov::Tensor v_tensor(inType, {seq_len, hidden_dim});
    std::memcpy(v_tensor.data<float>(), data.vData.data(), seq_len * hidden_dim * sizeof(float));

    auto infer_request = compiledModel.create_infer_request();

    // Create cache tensors with known shapes
    const size_t block_size = 16;
    const size_t block_nums = 1024 / block_size;
    ov::Tensor key_cache_tensor(inType, {block_nums, head_num, head_size, block_size});
    ov::Tensor value_cache_tensor(inType, {block_nums, head_num, block_size, head_size});

    auto params = function->get_parameters();

    // Prefill: past_lens=0, single sequence
    size_t batch_size = 1;
    int32_t total_blocks = static_cast<int32_t>((seq_len + block_size - 1) / block_size);

    ov::Tensor past_lens(ov::element::i32, {batch_size});
    ov::Tensor subsequence_begins(ov::element::i32, {batch_size + 1});
    ov::Tensor block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
    ov::Tensor block_indices_begins(ov::element::i32, {batch_size + 1});

    past_lens.data<int32_t>()[0] = 0;
    subsequence_begins.data<int32_t>()[0] = 0;
    subsequence_begins.data<int32_t>()[1] = static_cast<int32_t>(seq_len);
    block_indices_begins.data<int32_t>()[0] = 0;
    block_indices_begins.data<int32_t>()[1] = total_blocks;
    for (int32_t i = 0; i < total_blocks; i++) {
        block_indices.data<int32_t>()[i] = i;
    }

    for (auto& param : params) {
        auto name = param->get_friendly_name();
        if (name == "q")
            infer_request.set_tensor(param, q_tensor);
        else if (name == "k")
            infer_request.set_tensor(param, k_tensor);
        else if (name == "v")
            infer_request.set_tensor(param, v_tensor);
        else if (name == "key_cache.0")
            infer_request.set_tensor(param, key_cache_tensor);
        else if (name == "value_cache.0")
            infer_request.set_tensor(param, value_cache_tensor);
        else if (name == "past_lens")
            infer_request.set_tensor(param, past_lens);
        else if (name == "subsequence_begins")
            infer_request.set_tensor(param, subsequence_begins);
        else if (name == "block_indices")
            infer_request.set_tensor(param, block_indices);
        else if (name == "block_indices_begins")
            infer_request.set_tensor(param, block_indices_begins);
        else if (name == "token_type_ids")
            infer_request.set_tensor(param, token_type_tensor);
    }

    infer_request.infer();

    auto output = infer_request.get_output_tensor(0);
    ov::Tensor output_copy{output.get_element_type(), output.get_shape()};
    output.copy_to(output_copy);

    const std::vector<float> outputVec = helpers::GetOutputAsFloatVec(output_copy);

    const float tolerance = (inType == ElementType::f16) ? 1e-2f : 1e-5f;

    ASSERT_EQ(outputVec.size(), data.expectedOutput.size());

    for (size_t i = 0; i < outputVec.size(); i++) {
        float diff = std::abs(outputVec[i] - data.expectedOutput[i]);
        EXPECT_LE(diff, tolerance) << "Output differs from expected at index " << i << ": got " << outputVec[i]
                                   << ", expected " << data.expectedOutput[i];
    }
}

}  // namespace test
}  // namespace ov