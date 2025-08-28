// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest-param-test.h>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;
using namespace std;

namespace ov {
namespace test {
using InputShapes = std::vector<InputShape>;
using PagedAttnTestParams = std::tuple<ElementType, InputShapes, uint32_t>;

class PagedAttnScoreTest : public testing::WithParamInterface<PagedAttnTestParams>,
                           virtual public ov::test::SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedAttnTestParams>& obj) {
        const auto& [inType, inputShapes, score_aggregation_window] = obj.param;
        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "Prc=" << inType;
        result << "_ScoreAggregationWindow=" << score_aggregation_window;

        return result.str();
    }
    static std::shared_ptr<ov::op::v0::Parameter> make_param(const PartialShape& pshape,
                                                             element::Type element_type,
                                                             const std::string& name) {
        auto param = std::make_shared<v0::Parameter>(element_type, pshape);
        param->set_friendly_name(name);
        param->get_output_tensor(0).set_names({name});
        return param;
    }

    std::shared_ptr<ov::Model> get_model(ov::element::Type data_type,
                                         ov::Dimension::value_type head_size = 64,
                                         ov::Dimension::value_type head_num = 8,
                                         uint32_t score_aggregation_window = 0) {
        // q [batch_in_tokens, head_num * head_size]
        // k [batch_in_tokens, head_num * head_size]
        // v [batch_in_tokens, head_num * head_size]
        auto q = make_param(PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, data_type, "q");
        auto k = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "k");
        auto v = make_param(PartialShape{ov::Dimension::dynamic(), head_num * head_size}, data_type, "v");
        auto key_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                    ov::element::dynamic,
                                    "key_cache.0");
        auto value_cache = make_param(PartialShape{ov::Dimension::dynamic(), 32, ov::Dimension::dynamic()},
                                      ov::element::dynamic,
                                      "value_cache.0");
        auto past_lens = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "past_lens");
        auto subsequence_begins =
            make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "subsequence_begins");
        auto block_indices = make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices");
        auto block_indices_begins =
            make_param(PartialShape{ov::Dimension::dynamic()}, ov::element::i32, "block_indices_begins");
        float scale_value = 1.0 / std::sqrt(head_size);
        auto scale =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale_value});
        auto silding_windows =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<int32_t>{0});
        auto alibi_slopes = std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto max_context_len =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<float>{128});
        auto score_aggregation_window_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<uint32_t>{score_aggregation_window});
        auto rotated_block_indices =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<uint32_t>{});
        auto rotation_deltas =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{0}, std::vector<uint32_t>{});
        auto rotation_trig_lut =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto xattention_threshold =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, Shape{0}, std::vector<float>{});
        auto xattention_block_size =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<uint32_t>{0});
        auto xattention_stride =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, std::vector<uint32_t>{0});
        ParameterVector params =
            {q, k, v, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins};
        auto paged_attn = std::make_shared<op::PagedAttentionExtension>(OutputVector{q,
                                                                                     k,
                                                                                     v,
                                                                                     key_cache,
                                                                                     value_cache,
                                                                                     past_lens,
                                                                                     subsequence_begins,
                                                                                     block_indices,
                                                                                     block_indices_begins,
                                                                                     scale,
                                                                                     silding_windows,
                                                                                     alibi_slopes,
                                                                                     max_context_len,
                                                                                     score_aggregation_window_node,
                                                                                     rotated_block_indices,
                                                                                     rotation_deltas,
                                                                                     rotation_trig_lut,
                                                                                     xattention_threshold,
                                                                                     xattention_block_size,
                                                                                     xattention_stride});
        paged_attn->get_rt_info()["num_k_heads"] = head_num;
        paged_attn->get_rt_info()["k_head_size"] = head_size;
        paged_attn->get_rt_info()["num_v_heads"] = head_num;
        paged_attn->get_rt_info()["v_head_size"] = head_size;
        OutputVector outputs{paged_attn};
        if (score_aggregation_window) {
            outputs.push_back(paged_attn->output(1));
        }
        return std::make_shared<ov::Model>(outputs, params);
    }

    void SetUp() override {
        const auto& [inType, inputShapes, score_aggregation_window] = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 0.01f;
        abs_threshold = 0.01f;
        configuration[ov::hint::inference_precision.name()] = ov::element::f32;
        if (inType == ElementType::bf16) {
            configuration[ov::hint::inference_precision.name()] = ov::element::bf16;
        }
        if (inType != ElementType::f32) {
            rel_threshold = 0.02f;
            abs_threshold = 0.02f;
        }
        init_input_shapes(inputShapes);
        ov::ParameterVector inputParams;

        function = get_model(inType, 64, 8, score_aggregation_window);
        targetDevice = ov::test::utils::DEVICE_CPU;

        functionRefs = get_ref_model(inType, 64, 8, score_aggregation_window);
    }

    virtual void generate(int idx, const bool isPagedAttn, const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        auto create_input = [this](std::shared_ptr<ov::op::v0::Parameter> param, ov::Shape shape, float val = 0) {
            if (param->get_element_type() == ov::element::i32) {
                ov::Tensor t{ov::element::i32, shape};
                auto size = shape[0];
                auto* p = static_cast<int*>(t.data());
                auto start = static_cast<int>(val);
                for (size_t i = 0; i < size; i++) {
                    p[i] = (start + i) % size;
                }
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                utils::fill_data_random(static_cast<float*>(t.data()), t.get_size(), 2, -1, 10);
                inputs.insert({param, t});
            } else if (param->get_element_type() == ov::element::f16) {
                ov::Tensor t{ov::element::f16, shape};
                utils::fill_data_random(static_cast<ov::float16*>(t.data()), t.get_size(), 2, -1, 10);
                inputs.insert({param, t});
            } else {
                ASSERT_TRUE(param->get_element_type() == ov::element::bf16);
                ov::Tensor t{ov::element::bf16, shape};
                utils::fill_data_random(static_cast<ov::bfloat16*>(t.data()), t.get_size(), 2, -1, 10);
                inputs.insert({param, t});
            }
        };

        if (isPagedAttn) {
            auto qkv_shape = targetInputStaticShapes[0];
            // L, B, H, S -> L * B, H * S
            create_input(function->get_parameters()[0],
                         {qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]});
            create_input(function->get_parameters()[1],
                         {qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]});
            create_input(function->get_parameters()[2],
                         {qkv_shape[0] * qkv_shape[1], qkv_shape[2] * qkv_shape[3]});
            size_t batch_size_in_sequences = 1;
            // The test here simulates pagedAttn calcuation with 1 subsequence
            // idx = 0 means 1st token calculation, idx > 0 means 2nd token calculation
            int32_t total_blocks = intel_cpu::div_up(qkv_shape[0] + past_len_count, 32);
            ov::Tensor past_lens(ov::element::i32, {batch_size_in_sequences}),
                subsequence_begins(ov::element::i32, {batch_size_in_sequences + 1}),
                block_indices_begins(ov::element::i32, {batch_size_in_sequences + 1}),
                block_indices(ov::element::i32, {static_cast<size_t>(total_blocks)});
            int32_t *past_lens_data = reinterpret_cast<int32_t*>(past_lens.data()),
                    *subsequence_begins_data = reinterpret_cast<int32_t*>(subsequence_begins.data()),
                    *block_indices_begins_data = reinterpret_cast<int32_t*>(block_indices_begins.data()),
                    *block_indices_data = reinterpret_cast<int32_t*>(block_indices.data());
            inputs.insert({function->get_parameters()[3], key_cache});
            inputs.insert({function->get_parameters()[4], value_cache});
            if (idx == 0) {
                past_lens_data[0] = 0;
                subsequence_begins_data[0] = 0;
                subsequence_begins_data[1] = targetInputStaticShapes[0][0];
                block_indices_begins_data[0] = 0;
                block_indices_begins_data[1] = 1;
            } else {
                past_lens_data[0] = past_len_count;
                subsequence_begins_data[0] = 0;
                subsequence_begins_data[1] = targetInputStaticShapes[0][0];
                block_indices_begins_data[0] = 0;
                block_indices_begins_data[1] = total_blocks;
            }
            for (int32_t i = 0; i < total_blocks; i++) {
                block_indices_data[i] = i;
            }
            inputs.insert({function->get_parameters()[5], past_lens});
            inputs.insert({function->get_parameters()[6], subsequence_begins});
            inputs.insert({function->get_parameters()[7], block_indices});
            inputs.insert({function->get_parameters()[8], block_indices_begins});
            past_len_count += static_cast<int32_t>(qkv_shape[0]);
        } else {
            // q, k, v, pastkv
            create_input(function->get_parameters()[0], targetInputStaticShapes[0]);
            create_input(function->get_parameters()[1], targetInputStaticShapes[0]);
            create_input(function->get_parameters()[2], targetInputStaticShapes[0]);
            create_input(function->get_parameters()[3], targetInputStaticShapes[1]);
            create_input(function->get_parameters()[4], ov::Shape{targetInputStaticShapes[0][1]});
        }
    }
    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }
    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }

    std::shared_ptr<ov::Model> get_ref_model(ov::element::Type data_type,
                                             ov::Dimension::value_type head_size = 64,
                                             ov::Dimension::value_type head_num = 8,
                                             uint32_t score_aggregation_window = 0) {
        // q, k, v use L,B,H,S layout
        ov::PartialShape q_shape, kv_shape, past_shape;
        ov::ParameterVector inputParams;
        past_shape = {-1, 1, head_num, head_size};
        q_shape = {-1, 1, static_cast<int64_t>(head_num), head_size};
        kv_shape = {-1, 1, head_num, head_size};
        auto q = make_param(q_shape, data_type, "q");
        auto k = make_param(kv_shape, data_type, "k");
        auto v = make_param(kv_shape, data_type, "v");
        auto past_kv = make_param(past_shape, data_type, "past_kv");
        auto beam_idx = make_param(ov::PartialShape{-1}, ov::element::i32, "beam_idx");
        inputParams.push_back(q);
        inputParams.push_back(k);
        inputParams.push_back(v);
        inputParams.push_back(past_kv);
        inputParams.push_back(beam_idx);
        auto var_k =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{past_shape, data_type, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_v);
        pastv->set_friendly_name("pastv_r");
        std::vector<size_t> transposeOrder{1, 2, 0, 3};
        auto preOrder = op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);
        auto q_in = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);
        auto concat_axis = transposeOrder[2];
        auto gatherK =
            std::make_shared<ov::op::v8::Gather>(pastk,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto gatherV =
            std::make_shared<ov::op::v8::Gather>(pastv,
                                                 beam_idx,
                                                 op::v0::Constant::create(ov::element::i32, {1}, {transposeOrder[0]}));
        auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherK, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{gatherV, inputParams[2]}, concat_axis);
        auto pastk_assign = std::make_shared<ov::op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<ov::op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");
        auto zero_1d_const = op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto zero_scalar_const = op::v0::Constant::create(ov::element::i32, {}, {0});
        auto one_1d_const = op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto one_scalar_const = op::v0::Constant::create(ov::element::i32, {}, {1});
        auto neg_2_1d_const = op::v0::Constant::create(ov::element::i32, {1}, {-2});
        auto neg_2_scalar_const = op::v0::Constant::create(ov::element::i32, {}, {-2});
        // mha structure
        auto ConvertLike_484 =
            std::make_shared<ov::op::v1::ConvertLike>(op::v0::Constant::create(ov::element::i32, {}, {1}), q_in);
        auto ConvertLike_491 =
            std::make_shared<ov::op::v1::ConvertLike>(op::v0::Constant::create(ov::element::i32, {}, {64}), q_in);
        auto Sqrt_492 = std::make_shared<ov::op::v0::Sqrt>(ConvertLike_491);
        auto Divide_493 = std::make_shared<ov::op::v1::Divide>(ConvertLike_484, Sqrt_492);
        auto Multiply_494 = std::make_shared<ov::op::v1::Multiply>(q_in, Divide_493);
        auto Transpose_442 = std::make_shared<ov::op::v1::Transpose>(concatK, preOrder);
        auto ShapeOf_479 = std::make_shared<ov::op::v3::ShapeOf>(Transpose_442, ov::element::i32);
        auto ShapeOf_495 = std::make_shared<ov::op::v3::ShapeOf>(ShapeOf_479, ov::element::i32);
        auto Add_497 =
            std::make_shared<ov::op::v1::Add>(ShapeOf_495, op::v0::Constant::create(ov::element::i32, {1}, {-2}));
        auto Squeeze_500 = std::make_shared<ov::op::v0::Squeeze>(Add_497, zero_1d_const);
        auto Range_501 =
            std::make_shared<ov::op::v4::Range>(zero_scalar_const, Squeeze_500, one_scalar_const, ov::element::i32);
        auto Add_496 =
            std::make_shared<ov::op::v1::Add>(ShapeOf_495, op::v0::Constant::create(ov::element::i32, {1}, {-1}));
        auto Concat_505 = std::make_shared<ov::op::v0::Concat>(OutputVector{Range_501, Add_496, Add_497}, 0);
        auto Transpose_506 = std::make_shared<ov::op::v1::Transpose>(Transpose_442, Concat_505);
        auto MatMul_510 = std::make_shared<ov::op::v0::MatMul>(Multiply_494, Transpose_506, false, false);
        // k_len
        auto Gather_517 = std::make_shared<ov::op::v8::Gather>(ShapeOf_479, neg_2_scalar_const, zero_scalar_const, 0);
        auto Range_531 =
            std::make_shared<ov::op::v4::Range>(zero_scalar_const, Gather_517, one_scalar_const, ov::element::i32);
        auto Unsqueeze_532 = std::make_shared<ov::op::v0::Unsqueeze>(Range_531, zero_scalar_const);
        auto ShapeOf_478 = std::make_shared<ov::op::v3::ShapeOf>(q_in, ov::element::i32);
        // q_len
        auto q_len = std::make_shared<ov::op::v8::Gather>(ShapeOf_478, neg_2_scalar_const, zero_scalar_const, 0);
        // past_len
        auto shape_past_len = std::make_shared<ov::op::v3::ShapeOf>(past_kv, ov::element::i32);
        auto past_len = std::make_shared<ov::op::v8::Gather>(shape_past_len, zero_scalar_const, zero_scalar_const, 0);
        auto total_len = std::make_shared<ov::op::v1::Add>(q_len, past_len);
        auto Range_534 =
            std::make_shared<ov::op::v4::Range>(zero_scalar_const, q_len, one_scalar_const, ov::element::i32);
        auto add_past_len = std::make_shared<ov::op::v1::Add>(Range_534, past_len);
        auto Unsqueeze_597 = std::make_shared<ov::op::v0::Unsqueeze>(add_past_len, one_scalar_const);
        auto GreaterEqual_598 = std::make_shared<ov::op::v1::Greater>(Unsqueeze_532, Unsqueeze_597);
        auto Unsqueeze_521 = std::make_shared<ov::op::v0::Unsqueeze>(q_len, zero_scalar_const);
        auto Unsqueeze_520 = std::make_shared<ov::op::v0::Unsqueeze>(Gather_517, zero_scalar_const);
        auto Concat_522 = std::make_shared<ov::op::v0::Concat>(OutputVector{Unsqueeze_521, Unsqueeze_520}, 0);
        auto Constant_523 = op::v0::Constant::create(element::u8, ov::Shape({}), {0});
        float negative_inf = -INFINITY;
        auto ConvertLike_511 =
            std::make_shared<ov::op::v1::ConvertLike>(v0::Constant::create(ov::element::f32, Shape{}, {negative_inf}),
                                                      MatMul_510);
        // mask
        auto Broadcast_524 = std::make_shared<ov::op::v1::Broadcast>(ConvertLike_511,
                                                                     Concat_522,
                                                                     Constant_523,
                                                                     AutoBroadcastSpec(AutoBroadcastType::NUMPY));
        auto ConvertLike_485 = std::make_shared<ov::op::v1::ConvertLike>(zero_scalar_const, q_in);
        auto Select_599 = std::make_shared<ov::op::v1::Select>(GreaterEqual_598, Broadcast_524, ConvertLike_485);
        auto Add_600 = std::make_shared<ov::op::v1::Add>(MatMul_510, Select_599);

        // softmax
        auto Softmax_9787 = std::make_shared<ov::op::v8::Softmax>(Add_600, -1);
        auto Transpose_443 = std::make_shared<ov::op::v1::Transpose>(concatV, preOrder);
        auto mha = std::make_shared<ov::op::v0::MatMul>(Softmax_9787, Transpose_443, false, false);
        auto Transpose_9634 = std::make_shared<ov::op::v1::Transpose>(
            mha,
            op::v0::Constant::create(ov::element::i32, {4}, std::vector<int32_t>{2, 0, 1, 3}));
        auto Reshape_9636 = std::make_shared<ov::op::v1::Reshape>(
            Transpose_9634,
            op::v0::Constant::create(ov::element::i32, {2}, std::vector<int32_t>{0, 512}),
            true);
        SinkVector sinks{pastk_assign, pastv_assign};
        ov::OutputVector results{Reshape_9636};
        if (score_aggregation_window) {
            auto cvt = std::make_shared<ov::op::v0::Convert>(Softmax_9787, element::f32);
            results.push_back(cvt);
        }
        auto model = std::make_shared<Model>(results, sinks, inputParams, "model");
        return model;
    }

    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model, uint32_t score_aggregation_window) {
        configuration[ov::hint::kv_cache_precision.name()] = ov::element::f16;
        function = model;
        prepare();
        for (const auto& input : compiledModel.inputs()) {
            for (auto& name : input.get_names()) {
                auto cache_precision = input.get_element_type();
                const size_t block_nums = 4;
                ov::PartialShape pshape;
                if (name.find("key_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    key_cache = ov::Tensor(cache_precision, pshape.get_shape());
                    break;
                } else if (name.find("value_cache.") == 0) {
                    pshape = input.get_partial_shape();
                    pshape[0] = block_nums;
                    value_cache = ov::Tensor(cache_precision, pshape.get_shape());
                    break;
                }
            }
        }
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, true, shapes);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            auto logits = inferRequest.get_output_tensor(0);
            ov::Tensor copy{logits.get_element_type(), logits.get_shape()};
            logits.copy_to(copy);
            outputs.push_back(copy);

            if (score_aggregation_window) {
                auto score = inferRequest.get_output_tensor(1);
                ov::Tensor score_copy{score.get_element_type(), score.get_shape()};
                score.copy_to(score_copy);
                outputs.push_back(score_copy);
            }
        }
        return outputs;
    }

    ov::Tensor get_score_ref(ov::Tensor softmax_output, uint32_t score_aggregation_window) {
        auto softmax_shape = softmax_output.get_shape();
        OPENVINO_ASSERT(softmax_shape.size() == 4, "shape:", softmax_shape);
        auto softmax_p = softmax_output.data<float>();
        auto B = softmax_shape[0], H = softmax_shape[1],
            L_q = softmax_shape[2], L_kv = softmax_shape[3];
        ov::Shape shape = {L_kv};
        ov::Tensor score{ov::element::f32, shape};
        std::memset(score.data(), 0, score.get_byte_size());
        auto score_p = score.data<float>();
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                size_t l = L_q > score_aggregation_window ? L_q - score_aggregation_window : 0;
                for (; l < L_q; l++) {
                    for (size_t x = 0; x < L_kv; x++) {
                        score_p[b * B + x] += softmax_p[b * H * L_q * L_kv + h * L_q * L_kv + l * L_kv + x];
                    }
                }
            }
        }
        return score;
    }

    std::vector<ov::Tensor> run_ref_test(std::shared_ptr<ov::Model> model, uint32_t score_aggregation_window) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, false, shapes);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            auto logits = inferRequest.get_output_tensor(0);
            ov::Tensor copy{logits.get_element_type(), logits.get_shape()};
            logits.copy_to(copy);
            outputs.push_back(copy);

            if (score_aggregation_window)
                outputs.push_back(get_score_ref(inferRequest.get_output_tensor(1), score_aggregation_window));
        }
        reset();
        return outputs;
    }
    std::vector<size_t> transposeOrder;
    size_t keyGroupSize = 0;
    bool quantKeyByChannel = false;
    bool hasShapeOf;
    ov::Tensor key_cache;
    ov::Tensor value_cache;
    int32_t past_len_count = 0;
};

TEST_P(PagedAttnScoreTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    const auto& [inType, inputShapes, score_aggregation_window] = this->GetParam();
    if (inType == ElementType::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP();
    auto actualOutputs = run_test(function, score_aggregation_window);
    auto expectedOutputs = run_ref_test(functionRefs, score_aggregation_window);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {

const std::vector<InputShapes> inputShapes = {  // greedy search
    {
        // L1, B, H, S, across blocks(32 + 2)
        {{-1, 1, 8, 64}, {{32 + 2, 1, 8, 64}, {1, 1, 8, 64}}},
        // B, L0, H, S
        {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {32 + 2, 1, 8, 64}}},
    }};

INSTANTIATE_TEST_SUITE_P(smoke_PagedAttnScoreTest,
                         PagedAttnScoreTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::f16, ElementType::bf16),
                                            ::testing::ValuesIn(inputShapes),
                                            // 0: disable score function, 1: old function, 7: across blocks
                                            ::testing::Values(0, 1, 7)),
                         PagedAttnScoreTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
