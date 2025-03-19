// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace testing;
using namespace ov::gen_pattern;

static ov::OutputVector makeCosSinCache(size_t max_position_embeddings, size_t rotary_ndims) {
    std::vector<float> lut_sin(max_position_embeddings * rotary_ndims, 0.0f);
    std::vector<float> lut_cos(max_position_embeddings * rotary_ndims, 0.0f);

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (size_t i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
        auto xita_i = 1.0 / std::pow(10000.0, static_cast<double>(i) / rotary_ndims);
        float* psin = lut_sin.data();
        float* pcos = lut_cos.data();
        for (size_t m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
            auto vsin = std::sin(xita_i * m);
            auto vcos = std::cos(xita_i * m);
            pcos[k] = pcos[k + rotary_ndims / 2] = vcos;
            psin[k] = psin[k + rotary_ndims / 2] = vsin;
        }
    }
    auto Cos = makeConst(ov::element::f32, ov::Shape({1, 1, max_position_embeddings, rotary_ndims}), lut_cos);
    auto Sin = makeConst(ov::element::f32, ov::Shape({1, 1, max_position_embeddings, rotary_ndims}), lut_sin);

    return {Cos, Sin};
}

static std::shared_ptr<ov::Model> buildROPE_Llama2(const size_t batch,
                                                   const size_t seq_length,
                                                   const size_t max_position_embeddings,
                                                   const size_t ndims,
                                                   bool sin_cos_preprocessing) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, 32, ndims});
    auto param_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});
    auto param_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});

    auto seq_len = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
    auto gather_id = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, seq_length});

    auto gather_from_sin_cos = [&](const ov::Output<ov::Node>& const_tab) {
        auto ScatterUpdate_152236 = makeOP<ov::opset3::ScatterUpdate>({{0, 0, 0}, {2}, seq_len, {0}});
        auto slice_Slice = makeOP<ov::opset1::StridedSlice>({const_tab, {0, 0, 0}, ScatterUpdate_152236, {1, 1, 1}},
                                                            {{"begin_mask", {1, 1, 0}},
                                                             {"end_mask", {1, 1, 0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
        auto squeeze_Squeeze_435 =
            makeOP<ov::opset1::Reshape>({slice_Slice, {-1, static_cast<int>(ndims)}}, {{"special_zero", false}});
        auto index_441_Gather = makeOP<ov::opset8::Gather>({squeeze_Squeeze_435, gather_id, {0}}, {{"batch_dims", 0}});
        return makeOP<ov::opset1::Reshape>({index_441_Gather, {1, 1, -1, static_cast<int>(ndims)}},
                                           {{"special_zero", false}});
    };

    ov::OutputVector cos_sin(2);
    ov::ParameterVector parameters;
    if (sin_cos_preprocessing) {
        auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);
        cos_sin[0] = gather_from_sin_cos(cos_sin_cache[0]);
        cos_sin[1] = gather_from_sin_cos(cos_sin_cache[1]);
        parameters = ov::ParameterVector{input, seq_len, gather_id};
    } else {
        cos_sin[0] = param_cos;
        cos_sin[1] = param_sin;
        parameters = ov::ParameterVector{input, param_cos, param_sin};
    }

    auto transpose_Transpose = makeOP<ov::opset1::Transpose>({input, {0, 2, 1, 3}});
    auto mul_Multiply = makeOP<ov::opset1::Multiply>({transpose_Transpose, cos_sin[0]}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_459 =
        makeOP<ov::opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 64}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto Constant_182988 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto neg_Multiply = makeOP<ov::opset1::Multiply>({slice_Slice_459, Constant_182988}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice =
        makeOP<ov::opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 0}, {0, 0, 0, 64}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_Slice}, {{"axis", -1}});
    auto mul_Multiply_463 = makeOP<ov::opset1::Multiply>({cat_Concat, cos_sin[1]}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::opset1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, parameters);
}

TEST_F(TransformationTestsF, ConvertToROPE_LLama2_no_gather) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_length = 16;
    const size_t max_position_embeddings = 2048;
    const size_t ndims = 128;
    const size_t num_head = 32;

    model = buildROPE_Llama2(batch, seq_length, max_position_embeddings, ndims, false);
    manager.register_pass<ov::pass::RoPEFusion>();

    {
        auto hidden_states =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, num_head, ndims});
        auto param_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});
        auto param_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});
        auto add_Add = makeOP<ov::op::internal::RoPE>({hidden_states, param_cos, param_sin},
                                                      {{"config.slice_start", 0},
                                                       {"config.slice_stop", 0},
                                                       {"config.input_trans0213", true},
                                                       {"config.output_trans0213", false},
                                                       {"config.is_interleaved", false},
                                                       {"config.is_chatglm", false},
                                                       {"config.support_2d_rope", false},
                                                       {"config.is_qwen", false},
                                                       {"config.head_cnt", 0},
                                                       {"config.head_size", 0},
                                                       {"config.rotary_ndims", static_cast<int>(ndims)},
                                                       {"config.gather_position_arg_id", 0}});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{add_Add},
                                                ov::ParameterVector{hidden_states, param_cos, param_sin});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_LLama2_with_gather) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_length = 16;
    const size_t max_position_embeddings = 2048;
    const size_t ndims = 128;
    const size_t num_head = 32;

    model = buildROPE_Llama2(batch, seq_length, max_position_embeddings, ndims, true);
    manager.register_pass<ov::pass::RoPEFusion>();

    {
        auto hidden_states =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, num_head, ndims});
        auto seq_len = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto gather_id = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, seq_length});
        auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);

        auto add_Add = makeOP<ov::op::internal::RoPE>({hidden_states, cos_sin_cache[0], cos_sin_cache[1], gather_id},
                                                      {{"config.slice_start", 0},
                                                       {"config.slice_stop", 0},
                                                       {"config.input_trans0213", true},
                                                       {"config.output_trans0213", false},
                                                       {"config.is_interleaved", false},
                                                       {"config.is_chatglm", false},
                                                       {"config.support_2d_rope", false},
                                                       {"config.is_qwen", false},
                                                       {"config.head_cnt", 0},
                                                       {"config.head_size", 0},
                                                       {"config.rotary_ndims", static_cast<int>(ndims)},
                                                       {"config.gather_position_arg_id", 3}});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{add_Add},
                                                ov::ParameterVector{hidden_states, seq_len, gather_id});
    }
}

static std::shared_ptr<ov::Model> buildROPE_GPTNEOX(const int batch,
                                                    const int seq_length,
                                                    const int max_position_embeddings,
                                                    const int ndims,
                                                    const int num_heads,
                                                    const int rotary_ndims,
                                                    bool sin_cos_preprocessing) {
    auto batch_s = static_cast<size_t>(batch);
    auto seq_length_s = static_cast<size_t>(seq_length);
    auto ndims_s = static_cast<size_t>(ndims);
    auto rotary_ndims_s = static_cast<size_t>(rotary_ndims);
    auto num_heads_s = static_cast<size_t>(num_heads);

    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                         ov::Shape{batch_s, seq_length_s, num_heads_s, ndims_s * 3});
    auto seq_len = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
    auto gather_idx =
        std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, 1, seq_length_s, rotary_ndims_s});
    auto batch_limit = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});

    ov::ParameterVector parameters;
    ov::OutputVector cos_sin(2);
    if (sin_cos_preprocessing) {
        auto cos_sin_lut = makeCosSinCache(max_position_embeddings, rotary_ndims);
        auto ro_slice_Slice = makeOP<ov::opset1::StridedSlice>({cos_sin_lut[0], {0}, batch_limit, {1}},
                                                               {{"begin_mask", {0}},
                                                                {"end_mask", {0}},
                                                                {"new_axis_mask", {}},
                                                                {"shrink_axis_mask", {}},
                                                                {"ellipsis_mask", {}}});
        cos_sin[0] = makeOP<ov::opset6::GatherElements>({ro_slice_Slice, gather_idx}, {{"axis", 2}});

        auto ro_slice_Slice_385 = makeOP<ov::opset1::StridedSlice>({cos_sin_lut[1], {0}, batch_limit, {1}},
                                                                   {{"begin_mask", {0}},
                                                                    {"end_mask", {0}},
                                                                    {"new_axis_mask", {}},
                                                                    {"shrink_axis_mask", {}},
                                                                    {"ellipsis_mask", {}}});
        cos_sin[1] = makeOP<ov::opset6::GatherElements>({ro_slice_Slice_385, gather_idx}, {{"axis", 2}});
        parameters = ov::ParameterVector{input, gather_idx, batch_limit};
    } else {
        auto param_cos =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length_s, rotary_ndims_s});
        auto param_sin =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length_s, rotary_ndims_s});
        parameters = ov::ParameterVector{input, param_cos, param_sin};
        cos_sin[0] = param_cos;
        cos_sin[1] = param_sin;
    }

    auto slice_Slice = makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0, 0}, {0, 0, 0, ndims}, {1, 1, 1, 1}},
                                                        {{"begin_mask", {1, 1, 1, 0}},
                                                         {"end_mask", {1, 1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto permute_Transpose = makeOP<ov::opset1::Transpose>({slice_Slice, {0, 2, 1, 3}});
    auto slice_Slice_351 =
        makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto mul_Multiply = makeOP<ov::opset1::Multiply>({slice_Slice_351, cos_sin[0]}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_420 = makeOP<ov::opset1::StridedSlice>(
        {slice_Slice_351, {0, 0, 0, rotary_ndims / 2}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
        {{"begin_mask", {1, 1, 1, 0}},
         {"end_mask", {1, 1, 1, 0}},
         {"new_axis_mask", {}},
         {"shrink_axis_mask", {}},
         {"ellipsis_mask", {}}});
    auto Constant_396096 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto neg_Multiply = makeOP<ov::opset1::Multiply>({slice_Slice_420, Constant_396096}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_414 =
        makeOP<ov::opset1::StridedSlice>({slice_Slice_351, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims / 2}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_Slice_414}, {{"axis", -1}});
    auto mul_Multiply_424 = makeOP<ov::opset1::Multiply>({cat_Concat, cos_sin[1]}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::opset1::Add>({mul_Multiply, mul_Multiply_424}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_357 =
        makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat_458 = makeOP<ov::opset1::Concat>({add_Add, slice_Slice_357}, {{"axis", -1}});

    return std::make_shared<ov::Model>(ov::NodeVector{cat_Concat_458}, parameters);
}

TEST_F(TransformationTestsF, ConvertToROPE_GPTNEOX_no_gather) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 16;
    const int ndims = 80;
    const int num_heads = 32;
    const int rotary_ndims = 20;
    const int max_position_embeddings = 2048;

    model = buildROPE_GPTNEOX(batch, seq_len, max_position_embeddings, ndims, num_heads, rotary_ndims, false);
    manager.register_pass<ov::pass::RoPEFusion>();
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims * 3});
        auto param_cos =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_len, rotary_ndims});
        auto param_sin =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_len, rotary_ndims});
        auto rope = makeOP<ov::op::internal::RoPE>({input, param_cos, param_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", ndims},
                                                    {"config.input_trans0213", true},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, param_cos, param_sin});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_GPTNEOX_with_gather) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 16;
    const int ndims = 80;
    const int rotary_ndims = 20;
    const int num_heads = 32;
    const int max_position_embeddings = 2048;

    model = buildROPE_GPTNEOX(batch, seq_len, max_position_embeddings, ndims, num_heads, rotary_ndims, true);
    manager.register_pass<ov::pass::RoPEFusion>();
    {
        auto cos_sin = makeCosSinCache(max_position_embeddings, rotary_ndims);
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims * 3});
        auto gather_idx =
            std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, 1, seq_len, rotary_ndims});
        auto batch_limit = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});

        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin[0], cos_sin[1], gather_idx},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", ndims},
                                                    {"config.input_trans0213", true},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 3}});
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, gather_idx, batch_limit});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_GPTJ) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 16;
    const int ndims = 256;
    const int rotary_ndims = 64;
    {
        std::vector<int32_t> rpi_idx(rotary_ndims);
        for (int i = 0, index = 0; i < rotary_ndims; i += 2, index++) {
            rpi_idx[i] = index;
            rpi_idx[i + 1] = index;
        }
        auto repeat_interleave_index = makeConst(ov::element::i32, ov::Shape({rotary_ndims}), rpi_idx);

        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims});
        auto gather_sin_cos =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, seq_len, rotary_ndims});

        auto split = makeOP<ov::opset1::VariadicSplit>({gather_sin_cos, {-1}, {rotary_ndims / 2, -1}});
        auto sin_tab =
            makeOP<ov::opset1::Reshape>({split->output(0), {1, -1, 1, rotary_ndims / 2}}, {{"special_zero", false}});
        auto cos_tab =
            makeOP<ov::opset1::Reshape>({split->output(1), {1, -1, 1, rotary_ndims / 2}}, {{"special_zero", false}});

        auto slice_Slice_576 =
            makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto repeat_interleave_Cos =
            makeOP<ov::opset8::Gather>({cos_tab, repeat_interleave_index, {3}}, {{"batch_dims", 0}});
        auto mul_Multiply_757 =
            makeOP<ov::opset1::Multiply>({slice_Slice_576, repeat_interleave_Cos}, {{"auto_broadcast", "numpy"}});

        auto slice_Slice_787 =
            makeOP<ov::opset1::StridedSlice>({slice_Slice_576, {0, 0, 0, 1}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto Constant_191672 = makeConst(ov::element::f32,
                                         ov::Shape({
                                             1,
                                             1,
                                             1,
                                             1,
                                         }),
                                         {-1.000000f});
        auto neg_Multiply_790 =
            makeOP<ov::opset1::Multiply>({slice_Slice_787, Constant_191672}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_61918 = makeOP<ov::opset1::Unsqueeze>({neg_Multiply_790, {-1}});
        auto slice_Slice_781 =
            makeOP<ov::opset1::StridedSlice>({slice_Slice_576, {0, 0, 0, 0}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto Unsqueeze_61919 = makeOP<ov::opset1::Unsqueeze>({slice_Slice_781, {-1}});
        auto stack_795 = makeOP<ov::opset1::Concat>({Unsqueeze_61918, Unsqueeze_61919}, {{"axis", -1}});
        auto ShapeOf_165368 = makeOP<ov::op::TypeRelaxed<ov::opset1::ShapeOf>>(
            {stack_795},
            {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {ov::element::i32}}});
        auto flatten_Slice_811 = makeOP<ov::opset1::StridedSlice>({ShapeOf_165368, {0}, {3}, {1}},
                                                                  {{"begin_mask", {0}},
                                                                   {"end_mask", {0}},
                                                                   {"new_axis_mask", {}},
                                                                   {"shrink_axis_mask", {}},
                                                                   {"ellipsis_mask", {}}});
        auto flatten_Concat_814 = makeOP<ov::opset1::Concat>({flatten_Slice_811, {-1}}, {{"axis", 0}});
        auto flatten_Reshape_815 =
            makeOP<ov::opset1::Reshape>({stack_795, flatten_Concat_814}, {{"special_zero", true}});
        auto repeat_interleave_Sin =
            makeOP<ov::opset8::Gather>({sin_tab, repeat_interleave_index, {3}}, {{"batch_dims", 0}});
        auto mul_Multiply_816 =
            makeOP<ov::opset1::Multiply>({flatten_Reshape_815, repeat_interleave_Sin}, {{"auto_broadcast", "numpy"}});
        auto add_Add_819 = makeOP<ov::opset1::Add>({mul_Multiply_757, mul_Multiply_816}, {{"auto_broadcast", "numpy"}});
        auto slice_Slice_582 =
            makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto cat_Concat_826 = makeOP<ov::opset1::Concat>({add_Add_819, slice_Slice_582}, {{"axis", -1}});
        auto permute_Transpose_828 = makeOP<ov::opset1::Transpose>({cat_Concat_826, {0, 2, 1, 3}});
        model = std::make_shared<ov::Model>(ov::NodeVector{permute_Transpose_828},
                                            ov::ParameterVector{input, gather_sin_cos});
    }
    manager.register_pass<ov::pass::RoPEFusion>();
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims});
        auto cos_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, seq_len, rotary_ndims});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin, cos_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 0},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", true},
                                                    {"config.is_interleaved", true},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, cos_sin});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGML) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 32;
    const int ndims = 128;
    const int rotary_ndims = 64;
    const int max_pos_length = 2048;
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, 4608});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});

        auto ListUnpack_321 = makeOP<ov::opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
        auto view_Reshape = makeOP<ov::opset1::Reshape>({ListUnpack_321->output(0), {0, 0, num_heads, ndims}},
                                                        {{"special_zero", true}});
        auto aten_slice_Slice_357 =
            makeOP<ov::opset1::StridedSlice>({view_Reshape, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto ListConstruct_372_Concat =
            makeOP<ov::opset1::Concat>({seq_length, {-1}, {num_heads}, {rotary_ndims / 2}, {2}}, {{"axis", 0}});
        auto aten_reshape_Reshape_373 =
            makeOP<ov::opset1::Reshape>({aten_slice_Slice_357, ListConstruct_372_Concat}, {{"special_zero", false}});
        auto aten_select_Gather_381 =
            makeOP<ov::opset8::Gather>({aten_reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});
        auto aten_slice_Slice_369 = makeOP<ov::opset1::StridedSlice>({cos_sin_cache, {0}, seq_length, {1}},
                                                                     {{"begin_mask", {0}},
                                                                      {"end_mask", {0}},
                                                                      {"new_axis_mask", {}},
                                                                      {"shrink_axis_mask", {}},
                                                                      {"ellipsis_mask", {}}});
        auto ListConstruct_379_Concat =
            makeOP<ov::opset1::Concat>({seq_length, {-1}, {1}, {rotary_ndims / 2}, {2}}, {{"axis", 0}});
        auto aten_view_Reshape_380 =
            makeOP<ov::opset1::Reshape>({aten_slice_Slice_369, ListConstruct_379_Concat}, {{"special_zero", false}});
        auto aten_select_Gather_382 = makeOP<ov::opset8::Gather>({aten_view_Reshape_380, 0, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_383 = makeOP<ov::opset1::Multiply>({aten_select_Gather_381, aten_select_Gather_382},
                                                                  {{"auto_broadcast", "numpy"}});
        auto aten_select_Gather_384 =
            makeOP<ov::opset8::Gather>({aten_reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});
        auto aten_select_Gather_385 = makeOP<ov::opset8::Gather>({aten_view_Reshape_380, 1, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_386 = makeOP<ov::opset1::Multiply>({aten_select_Gather_384, aten_select_Gather_385},
                                                                  {{"auto_broadcast", "numpy"}});
        auto Multiply_101315 =
            makeOP<ov::opset1::Multiply>({aten_mul_Multiply_386, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto aten_sub_Subtract_389 =
            makeOP<ov::opset1::Add>({aten_mul_Multiply_383, Multiply_101315}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_62716 = makeOP<ov::opset1::Unsqueeze>({aten_sub_Subtract_389, -1});
        auto aten_mul_Multiply_391 = makeOP<ov::opset1::Multiply>({aten_select_Gather_384, aten_select_Gather_382},
                                                                  {{"auto_broadcast", "numpy"}});
        auto aten_mul_Multiply_393 = makeOP<ov::opset1::Multiply>({aten_select_Gather_381, aten_select_Gather_385},
                                                                  {{"auto_broadcast", "numpy"}});
        auto aten_add_Add_396 =
            makeOP<ov::opset1::Add>({aten_mul_Multiply_391, aten_mul_Multiply_393}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_62717 = makeOP<ov::opset1::Unsqueeze>({aten_add_Add_396, -1});
        auto aten_stack_401 = makeOP<ov::opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
        auto ShapeOf_134820 = makeOP<ov::op::TypeRelaxed<ov::opset1::ShapeOf>>(
            {aten_stack_401},
            {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {ov::element::i32}}});
        auto aten_flatten_Slice_417 = makeOP<ov::opset1::StridedSlice>({ShapeOf_134820, {0}, {3}, {1}},
                                                                       {{"begin_mask", {0}},
                                                                        {"end_mask", {0}},
                                                                        {"new_axis_mask", {}},
                                                                        {"shrink_axis_mask", {}},
                                                                        {"ellipsis_mask", {}}});
        auto aten_flatten_Concat_420 = makeOP<ov::opset1::Concat>({aten_flatten_Slice_417, {-1}}, {{"axis", 0}});
        auto aten_flatten_Reshape_421 =
            makeOP<ov::opset1::Reshape>({aten_stack_401, aten_flatten_Concat_420}, {{"special_zero", true}});
        auto aten_slice_Slice_363 =
            makeOP<ov::opset1::StridedSlice>({view_Reshape, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto aten_cat_Concat_425 =
            makeOP<ov::opset1::Concat>({aten_flatten_Reshape_421, aten_slice_Slice_363}, {{"axis", -1}});
        model = std::make_shared<ov::Model>(ov::NodeVector{aten_cat_Concat_425},
                                            ov::ParameterVector{input, seq_length, cos_sin_cache});
    }
    manager.register_pass<ov::pass::RoPEFusion>();
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, 4608});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin_cache, cos_sin_cache},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 4096},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, seq_length, cos_sin_cache});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGML_Slice) {
    using namespace ov;
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 32;
    const int ndims = 128;
    const int rotary_ndims = 64;
    const int max_pos_length = 2048;
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, 4608});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});
        auto ListUnpack = makeOP<opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
        auto view_Reshape =
            makeOP<opset1::Reshape>({ListUnpack->output(0), {0, 0, num_heads, ndims}}, {{"special_zero", true}});

        auto VariadicSplit_20795 = makeOP<opset1::VariadicSplit>({view_Reshape, 3, {rotary_ndims, -1}});
        auto reshape_Reshape =
            makeOP<opset1::Reshape>({VariadicSplit_20795->output(0), {0, 0, num_heads, rotary_ndims / 2, 2}},
                                    {{"special_zero", true}});

        auto select_Gather = makeOP<opset8::Gather>({reshape_Reshape, 0, -1}, {{"batch_dims", 0}});
        auto slice_Slice_1 = makeOP<opset8::Slice>({cos_sin_cache, {0}, seq_length, {1}, {0}});
        auto ListConstruct_Concat_1 =
            makeOP<opset1::Concat>({seq_length, {-1}, {1}, {rotary_ndims / 2}, {2}}, {{"axis", 0}});
        auto view_Reshape_1 =
            makeOP<opset1::Reshape>({slice_Slice_1, ListConstruct_Concat_1}, {{"special_zero", false}});

        auto select_Gather_1 = makeOP<opset8::Gather>({view_Reshape_1, 0, -1}, {{"batch_dims", 0}});
        auto mul_Multiply_1 = makeOP<opset1::Multiply>({select_Gather, select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto select_Gather_2 = makeOP<opset8::Gather>({reshape_Reshape, 1, -1}, {{"batch_dims", 0}});
        auto select_Gather_3 = makeOP<opset8::Gather>({view_Reshape_1, 1, -1}, {{"batch_dims", 0}});
        auto mul_Multiply_2 =
            makeOP<opset1::Multiply>({select_Gather_2, select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto Multiply_23724 = makeOP<opset1::Multiply>({mul_Multiply_2, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto sub_Subtract = makeOP<opset1::Add>({mul_Multiply_1, Multiply_23724}, {{"auto_broadcast", "numpy"}});

        auto Unsqueeze_57121 = makeOP<opset1::Unsqueeze>({sub_Subtract, -1});
        auto mul_Multiply_3 =
            makeOP<opset1::Multiply>({select_Gather_2, select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto mul_Multiply_4 = makeOP<opset1::Multiply>({select_Gather, select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto add_Add = makeOP<opset1::Add>({mul_Multiply_3, mul_Multiply_4}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_57122 = makeOP<opset1::Unsqueeze>({add_Add, -1});
        auto stack = makeOP<opset1::Concat>({Unsqueeze_57121, Unsqueeze_57122}, {{"axis", -1}});
        auto flatten_Reshape =
            makeOP<opset1::Reshape>({stack, {0, 0, num_heads, rotary_ndims}}, {{"special_zero", true}});
        auto cat_Concat = makeOP<opset1::Concat>({flatten_Reshape, VariadicSplit_20795->output(1)}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::NodeVector{cat_Concat},
                                            ov::ParameterVector{input, seq_length, cos_sin_cache});
    }
    manager.register_pass<ov::pass::RoPEFusion>();
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, 4608});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin_cache, cos_sin_cache},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 4096},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, seq_length, cos_sin_cache});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_GPTJ_Slice) {
    disable_rt_info_check();
    using namespace ov;

    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 16;
    const int ndims = 256;
    const int rotary_ndims = 64;
    {
        std::vector<int32_t> rpi_idx(rotary_ndims);
        for (int i = 0, index = 0; i < rotary_ndims; i += 2, index++) {
            rpi_idx[i] = index;
            rpi_idx[i + 1] = index;
        }
        auto repeat_interleave_index = makeConst(ov::element::i32, ov::Shape({rotary_ndims}), rpi_idx);

        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims});
        auto gather_sin_cos =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, seq_len, rotary_ndims});

        auto ListUnpack_VariadicSplit = makeOP<opset1::VariadicSplit>({gather_sin_cos, -1, {rotary_ndims / 2, -1}});
        auto sin_tab = makeOP<opset1::Unsqueeze>({ListUnpack_VariadicSplit->output(0), 2});
        auto cos_tab = makeOP<opset1::Unsqueeze>({ListUnpack_VariadicSplit->output(1), 2});

        auto repeat_interleave_Sin =
            makeOP<opset8::Gather>({sin_tab, repeat_interleave_index, {3}}, {{"batch_dims", 0}});
        auto repeat_interleave_Cos =
            makeOP<opset8::Gather>({cos_tab, repeat_interleave_index, {3}}, {{"batch_dims", 0}});

        auto VariadicSplit_39740 = makeOP<opset1::VariadicSplit>({input, 3, {rotary_ndims, -1}});

        auto mul_Multiply = makeOP<opset1::Multiply>({VariadicSplit_39740->output(0), repeat_interleave_Cos},
                                                     {{"auto_broadcast", "numpy"}});
        auto slice_Slice_10 = makeOP<opset8::Slice>({VariadicSplit_39740->output(0), {1}, {INT_MAX}, {2}, {3}});
        auto Constant_134252 = makeConst(element::f32,
                                         ov::Shape({
                                             1,
                                             1,
                                             1,
                                             1,
                                         }),
                                         {-1.000000f});

        auto neg_Multiply = makeOP<opset1::Multiply>({slice_Slice_10, Constant_134252}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_47361 = makeOP<opset1::Unsqueeze>({neg_Multiply, -1});
        auto slice_Slice_14 = makeOP<opset8::Slice>({VariadicSplit_39740->output(0), {0}, {INT_MAX}, {2}, {3}});
        auto Unsqueeze_47362 = makeOP<opset1::Unsqueeze>({slice_Slice_14, -1});
        auto stack = makeOP<opset1::Concat>({Unsqueeze_47361, Unsqueeze_47362}, {{"axis", -1}});
        auto flatten_Reshape = makeOP<opset1::Reshape>({stack, {0, 0, 16, rotary_ndims}}, {{"special_zero", true}});
        auto mul_Multiply_1 =
            makeOP<opset1::Multiply>({flatten_Reshape, repeat_interleave_Sin}, {{"auto_broadcast", "numpy"}});
        auto add_Add = makeOP<opset1::Add>({mul_Multiply, mul_Multiply_1}, {{"auto_broadcast", "numpy"}});
        auto cat_Concat = makeOP<opset1::Concat>({add_Add, VariadicSplit_39740->output(1)}, {{"axis", -1}});
        auto permute_Transpose = makeOP<opset1::Transpose>({cat_Concat, {0, 2, 1, 3}});

        model =
            std::make_shared<ov::Model>(ov::NodeVector{permute_Transpose}, ov::ParameterVector{input, gather_sin_cos});
    }
    manager.register_pass<ov::pass::RoPEFusion>();
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims});
        auto cos_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, seq_len, rotary_ndims});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin, cos_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 0},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", true},
                                                    {"config.is_interleaved", true},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, cos_sin});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGML_2d_rope) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 32;
    const int ndims = 128;
    const int rotary_ndims = 64;
    const int max_pos_length = 2048;
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, seq_len, 4608});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::PartialShape{max_pos_length, (rotary_ndims / 2), 2});
        auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{batch, seq_len});

        auto __module_transformer_index_67_Gather =
            makeOP<ov::opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});

        auto ListUnpack_321 = makeOP<ov::opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
        auto view_Reshape = makeOP<ov::opset1::Reshape>({ListUnpack_321->output(0), {0, 0, num_heads, ndims}},
                                                        {{"special_zero", true}});

        auto permute_Transpose = makeOP<ov::opset1::Transpose>({view_Reshape, {0, 2, 1, 3}}, {});

        auto slice_Slice_357 =
            makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});

        auto aten_view_Reshape_1 =
            makeOP<ov::opset1::Reshape>({ListUnpack_321->output(1), {0, 0, 2, ndims}}, {{"special_zero", true}});
        auto aten_transpose_1 = makeOP<ov::opset8::Transpose>({aten_view_Reshape_1, {0, 2, 1, 3}});
        auto shape_of_105249 = makeOP<ov::opset8::ShapeOf>({aten_transpose_1}, {{"output_type", "i32"}});
        auto gather_105252 = makeOP<ov::opset8::Gather>({shape_of_105249, {2}, {0}}, {{"batch_dims", 0}});
        auto scatter_update_63441 = makeOP<ov::opset8::ScatterUpdate>({{0, 0}, {1}, gather_105252, {0}});
        // connected to cos_sin_cache
        auto slice_Slice_369 = makeOP<ov::opset1::StridedSlice>(
            {__module_transformer_index_67_Gather, {0, 0}, scatter_update_63441, {1, 1}},
            {{"begin_mask", {1, 0}},
             {"end_mask", {1, 0}},
             {"new_axis_mask", {}},
             {"shrink_axis_mask", {}},
             {"ellipsis_mask", {}}});
        auto list_construct_concat_1 =
            makeOP<ov::opset1::Concat>({{-1}, {1}, gather_105252, {rotary_ndims / 2}, {2}}, {{"axis", 0}});

        auto reshape_Reshape_373 =
            makeOP<ov::opset1::Reshape>({slice_Slice_357, {0, 32, 0, 32, 2}}, {{"special_zero", true}});
        auto select_Gather_384 =
            makeOP<ov::opset8::Gather>({reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});  // x_even
        auto select_Gather_381 =
            makeOP<ov::opset8::Gather>({reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});  // x_odd
        auto view_Reshape_380 =
            makeOP<ov::opset1::Reshape>({slice_Slice_369, list_construct_concat_1}, {{"special_zero", false}});
        auto select_Gather_385 = makeOP<ov::opset8::Gather>({view_Reshape_380, 0, -1}, {{"batch_dims", 0}});  // cos_tab
        auto select_Gather_382 = makeOP<ov::opset8::Gather>({view_Reshape_380, 1, -1}, {{"batch_dims", 0}});  // sin_tab

        auto mul_Multiply_386 = makeOP<ov::opset1::Multiply>({select_Gather_381, select_Gather_382},
                                                             {{"auto_broadcast", "numpy"}});  // x_odd_sin
        auto mul_Multiply_383 = makeOP<ov::opset1::Multiply>({select_Gather_384, select_Gather_385},
                                                             {{"auto_broadcast", "numpy"}});  // x_even_cos
        auto Multiply_101315 =
            makeOP<ov::opset1::Multiply>({mul_Multiply_386, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto sub_Subtract_389 =
            makeOP<ov::opset1::Add>({mul_Multiply_383, Multiply_101315}, {{"auto_broadcast", "numpy"}});

        auto mul_Multiply_391 = makeOP<ov::opset1::Multiply>({select_Gather_381, select_Gather_385},
                                                             {{"auto_broadcast", "numpy"}});  // x_odd_cos
        auto mul_Multiply_393 = makeOP<ov::opset1::Multiply>({select_Gather_384, select_Gather_382},
                                                             {{"auto_broadcast", "numpy"}});  // x_even_sin
        auto add_Add_396 = makeOP<ov::opset1::Add>({mul_Multiply_391, mul_Multiply_393}, {{"auto_broadcast", "numpy"}});

        auto Unsqueeze_62716 = makeOP<ov::opset1::Unsqueeze>({sub_Subtract_389, -1}, {});
        auto Unsqueeze_62717 = makeOP<ov::opset1::Unsqueeze>({add_Add_396, -1}, {});

        auto stack_401 = makeOP<ov::opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
        auto flatten_Reshape_421 =
            makeOP<ov::opset1::Reshape>({stack_401, {0, num_heads, 0, rotary_ndims}}, {{"special_zero", true}});
        auto slice_Slice_363 = makeOP<ov::opset1::StridedSlice>(
            {permute_Transpose, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
            {{"begin_mask", {1, 1, 1, 0}},
             {"end_mask", {1, 1, 1, 0}},
             {"new_axis_mask", {}},
             {"shrink_axis_mask", {}},
             {"ellipsis_mask", {}}});
        auto cat_Concat_425 = makeOP<ov::opset1::Concat>({flatten_Reshape_421, slice_Slice_363}, {{"axis", -1}});
        model = std::make_shared<ov::Model>(ov::NodeVector{cat_Concat_425},
                                            ov::ParameterVector{input, cos_sin_cache, position_ids});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, 4608});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{max_pos_length, (rotary_ndims / 2), 2});
        auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{batch, seq_len});
        auto gather_cos_sin = makeOP<ov::opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});
        auto rope = makeOP<ov::op::internal::RoPE>({input, gather_cos_sin, gather_cos_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 4096},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", true},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, cos_sin_cache, position_ids});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGML_nano_2d_rope) {
    disable_rt_info_check();
    const int batch = 2;
    const int seq_len = 7;
    const int num_heads = 16;
    const int ndims = 128;
    const int rotary_ndims = 128;
    const int max_pos_length = 2048;
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, seq_len, 3072});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::PartialShape{max_pos_length, (rotary_ndims / 2), 2});
        auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{batch, seq_len});

        auto __module_transformer_index_67_Gather =
            makeOP<ov::opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});

        auto ListUnpack_321 = makeOP<ov::opset1::VariadicSplit>({input, -1, {2048, 512, 512}});
        auto view_Reshape = makeOP<ov::opset1::Reshape>({ListUnpack_321->output(0), {0, 0, num_heads, ndims}},
                                                        {{"special_zero", true}});

        auto permute_Transpose = makeOP<ov::opset1::Transpose>({view_Reshape, {0, 2, 1, 3}}, {});

        auto slice_Slice_357 =
            makeOP<ov::opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});

        auto aten_view_Reshape_1 =
            makeOP<ov::opset1::Reshape>({ListUnpack_321->output(1), {0, 0, 2, ndims}}, {{"special_zero", true}});
        auto aten_transpose_1 = makeOP<ov::opset8::Transpose>({aten_view_Reshape_1, {0, 2, 1, 3}});
        auto shape_of_105249 = makeOP<ov::opset8::ShapeOf>({aten_transpose_1}, {{"output_type", "i32"}});
        auto gather_105252 = makeOP<ov::opset8::Gather>({shape_of_105249, {2}, {0}}, {{"batch_dims", 0}});
        auto scatter_update_63441 = makeOP<ov::opset8::ScatterUpdate>({{0, 0}, {1}, gather_105252, {0}});
        // connected to cos_sin_cache
        auto slice_Slice_369 = makeOP<ov::opset1::StridedSlice>(
            {__module_transformer_index_67_Gather, {0, 0}, scatter_update_63441, {1, 1}},
            {{"begin_mask", {1, 0}},
             {"end_mask", {1, 0}},
             {"new_axis_mask", {}},
             {"shrink_axis_mask", {}},
             {"ellipsis_mask", {}}});
        auto list_construct_concat_1 =
            makeOP<ov::opset1::Concat>({{-1}, {1}, gather_105252, {rotary_ndims / 2}, {2}}, {{"axis", 0}});

        auto reshape_Reshape_373 =
            makeOP<ov::opset1::Reshape>({slice_Slice_357, {0, 16, 0, 64, 2}}, {{"special_zero", true}});
        auto select_Gather_384 =
            makeOP<ov::opset8::Gather>({reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});  // x_even
        auto select_Gather_381 =
            makeOP<ov::opset8::Gather>({reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});  // x_odd
        auto view_Reshape_380 =
            makeOP<ov::opset1::Reshape>({slice_Slice_369, list_construct_concat_1}, {{"special_zero", false}});
        auto select_Gather_385 = makeOP<ov::opset8::Gather>({view_Reshape_380, 0, -1}, {{"batch_dims", 0}});  // cos_tab
        auto select_Gather_382 = makeOP<ov::opset8::Gather>({view_Reshape_380, 1, -1}, {{"batch_dims", 0}});  // sin_tab

        auto mul_Multiply_386 = makeOP<ov::opset1::Multiply>({select_Gather_381, select_Gather_382},
                                                             {{"auto_broadcast", "numpy"}});  // x_odd_sin
        auto mul_Multiply_383 = makeOP<ov::opset1::Multiply>({select_Gather_384, select_Gather_385},
                                                             {{"auto_broadcast", "numpy"}});  // x_even_cos
        auto Multiply_101315 =
            makeOP<ov::opset1::Multiply>({mul_Multiply_386, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto sub_Subtract_389 =
            makeOP<ov::opset1::Add>({mul_Multiply_383, Multiply_101315}, {{"auto_broadcast", "numpy"}});

        auto mul_Multiply_391 = makeOP<ov::opset1::Multiply>({select_Gather_381, select_Gather_385},
                                                             {{"auto_broadcast", "numpy"}});  // x_odd_cos
        auto mul_Multiply_393 = makeOP<ov::opset1::Multiply>({select_Gather_384, select_Gather_382},
                                                             {{"auto_broadcast", "numpy"}});  // x_even_sin
        auto add_Add_396 = makeOP<ov::opset1::Add>({mul_Multiply_391, mul_Multiply_393}, {{"auto_broadcast", "numpy"}});

        auto Unsqueeze_62716 = makeOP<ov::opset1::Unsqueeze>({sub_Subtract_389, -1}, {});
        auto Unsqueeze_62717 = makeOP<ov::opset1::Unsqueeze>({add_Add_396, -1}, {});

        auto stack_401 = makeOP<ov::opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
        auto flatten_Reshape_421 =
            makeOP<ov::opset1::Reshape>({stack_401, {0, num_heads, 0, rotary_ndims}}, {{"special_zero", true}});
        model = std::make_shared<ov::Model>(ov::NodeVector{flatten_Reshape_421},
                                            ov::ParameterVector{input, cos_sin_cache, position_ids});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, 3072});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{max_pos_length, (rotary_ndims / 2), 2});
        auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{batch, seq_len});
        auto gather_cos_sin = makeOP<ov::opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});
        auto rope = makeOP<ov::op::internal::RoPE>({input, gather_cos_sin, gather_cos_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 2048},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", true},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, cos_sin_cache, position_ids});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_Flux_mul) {
    disable_rt_info_check();
    const int batch = 2;
    const int num_heads = 32;
    const int ndims = 128;
    {
        auto x =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
        auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});

        auto x1_shape = makeConst(ov::element::i64, ov::Shape({5}), {0, num_heads, 0, -1, 2});
        auto x1 = std::make_shared<ov::op::v1::Reshape>(x, x1_shape, true);

        auto split_axis = makeConst(ov::element::i64, ov::Shape(), {-1});
        auto split = std::make_shared<ov::op::v1::Split>(x1, split_axis, 2);

        auto minus_one = makeConst(ov::element::f32, ov::Shape({}), {-1.0f});
        auto x1_1_neg = std::make_shared<ov::op::v1::Multiply>(split->output(1), minus_one);

        auto x2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x1_1_neg->output(0), split->output(0)}, -1);

        auto x3_shape = makeConst(ov::element::i64, ov::Shape({4}), {0, num_heads, 0, ndims});
        auto x3 = std::make_shared<ov::op::v1::Reshape>(x2, x3_shape, true);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(x, t_cos);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x3, t_sin);
        auto y = std::make_shared<ov::op::v1::Add>(y1, y2);

        model = std::make_shared<ov::Model>(ov::NodeVector{y}, ov::ParameterVector{x, t_cos, t_sin});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto x =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
        auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        ov::op::internal::RoPE::Config config;
        config.is_interleaved = true;
        config.rotary_ndims = ndims;
        config.head_cnt = num_heads;
        config.head_size = ndims;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{x, t_cos, t_sin}, config);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{x, t_cos, t_sin});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertToROPE_Flux_squeeze_mul_unsqueeze) {
    disable_rt_info_check();
    const int batch = 2;
    const int num_heads = 32;
    const int ndims = 128;
    {
        auto x =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
        auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});

        auto x1_shape = makeConst(ov::element::i64, ov::Shape({5}), {0, num_heads, 0, -1, 2});
        auto x1 = std::make_shared<ov::op::v1::Reshape>(x, x1_shape, true);

        auto split_axis = makeConst(ov::element::i64, ov::Shape(), {-1});
        auto split = std::make_shared<ov::op::v1::Split>(x1, split_axis, 2);

        auto squeeze_axis = makeConst(ov::element::i32, ov::Shape({}), {-1});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(split->output(1), squeeze_axis);

        auto minus_one = makeConst(ov::element::f32, ov::Shape({}), {-1.0f});
        auto x1_1_neg = std::make_shared<ov::op::v1::Multiply>(squeeze, minus_one);

        auto unsqueeze_axis = makeConst(ov::element::i32, ov::Shape({}), {-1});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(x1_1_neg, unsqueeze_axis);

        auto x2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{unsqueeze->output(0), split->output(0)}, -1);

        auto x3_shape = makeConst(ov::element::i64, ov::Shape({4}), {0, num_heads, 0, ndims});
        auto x3 = std::make_shared<ov::op::v1::Reshape>(x2, x3_shape, true);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(x, t_cos);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x3, t_sin);
        auto y = std::make_shared<ov::op::v1::Add>(y1, y2);

        model = std::make_shared<ov::Model>(ov::NodeVector{y}, ov::ParameterVector{x, t_cos, t_sin});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto x =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
        auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        ov::op::internal::RoPE::Config config;
        config.is_interleaved = true;
        config.rotary_ndims = ndims;
        config.head_cnt = num_heads;
        config.head_size = ndims;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{x, t_cos, t_sin}, config);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{x, t_cos, t_sin});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertToROPE_Flux_mul_squeeze_unsqueeze) {
    disable_rt_info_check();
    const int batch = 2;
    const int num_heads = 32;
    const int ndims = 128;
    {
        auto x =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
        auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});

        auto x1_shape = makeConst(ov::element::i64, ov::Shape({5}), {0, num_heads, 0, -1, 2});
        auto x1 = std::make_shared<ov::op::v1::Reshape>(x, x1_shape, true);

        auto split_axis = makeConst(ov::element::i64, ov::Shape(), {-1});
        auto split = std::make_shared<ov::op::v1::Split>(x1, split_axis, 2);

        auto minus_one = makeConst(ov::element::f32, ov::Shape({}), {-1.0f});
        auto x1_1_neg = std::make_shared<ov::op::v1::Multiply>(split->output(1), minus_one);

        auto squeeze_axis = makeConst(ov::element::i32, ov::Shape({}), {-1});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(x1_1_neg, squeeze_axis);

        auto unsqueeze_axis = makeConst(ov::element::i32, ov::Shape({}), {-1});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(squeeze, unsqueeze_axis);

        auto x2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{unsqueeze->output(0), split->output(0)}, -1);

        auto x3_shape = makeConst(ov::element::i64, ov::Shape({4}), {0, num_heads, 0, ndims});
        auto x3 = std::make_shared<ov::op::v1::Reshape>(x2, x3_shape, true);

        auto y1 = std::make_shared<ov::op::v1::Multiply>(x, t_cos);
        auto y2 = std::make_shared<ov::op::v1::Multiply>(x3, t_sin);
        auto y = std::make_shared<ov::op::v1::Add>(y1, y2);

        model = std::make_shared<ov::Model>(ov::NodeVector{y}, ov::ParameterVector{x, t_cos, t_sin});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto x =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, num_heads, -1, ndims});
        auto t_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        auto t_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, ndims});
        ov::op::internal::RoPE::Config config;
        config.is_interleaved = true;
        config.rotary_ndims = ndims;
        config.head_cnt = num_heads;
        config.head_size = ndims;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{x, t_cos, t_sin}, config);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{x, t_cos, t_sin});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGLM3_PagedAttention) {
    disable_rt_info_check();
    const int batch = -1;
    const int seq_len = 1;
    const int num_heads = 32;
    const int num_heads_kv = 2;
    const int ndims = 128;
    const int rotary_ndims = 64;
    const int hidden_size = ndims * (num_heads + 2 * num_heads_kv);
    const int hidden_size_q = ndims * num_heads;
    const int hidden_size_kv = ndims * num_heads_kv;
    using namespace ov;
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, batch, hidden_size});
        auto cos_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                               ov::PartialShape{seq_len, batch, rotary_ndims / 2, 2});
        auto aten_slice_Slice_1 = makeOP<opset8::Slice>({cos_sin, {0}, {1}, {1}, {0}});
        auto aten_view_Reshape = makeOP<opset1::Reshape>({aten_slice_Slice_1, {seq_len, batch, 1, rotary_ndims / 2, 2}},
                                                         {{"special_zero", false}});
        auto aten_select_Gather_1 = makeOP<opset8::Gather>({aten_view_Reshape, 0, -1}, {{"batch_dims", 0}});
        auto aten_select_Gather_3 = makeOP<opset8::Gather>({aten_view_Reshape, 1, -1}, {{"batch_dims", 0}});

        auto attn_prim_ListUnpack =
            makeOP<opset1::VariadicSplit>({input, -1, {hidden_size_q, hidden_size_kv, hidden_size_kv}});
        auto attn_aten_view_Reshape_2 =
            makeOP<opset1::Reshape>({attn_prim_ListUnpack->output(0), {0, 0, num_heads, ndims}},
                                    {{"special_zero", true}});
        auto VariadicSplit_29663 =
            makeOP<opset1::VariadicSplit>({attn_aten_view_Reshape_2, 3, {rotary_ndims, ndims - rotary_ndims}});
        auto aten_reshape_Reshape_55 =
            makeOP<opset1::Reshape>({VariadicSplit_29663->output(0), {0, 0, num_heads, rotary_ndims / 2, 2}},
                                    {{"special_zero", true}});
        auto aten_select_Gather_440 = makeOP<opset8::Gather>({aten_reshape_Reshape_55, 0, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_276 =
            makeOP<opset1::Multiply>({aten_select_Gather_440, aten_select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto aten_select_Gather_442 = makeOP<opset8::Gather>({aten_reshape_Reshape_55, 1, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_277 =
            makeOP<opset1::Multiply>({aten_select_Gather_442, aten_select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto Multiply_34833 =
            makeOP<opset1::Multiply>({aten_mul_Multiply_277, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto aten_sub_Subtract_55 =
            makeOP<opset1::Add>({aten_mul_Multiply_276, Multiply_34833}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_62197 = makeOP<opset1::Reshape>({aten_sub_Subtract_55, {1, -1, num_heads, rotary_ndims / 2, 1}},
                                                       {{"special_zero", false}});
        auto aten_mul_Multiply_278 =
            makeOP<opset1::Multiply>({aten_select_Gather_442, aten_select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto aten_mul_Multiply_279 =
            makeOP<opset1::Multiply>({aten_select_Gather_440, aten_select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto aten_add_Add_55 =
            makeOP<opset1::Add>({aten_mul_Multiply_278, aten_mul_Multiply_279}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_62198 = makeOP<opset1::Reshape>({aten_add_Add_55, {1, -1, num_heads, rotary_ndims / 2, 1}},
                                                       {{"special_zero", false}});
        auto aten_stack_55 = makeOP<opset1::Concat>({Unsqueeze_62197, Unsqueeze_62198}, {{"axis", -1}});
        auto aten_flatten_Reshape_55 =
            makeOP<opset1::Reshape>({aten_stack_55, {0, 0, num_heads, rotary_ndims}}, {{"special_zero", true}});
        auto aten_cat_Concat_55 =
            makeOP<opset1::Concat>({aten_flatten_Reshape_55, VariadicSplit_29663->output(1)}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::NodeVector{aten_cat_Concat_55}, ov::ParameterVector{input, cos_sin});
    }
    manager.register_pass<ov::pass::RoPEFusion>(false);
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, batch, hidden_size});
        auto gather_cos_sin =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::PartialShape{seq_len, batch, rotary_ndims / 2, 2});
        auto rope = makeOP<ov::op::internal::RoPE>({input, gather_cos_sin, gather_cos_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 4096},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, gather_cos_sin});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_Qwen_PagedAttention) {
    using namespace ov;

    {
        auto position_ids = std::make_shared<opset1::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        auto qkv = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 3 * 4096});

        auto qkv_proj = makeOP<opset1::VariadicSplit>({qkv, 2, {4096, 4096, -1}});

        auto view_Reshape = makeOP<opset1::Reshape>({qkv_proj->output(0), {0, 0, 32, 128}}, {{"special_zero", true}});
        auto slice_Slice_4 = makeOP<opset8::Slice>({view_Reshape, {0}, {128}, {1}, {3}});
        auto slice_Slice = makeConst(element::f32, ov::Shape({1, 4096, 1, 128}), {1});

        auto Convert_50535 = makeOP<opset1::Convert>({position_ids}, {{"destination_type", "i32"}});
        auto Unsqueeze_23750 = makeOP<opset1::Reshape>({Convert_50535, {-1, 1}}, {{"special_zero", false}});

        auto slice_Slice_1 = makeOP<opset8::Gather>({slice_Slice, Unsqueeze_23750, 1}, {{"batch_dims", 0}});
        auto Reshape_27400 = makeOP<opset1::Reshape>({slice_Slice_1, {-1, 1, 1, 128}}, {{"special_zero", false}});

        auto mul_Multiply = makeOP<opset1::Multiply>({slice_Slice_4, Reshape_27400}, {{"auto_broadcast", "numpy"}});
        auto reshape_Reshape = makeOP<opset1::Reshape>({slice_Slice_4, {0, 0, 32, 2, 64}}, {{"special_zero", true}});
        auto ListUnpack_Split = makeOP<opset1::Split>({reshape_Reshape, -2}, {{"num_splits", 2}});
        auto Multiply_54136 =
            makeOP<opset1::Multiply>({ListUnpack_Split->output(1), -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto ListUnpack_Squeeze_0 =
            makeOP<opset1::Reshape>({Multiply_54136, {-1, 1, 32, 64}}, {{"special_zero", false}});
        auto ListUnpack_Squeeze =
            makeOP<opset1::Reshape>({ListUnpack_Split->output(0), {-1, 1, 32, 64}}, {{"special_zero", false}});
        auto cat_Concat = makeOP<opset1::Concat>({ListUnpack_Squeeze_0, ListUnpack_Squeeze}, {{"axis", -1}});

        auto slice_Slice_2 = makeConst(element::f32, ov::Shape({1, 4096, 1, 128}), {1});
        auto slice_Slice_6 = makeOP<opset8::Gather>({slice_Slice_2, Unsqueeze_23750, 1}, {{"batch_dims", 0}});
        auto Reshape_27408 = makeOP<opset1::Reshape>({slice_Slice_6, {-1, 1, 1, 128}}, {{"special_zero", false}});
        auto mul_Multiply_1 = makeOP<opset1::Multiply>({cat_Concat, Reshape_27408}, {{"auto_broadcast", "numpy"}});
        auto add_Add = makeOP<opset1::Add>({mul_Multiply, mul_Multiply_1}, {{"auto_broadcast", "numpy"}});

        auto slice_Slice_10 = makeConst(element::f32, ov::Shape({1, 32767, 1, 1}), {1});
        auto view_Reshape_1 = makeOP<opset1::Reshape>({qkv_proj->output(1), {0, 0, 32, 128}}, {{"special_zero", true}});
        auto slice_Slice_11 = makeOP<opset8::Slice>({view_Reshape_1, {0}, {128}, {1}, {3}});
        auto mul_Multiply_2 = makeOP<opset1::Multiply>({slice_Slice_11, Reshape_27400}, {{"auto_broadcast", "numpy"}});
        auto reshape_Reshape_1 = makeOP<opset1::Reshape>({slice_Slice_11, {0, 0, 32, 2, 64}}, {{"special_zero", true}});
        auto ListUnpack_Split_1 = makeOP<opset1::Split>({reshape_Reshape_1, -2}, {{"num_splits", 2}});
        auto Multiply_54139 =
            makeOP<opset1::Multiply>({ListUnpack_Split_1->output(1), -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto ListUnpack_Squeeze_0_1 =
            makeOP<opset1::Reshape>({Multiply_54139, {-1, 1, 32, 64}}, {{"special_zero", false}});
        auto ListUnpack_Squeeze_1 =
            makeOP<opset1::Reshape>({ListUnpack_Split_1->output(0), {-1, 1, 32, 64}}, {{"special_zero", false}});
        auto cat_Concat_2 = makeOP<opset1::Concat>({ListUnpack_Squeeze_0_1, ListUnpack_Squeeze_1}, {{"axis", -1}});
        auto mul_Multiply_3 = makeOP<opset1::Multiply>({cat_Concat_2, Reshape_27408}, {{"auto_broadcast", "numpy"}});
        auto add_Add_1 = makeOP<opset1::Add>({mul_Multiply_2, mul_Multiply_3}, {{"auto_broadcast", "numpy"}});
        model = std::make_shared<ov::Model>(ov::NodeVector{add_Add_1}, ov::ParameterVector{position_ids, qkv});
    }

    manager.register_pass<ov::pass::RoPEFusion>(false);

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 4096 * 3});
        auto rotary_emp_sin = makeConst(element::f32, ov::Shape({1, 4096, 1, 128}), {1});
        auto rotary_emp_cos = makeConst(element::f32, ov::Shape({1, 4096, 1, 128}), {1});
        auto position_ids = std::make_shared<opset1::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        auto Convert_50535 = makeOP<opset1::Convert>({position_ids}, {{"destination_type", "i32"}});
        auto Unsqueeze_23750 = makeOP<opset1::Reshape>({Convert_50535, {-1, 1}}, {{"special_zero", false}});
        auto rope = makeOP<ov::op::internal::RoPE>({input, rotary_emp_sin, rotary_emp_cos, Unsqueeze_23750},
                                                   {{"config.slice_start", 4096},
                                                    {"config.slice_stop", 8192},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", 128},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", true},
                                                    {"config.head_cnt", 32},
                                                    {"config.head_size", 128},
                                                    {"config.gather_position_arg_id", 3}});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, position_ids});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_GPTJ_PagedAttention) {
    disable_rt_info_check();
    const int batch = -1;
    const int num_heads = 16;
    const int ndims = 256;
    const int rotary_ndims = 64;
    using namespace ov;
    {
        std::vector<int32_t> rpi_idx(rotary_ndims);
        for (int i = 0, index = 0; i < rotary_ndims; i += 2, index++) {
            rpi_idx[i] = index;
            rpi_idx[i + 1] = index;
        }
        auto repeat_interleave_index = makeConst(ov::element::i32, ov::Shape({rotary_ndims}), rpi_idx);

        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, 1, num_heads, ndims});
        auto aten_gather_GatherElements =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, rotary_ndims});

        auto prim_ListUnpack_VariadicSplit =
            makeOP<opset1::VariadicSplit>({aten_gather_GatherElements, -1, {rotary_ndims / 2, -1}});
        auto aten_unsqueeze_Unsqueeze_1 =
            makeOP<opset1::Reshape>({prim_ListUnpack_VariadicSplit->output(1), {-1, 1, 1, rotary_ndims / 2}},
                                    {{"special_zero", false}});
        auto aten_repeat_interleave_Gather_1 =
            makeOP<opset8::Gather>({aten_unsqueeze_Unsqueeze_1, repeat_interleave_index, 3}, {{"batch_dims", 0}});

        auto aten_unsqueeze_Unsqueeze_2 =
            makeOP<opset1::Reshape>({prim_ListUnpack_VariadicSplit->output(0), {-1, 1, 1, rotary_ndims / 2}},
                                    {{"special_zero", false}});
        auto aten_repeat_interleave_Gather_3 =
            makeOP<opset8::Gather>({aten_unsqueeze_Unsqueeze_2, repeat_interleave_index, 3}, {{"batch_dims", 0}});

        auto VariadicSplit_32371 = makeOP<opset1::VariadicSplit>({input, 3, {rotary_ndims, ndims - rotary_ndims}});
        auto aten_mul_Multiply =
            makeOP<opset1::Multiply>({VariadicSplit_32371->output(0), aten_repeat_interleave_Gather_1},
                                     {{"auto_broadcast", "numpy"}});
        auto aten_slice_Slice_10 = makeOP<opset8::Slice>({VariadicSplit_32371->output(0), {1}, {INT_MAX}, {2}, {3}});
        auto Constant_65243 = makeConst(element::f32, ov::Shape({1, 1, 1, 1}), {-1.000000f});
        auto aten_neg_Multiply =
            makeOP<opset1::Multiply>({aten_slice_Slice_10, Constant_65243}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_28998 = makeOP<opset1::Reshape>({aten_neg_Multiply, {-1, 1, num_heads, rotary_ndims / 2, 1}},
                                                       {{"special_zero", false}});
        auto aten_slice_Slice_14 = makeOP<opset8::Slice>({VariadicSplit_32371->output(0), {0}, {INT_MAX}, {2}, {3}});
        auto Unsqueeze_28999 = makeOP<opset1::Reshape>({aten_slice_Slice_14, {-1, 1, num_heads, rotary_ndims / 2, 1}},
                                                       {{"special_zero", false}});
        auto aten_stack = makeOP<opset1::Concat>({Unsqueeze_28998, Unsqueeze_28999}, {{"axis", -1}});
        auto aten_flatten_Reshape =
            makeOP<opset1::Reshape>({aten_stack, {0, 0, num_heads, rotary_ndims}}, {{"special_zero", true}});
        auto aten_mul_Multiply_1 = makeOP<opset1::Multiply>({aten_flatten_Reshape, aten_repeat_interleave_Gather_3},
                                                            {{"auto_broadcast", "numpy"}});
        auto aten_add_Add =
            makeOP<opset1::Add>({aten_mul_Multiply, aten_mul_Multiply_1}, {{"auto_broadcast", "numpy"}});
        auto aten_cat_Concat_1 = makeOP<opset1::Concat>({aten_add_Add, VariadicSplit_32371->output(1)}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::NodeVector{aten_cat_Concat_1},
                                            ov::ParameterVector{input, aten_gather_GatherElements});
    }
    manager.register_pass<ov::pass::RoPEFusion>(false);
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{batch, 1, num_heads, ndims});
        auto aten_gather_GatherElements =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 64});
        auto rope = makeOP<ov::op::internal::RoPE>({input, aten_gather_GatherElements, aten_gather_GatherElements},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 0},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", true},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, aten_gather_GatherElements});
    }
}

std::vector<int> MOCK_VALUE = {1};

TEST_F(TransformationTestsF, ConvertToROPE_chatGLM4_PagedAttention) {
    {
        disable_rt_info_check();
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{DYN, 1, 4608});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{DYN, 1, 32, 2});

        auto aten_slice_Slice_2 = makeOP<ov::opset8::Slice>({input1, {0}, {1}, {1}, {1}});
        auto aten_view_Reshape =
            makeOP<ov::opset1::Reshape>({aten_slice_Slice_2, {-1, 1, 1, 32, 2}}, {{"special_zero", false}});
        auto aten_select_Gather_1 = makeOP<ov::opset8::Gather>({aten_view_Reshape, 0, -1}, {{"batch_dims", 0}});
        auto aten_select_Gather_3 = makeOP<ov::opset8::Gather>({aten_view_Reshape, 1, -1}, {{"batch_dims", 0}});

        auto module_transformer_encoder_layers_0_self_attention_prim_ListUnpack =
            makeOP<ov::opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
        auto module_transformer_encoder_layers_0_self_attention_aten_transpose_Transpose = makeOP<ov::opset1::Reshape>(
            {module_transformer_encoder_layers_0_self_attention_prim_ListUnpack->output(0), {-1, 32, 1, 128}},
            {{"special_zero", false}});
        auto VariadicSplit_41226 = makeOP<ov::opset1::VariadicSplit>(
            {module_transformer_encoder_layers_0_self_attention_aten_transpose_Transpose, 3, {64, 64}});
        auto aten_reshape_Reshape =
            makeOP<ov::opset1::Reshape>({VariadicSplit_41226->output(0), {0, 32, 0, 32, 2}}, {{"special_zero", true}});
        auto aten_select_Gather = makeOP<ov::opset8::Gather>({aten_reshape_Reshape, 0, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_1 =
            makeOP<ov::opset1::Multiply>({aten_select_Gather, aten_select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto aten_select_Gather_2 = makeOP<ov::opset8::Gather>({aten_reshape_Reshape, 1, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_2 =
            makeOP<ov::opset1::Multiply>({aten_select_Gather_2, aten_select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto Multiply_45963 =
            makeOP<ov::opset1::Multiply>({aten_mul_Multiply_2, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto aten_sub_Subtract =
            makeOP<ov::opset1::Add>({aten_mul_Multiply_1, Multiply_45963}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_81695 =
            makeOP<ov::opset1::Reshape>({aten_sub_Subtract, {-1, 32, 1, 32, 1}}, {{"special_zero", false}});
        auto aten_mul_Multiply_3 =
            makeOP<ov::opset1::Multiply>({aten_select_Gather_2, aten_select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto aten_mul_Multiply_4 =
            makeOP<ov::opset1::Multiply>({aten_select_Gather, aten_select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto aten_add_Add =
            makeOP<ov::opset1::Add>({aten_mul_Multiply_3, aten_mul_Multiply_4}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_81696 =
            makeOP<ov::opset1::Reshape>({aten_add_Add, {-1, 32, 1, 32, 1}}, {{"special_zero", false}});
        auto aten_stack = makeOP<ov::opset1::Concat>({Unsqueeze_81695, Unsqueeze_81696}, {{"axis", -1}});
        auto aten_flatten_Reshape = makeOP<ov::opset1::Reshape>({aten_stack, {0, 32, 0, 64}}, {{"special_zero", true}});
        auto aten_cat_Concat =
            makeOP<ov::opset1::Concat>({aten_flatten_Reshape, VariadicSplit_41226->output(1)}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::NodeVector{aten_cat_Concat}, ov::ParameterVector{input, input1});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{DYN, 1, 4608});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{DYN, 1, 32, 2});

        auto rope = makeOP<ov::op::internal::RoPE>({input, input1, input1},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 4096},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", 64},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", true},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 32},
                                                    {"config.head_size", 128},
                                                    {"config.gather_position_arg_id", 0}});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, input1});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGLM4_PagedAttention_GPU) {
    using namespace ov;
    {
        disable_rt_info_check();
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::PartialShape{DYN, 1, 4608});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::PartialShape{DYN, 1, 32, 2});

        auto ListUnpack = makeOP<opset1::VariadicSplit>({input0, -1, {4096, 256, 256}});
        auto aten_transpose =
            makeOP<opset1::Reshape>({ListUnpack->output(0), {-1, 32, 1, 128}}, {{"special_zero", false}});
        auto aten_slice_Slice =
            makeOP<opset1::StridedSlice>({aten_transpose, {0, 0, 0, 0}, {0, 0, 0, 64}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto aten_reshape_Reshape =
            makeOP<opset1::Reshape>({aten_slice_Slice, {0, 32, 0, 32, 2}}, {{"special_zero", true}});
        auto aten_select_Gather = makeOP<opset8::Gather>({aten_reshape_Reshape, 0, -1}, {{"batch_dims", 0}});
        auto aten_slice_Slice_2 = makeOP<opset1::StridedSlice>({input1, {0, 0}, {0, 1}, {1, 1}},
                                                               {{"begin_mask", {1, 0}},
                                                                {"end_mask", {1, 0}},
                                                                {"new_axis_mask", {}},
                                                                {"shrink_axis_mask", {}},
                                                                {"ellipsis_mask", {}}});
        auto aten_view_Reshape =
            makeOP<opset1::Reshape>({aten_slice_Slice_2, {-1, 1, 1, 32, 2}}, {{"special_zero", false}});
        auto aten_select_Gather_1 = makeOP<opset8::Gather>({aten_view_Reshape, 0, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_1 =
            makeOP<opset1::Multiply>({aten_select_Gather, aten_select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto aten_select_Gather_2 = makeOP<opset8::Gather>({aten_reshape_Reshape, 1, -1}, {{"batch_dims", 0}});
        auto aten_select_Gather_3 = makeOP<opset8::Gather>({aten_view_Reshape, 1, -1}, {{"batch_dims", 0}});
        auto aten_mul_Multiply_2 =
            makeOP<opset1::Multiply>({aten_select_Gather_2, aten_select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto Constant_56849 = makeConst(element::f16, ov::Shape({}), {-1});
        auto Multiply_56850 =
            makeOP<opset1::Multiply>({aten_mul_Multiply_2, Constant_56849}, {{"auto_broadcast", "numpy"}});
        auto aten_sub_Subtract =
            makeOP<opset1::Add>({aten_mul_Multiply_1, Multiply_56850}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_81695 =
            makeOP<opset1::Reshape>({aten_sub_Subtract, {-1, 32, 1, 32, 1}}, {{"special_zero", false}});
        auto aten_mul_Multiply_3 =
            makeOP<opset1::Multiply>({aten_select_Gather_2, aten_select_Gather_1}, {{"auto_broadcast", "numpy"}});
        auto aten_mul_Multiply_4 =
            makeOP<opset1::Multiply>({aten_select_Gather, aten_select_Gather_3}, {{"auto_broadcast", "numpy"}});
        auto aten_add_Add =
            makeOP<opset1::Add>({aten_mul_Multiply_3, aten_mul_Multiply_4}, {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_81696 = makeOP<opset1::Reshape>({aten_add_Add, {-1, 32, 1, 32, 1}}, {{"special_zero", false}});
        auto aten_stack = makeOP<opset1::Concat>({Unsqueeze_81695, Unsqueeze_81696}, {{"axis", -1}});
        auto aten_flatten_Reshape = makeOP<opset1::Reshape>({aten_stack, {0, 32, 0, 64}}, {{"special_zero", true}});
        auto aten_slice_Slice_3 =
            makeOP<opset1::StridedSlice>({aten_transpose, {0, 0, 0, 64}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto aten_cat_Concat = makeOP<opset1::Concat>({aten_flatten_Reshape, aten_slice_Slice_3}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::NodeVector{aten_cat_Concat}, ov::ParameterVector{input0, input1});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::PartialShape{DYN, 1, 4608});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::PartialShape{DYN, 1, 32, 2});

        auto rope = makeOP<ov::op::internal::RoPE>({input, input1, input1},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 4096},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", 64},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", true},
                                                    {"config.is_qwen", false},
                                                    {"config.head_cnt", 32},
                                                    {"config.head_size", 128},
                                                    {"config.gather_position_arg_id", 0}});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, input1});
    }
}