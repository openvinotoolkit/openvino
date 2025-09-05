// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset3_decl.hpp"
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

    return std::make_shared<ov::Model>(ov::OutputVector{add_Add}, parameters);
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
                                                       {"config.use_rope_cache", false},
                                                       {"config.head_cnt", 0},
                                                       {"config.head_size", 0},
                                                       {"config.rotary_ndims", static_cast<int>(ndims)},
                                                       {"config.gather_position_arg_id", 0}});

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add_Add},
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
                                                       {"config.use_rope_cache", false},
                                                       {"config.head_cnt", 0},
                                                       {"config.head_size", 0},
                                                       {"config.rotary_ndims", static_cast<int>(ndims)},
                                                       {"config.gather_position_arg_id", 3}});

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add_Add},
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

    return std::make_shared<ov::Model>(ov::OutputVector{cat_Concat_458}, parameters);
}

static std::shared_ptr<ov::Model> buildROPE_VIT(const int seq_length,
                                                const int num_heads,
                                                const int rotary_ndims,
                                                std::string split_op_type) {
    auto seq_length_s = static_cast<size_t>(seq_length);
    auto rotary_ndims_s = static_cast<size_t>(rotary_ndims);
    auto num_heads_s = static_cast<size_t>(num_heads);
    auto input =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_length_s, num_heads_s, rotary_ndims_s});
    auto Constant_396096 = makeConst(ov::element::f32, ov::Shape({1, 1, 1}), {-1.000000f});

    auto param_cos =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_length_s, 1, rotary_ndims_s});
    auto param_sin =
        std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_length_s, 1, rotary_ndims_s});
    ov::Output<ov::Node> cat_Concat;
    if (split_op_type == "VariadicSplit") {
        auto split = makeOP<ov::opset1::VariadicSplit>({input, {2}, {rotary_ndims / 2, rotary_ndims / 2}});
        auto neg_Multiply =
            makeOP<ov::opset1::Multiply>({split->output(1), Constant_396096}, {{"auto_broadcast", "numpy"}});
        cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, split->output(0)}, {{"axis", -1}});
    } else if (split_op_type == "Slice") {
        auto slice_right_part = makeOP<ov::opset8::Slice>({input, {rotary_ndims / 2}, {INT_MAX}, {1}, {2}});
        auto slice_left_part = makeOP<ov::opset8::Slice>({input, {0}, {rotary_ndims / 2}, {1}, {2}});
        auto neg_Multiply =
            makeOP<ov::opset1::Multiply>({slice_right_part, Constant_396096}, {{"auto_broadcast", "numpy"}});
        cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_left_part}, {{"axis", -1}});
    } else if (split_op_type == "StridedSlice") {
        auto slice_right_part =
            makeOP<ov::opset1::StridedSlice>({input, {0, 0, rotary_ndims / 2}, {0, 0, INT_MAX}, {1, 1, 1}},
                                             {{"begin_mask", {1, 1, 0}},
                                              {"end_mask", {1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto slice_left_part = makeOP<ov::opset1::StridedSlice>({input, {0, 0, 0}, {0, 0, rotary_ndims / 2}, {1, 1, 1}},
                                                                {{"begin_mask", {1, 1, 0}},
                                                                 {"end_mask", {1, 1, 0}},
                                                                 {"new_axis_mask", {}},
                                                                 {"shrink_axis_mask", {}},
                                                                 {"ellipsis_mask", {}}});
        auto neg_Multiply =
            makeOP<ov::opset1::Multiply>({slice_right_part, Constant_396096}, {{"auto_broadcast", "numpy"}});
        cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_left_part}, {{"axis", -1}});
    } else {
        return nullptr;
    }
    auto mul_sin_Multiply = makeOP<ov::opset1::Multiply>({cat_Concat, param_sin}, {{"auto_broadcast", "numpy"}});
    auto mul_cos_Multiply = makeOP<ov::opset1::Multiply>({input, param_cos}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::opset1::Add>({mul_cos_Multiply, mul_sin_Multiply}, {{"auto_broadcast", "numpy"}});
    ov::ParameterVector parameters = ov::ParameterVector{input, param_cos, param_sin};
    return std::make_shared<ov::Model>(ov::OutputVector{add_Add}, parameters);
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
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, param_cos, param_sin});
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
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 3}});
        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, gather_idx, batch_limit});
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
        model = std::make_shared<ov::Model>(ov::OutputVector{permute_Transpose_828},
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
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, cos_sin});
    }
}

// Parametrized ConvertToROPE_chatGLM tests to check both unpack->output(0) and unpack->output(1)
class ConvertToROPETest : public TransformationTestsF, public ::testing::WithParamInterface<int> {};

TEST_P(ConvertToROPETest, ConvertToROPE_chatGLM) {
    disable_rt_info_check();

    const int batch = 2, seq_len = 7, num_heads = 32, ndims = 128, rotary_ndims = 64, max_pos_length = 2048;
    const int total_size_q = 4096, total_size_k = 256, total_size_v = 256;
    const int total_size = total_size_q + total_size_k + total_size_v;

    int unpack_output_idx = GetParam();

    // Build the original model
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, total_size});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});

        auto unpack = makeOP<ov::opset1::VariadicSplit>({input, -1, {total_size_q, total_size_k, total_size_v}});
        auto reshaped = makeOP<ov::opset1::Reshape>({unpack->output(unpack_output_idx), {0, 0, num_heads, ndims}},
                                                    {{"special_zero", true}});
        auto slice = makeOP<ov::opset1::StridedSlice>({reshaped, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                                      {{"begin_mask", {1, 1, 1, 0}},
                                                       {"end_mask", {1, 1, 1, 0}},
                                                       {"new_axis_mask", {}},
                                                       {"shrink_axis_mask", {}},
                                                       {"ellipsis_mask", {}}});

        auto shape_concat =
            makeOP<ov::opset1::Concat>({seq_length, {-1}, {num_heads}, {rotary_ndims / 2}, {2}}, {{"axis", 0}});
        auto reshaped_slice = makeOP<ov::opset1::Reshape>({slice, shape_concat}, {{"special_zero", false}});

        auto gather_even = makeOP<ov::opset8::Gather>({reshaped_slice, 0, -1}, {{"batch_dims", 0}});
        auto gather_odd = makeOP<ov::opset8::Gather>({reshaped_slice, 1, -1}, {{"batch_dims", 0}});

        auto cache_slice = makeOP<ov::opset1::StridedSlice>({cos_sin_cache, {0}, seq_length, {1}},
                                                            {{"begin_mask", {0}},
                                                             {"end_mask", {0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
        auto cache_shape = makeOP<ov::opset1::Concat>({seq_length, {-1}, {1}, {rotary_ndims / 2}, {2}}, {{"axis", 0}});
        auto cache_reshaped = makeOP<ov::opset1::Reshape>({cache_slice, cache_shape}, {{"special_zero", false}});

        auto gather_cos = makeOP<ov::opset8::Gather>({cache_reshaped, 0, -1}, {{"batch_dims", 0}});
        auto gather_sin = makeOP<ov::opset8::Gather>({cache_reshaped, 1, -1}, {{"batch_dims", 0}});

        auto mul_even_cos = makeOP<ov::opset1::Multiply>({gather_even, gather_cos}, {{"auto_broadcast", "numpy"}});
        auto mul_odd_sin = makeOP<ov::opset1::Multiply>({gather_odd, gather_sin}, {{"auto_broadcast", "numpy"}});
        auto neg_odd_sin = makeOP<ov::opset1::Multiply>({mul_odd_sin, -1.0f}, {{"auto_broadcast", "numpy"}});
        auto add_even = makeOP<ov::opset1::Add>({mul_even_cos, neg_odd_sin}, {{"auto_broadcast", "numpy"}});

        auto unsq_even = makeOP<ov::opset1::Unsqueeze>({add_even, -1});
        auto mul_odd_cos = makeOP<ov::opset1::Multiply>({gather_odd, gather_cos}, {{"auto_broadcast", "numpy"}});
        auto mul_even_sin = makeOP<ov::opset1::Multiply>({gather_even, gather_sin}, {{"auto_broadcast", "numpy"}});
        auto add_odd = makeOP<ov::opset1::Add>({mul_odd_cos, mul_even_sin}, {{"auto_broadcast", "numpy"}});
        auto unsq_odd = makeOP<ov::opset1::Unsqueeze>({add_odd, -1});

        auto stack = makeOP<ov::opset1::Concat>({unsq_even, unsq_odd}, {{"axis", -1}});
        auto shapeof = makeOP<ov::op::TypeRelaxed<ov::opset1::ShapeOf>>(
            {stack},
            {{"type_relax", true}, {"input_data_types", {}}, {"output_data_types", {ov::element::i32}}});
        auto flatten_shape = makeOP<ov::opset1::StridedSlice>({shapeof, {0}, {3}, {1}},
                                                              {{"begin_mask", {0}},
                                                               {"end_mask", {0}},
                                                               {"new_axis_mask", {}},
                                                               {"shrink_axis_mask", {}},
                                                               {"ellipsis_mask", {}}});
        auto flatten_concat = makeOP<ov::opset1::Concat>({flatten_shape, {-1}}, {{"axis", 0}});
        auto flatten = makeOP<ov::opset1::Reshape>({stack, flatten_concat}, {{"special_zero", true}});

        auto slice_rest =
            makeOP<ov::opset1::StridedSlice>({reshaped, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto concat = makeOP<ov::opset1::Concat>({flatten, slice_rest}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::OutputVector{concat},
                                            ov::ParameterVector{input, seq_length, cos_sin_cache});
    }

    manager.register_pass<ov::pass::RoPEFusion>();

    // Build the reference model
    {
        int slice_start = 0, slice_stop = 0;
        if (unpack_output_idx == 0) {
            slice_stop = total_size_q;
        } else {
            slice_start = total_size_q;
            slice_stop = slice_start + total_size_k;
        }

        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, total_size});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin_cache, cos_sin_cache},
                                                   {{"config.slice_start", slice_start},
                                                    {"config.slice_stop", slice_stop},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, seq_length, cos_sin_cache});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_P(ConvertToROPETest, ConvertToROPE_chatGLM_Slice) {
    disable_rt_info_check();

    const int batch = 2, seq_len = 7, num_heads = 32, ndims = 128, rotary_ndims = 64, max_pos_length = 2048;
    const int total_size_q = 4096, total_size_k = 256, total_size_v = 256;
    const int total_size = total_size_q + total_size_k + total_size_v;
    const int unpack_output_idx = GetParam();

    // Build original model
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, total_size});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});

        auto unpack = makeOP<ov::opset1::VariadicSplit>({input, -1, {total_size_q, total_size_k, total_size_v}});
        auto reshaped = makeOP<ov::opset1::Reshape>({unpack->output(unpack_output_idx), {0, 0, num_heads, ndims}},
                                                    {{"special_zero", true}});
        auto split = makeOP<ov::opset1::VariadicSplit>({reshaped, 3, {rotary_ndims, -1}});
        auto rope_part = makeOP<ov::opset1::Reshape>({split->output(0), {0, 0, num_heads, rotary_ndims / 2, 2}},
                                                     {{"special_zero", true}});
        auto x_even = makeOP<ov::opset8::Gather>({rope_part, 0, -1}, {{"batch_dims", 0}});
        auto x_odd = makeOP<ov::opset8::Gather>({rope_part, 1, -1}, {{"batch_dims", 0}});

        auto cache_slice = makeOP<ov::opset8::Slice>({cos_sin_cache, {0}, seq_length, {1}, {0}});
        auto cache_shape = makeOP<ov::opset1::Concat>({seq_length, {-1}, {1}, {rotary_ndims / 2}, {2}}, {{"axis", 0}});
        auto cache_reshaped = makeOP<ov::opset1::Reshape>({cache_slice, cache_shape}, {{"special_zero", false}});
        auto cos_tab = makeOP<ov::opset8::Gather>({cache_reshaped, 0, -1}, {{"batch_dims", 0}});
        auto sin_tab = makeOP<ov::opset8::Gather>({cache_reshaped, 1, -1}, {{"batch_dims", 0}});

        auto even_cos = makeOP<ov::opset1::Multiply>({x_even, cos_tab}, {{"auto_broadcast", "numpy"}});
        auto odd_sin = makeOP<ov::opset1::Multiply>({x_odd, sin_tab}, {{"auto_broadcast", "numpy"}});
        auto neg_odd_sin = makeOP<ov::opset1::Multiply>({odd_sin, -1.0f}, {{"auto_broadcast", "numpy"}});
        auto add_even = makeOP<ov::opset1::Add>({even_cos, neg_odd_sin}, {{"auto_broadcast", "numpy"}});
        auto unsq_even = makeOP<ov::opset1::Unsqueeze>({add_even, -1});

        auto odd_cos = makeOP<ov::opset1::Multiply>({x_odd, cos_tab}, {{"auto_broadcast", "numpy"}});
        auto even_sin = makeOP<ov::opset1::Multiply>({x_even, sin_tab}, {{"auto_broadcast", "numpy"}});
        auto add_odd = makeOP<ov::opset1::Add>({odd_cos, even_sin}, {{"auto_broadcast", "numpy"}});
        auto unsq_odd = makeOP<ov::opset1::Unsqueeze>({add_odd, -1});

        auto stack = makeOP<ov::opset1::Concat>({unsq_even, unsq_odd}, {{"axis", -1}});
        auto flatten = makeOP<ov::opset1::Reshape>({stack, {0, 0, num_heads, rotary_ndims}}, {{"special_zero", true}});
        auto concat = makeOP<ov::opset1::Concat>({flatten, split->output(1)}, {{"axis", -1}});

        model = std::make_shared<ov::Model>(ov::OutputVector{concat},
                                            ov::ParameterVector{input, seq_length, cos_sin_cache});
    }

    manager.register_pass<ov::pass::RoPEFusion>();

    // Build reference model
    {
        int slice_start = 0, slice_stop = 0;
        if (unpack_output_idx == 0) {
            slice_stop = total_size_q;
        } else {
            slice_start = total_size_q;
            slice_stop = slice_start + total_size_k;
        }

        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, batch, total_size});
        auto seq_length = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto cos_sin_cache =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                    ov::Shape{max_pos_length, batch, rotary_ndims / 2, 2});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos_sin_cache, cos_sin_cache},
                                                   {{"config.slice_start", slice_start},
                                                    {"config.slice_stop", slice_stop},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", false},
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, seq_length, cos_sin_cache});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(TransformationTestsF, ConvertToROPETest, ::testing::ValuesIn({0, 1}));

class ConvertToROPETestVIT : public TransformationTestsF, public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
        const auto& split_op_type = obj.param;
        std::ostringstream result;
        result << "split_op_type=" << split_op_type;
        return result.str();
    }
};
TEST_P(ConvertToROPETestVIT, ConvertToROPE_qwen) {
    disable_rt_info_check();
    const int seq_len = 16;
    const int num_heads = 32;
    const int rotary_ndims = 80;
    const std::string split_op_type = GetParam();
    model = buildROPE_VIT(seq_len, num_heads, rotary_ndims, split_op_type);
    ASSERT_TRUE(model != nullptr);
    manager.register_pass<ov::pass::RoPEFusionVIT3D>();
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, num_heads, rotary_ndims});
        auto param_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, 1, rotary_ndims});
        auto param_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{seq_len, 1, rotary_ndims});
        auto rope = makeOP<ov::op::internal::RoPE>({input, param_cos, param_sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 0},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.support_3d_rope", true},
                                                    {"config.is_qwen", false},
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, param_cos, param_sin});
    }
}

const std::vector<std::string> vit_param = {"VariadicSplit", "Slice", "StridedSlice"};
INSTANTIATE_TEST_SUITE_P(TransformationTestsF,
                         ConvertToROPETestVIT,
                         ::testing::ValuesIn(vit_param),
                         ConvertToROPETestVIT::getTestCaseName);

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

        model = std::make_shared<ov::Model>(ov::OutputVector{permute_Transpose},
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
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, cos_sin});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGLM_2d_rope) {
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
        model = std::make_shared<ov::Model>(ov::OutputVector{cat_Concat_425},
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
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope},
                                                ov::ParameterVector{input, cos_sin_cache, position_ids});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGLM_nano_2d_rope) {
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
        model = std::make_shared<ov::Model>(ov::OutputVector{flatten_Reshape_421},
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
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope},
                                                ov::ParameterVector{input, cos_sin_cache, position_ids});
    }
}

TEST_F(TransformationTestsF, ConvertToROPE_chatGLMHF_2d_rope) {
    disable_rt_info_check();
    const int seq_len = 7;
    const int num_heads = 32;
    const int ndims = 128;
    const int rotary_ndims = 64;
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, 1, 4096});
        auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                           ov::PartialShape{seq_len, 1, 1, (rotary_ndims / 2)});
        auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                           ov::PartialShape{seq_len, 1, 1, (rotary_ndims / 2)});

        auto transpose = makeOP<ov::opset1::Reshape>({input, {-1, num_heads, 1, ndims}}, {{"special_zero", false}});
        auto slice_1 =
            makeOP<ov::opset1::StridedSlice>({transpose, {0, 0, 0, 0}, {0, 0, 0, rotary_ndims}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});

        std::vector<int32_t> rpi_idx(rotary_ndims);
        for (int i = 0, index = 0; i < rotary_ndims; i += 2, index++) {
            rpi_idx[i] = index;
            rpi_idx[i + 1] = index;
        }
        auto repeat_interleave_index = makeConst(ov::element::i32, ov::Shape({rotary_ndims}), rpi_idx);
        auto repeat_interleave_cos =
            makeOP<ov::opset8::Gather>({cos, repeat_interleave_index, -1}, {{"batch_dims", 0}});
        auto repeat_interleave_sin =
            makeOP<ov::opset8::Gather>({cos, repeat_interleave_index, -1}, {{"batch_dims", 0}});

        auto multiply = makeOP<ov::opset1::Multiply>({slice_1, repeat_interleave_cos}, {{"auto_broadcast", "numpy"}});
        auto slice_2 = makeOP<ov::opset1::StridedSlice>({slice_1, {0, 0, 0, 1}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                                        {{"begin_mask", {1, 1, 1, 0}},
                                                         {"end_mask", {1, 1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
        auto neg = makeOP<ov::opset1::Multiply>({slice_2, -1.000000f}, {{"auto_broadcast", "numpy"}});
        auto unsqueeze_1 =
            makeOP<ov::opset1::Reshape>({neg, {-1, num_heads, 1, (rotary_ndims / 2), 1}}, {{"special_zero", false}});
        auto slice_3 = makeOP<ov::opset1::StridedSlice>({slice_1, {0, 0, 0, 0}, {0, 0, 0, INT_MAX}, {1, 1, 1, 2}},
                                                        {{"begin_mask", {1, 1, 1, 0}},
                                                         {"end_mask", {1, 1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
        auto unsqueeze_2 = makeOP<ov::opset1::Reshape>({slice_3, {-1, num_heads, 1, (rotary_ndims / 2), 1}},
                                                       {{"special_zero", false}});
        auto stack = makeOP<ov::opset1::Concat>({unsqueeze_1, unsqueeze_2}, {{"axis", -1}});
        auto flatten = makeOP<ov::opset1::Reshape>({stack, {0, num_heads, 0, rotary_ndims}}, {{"special_zero", true}});
        auto multiply_1 = makeOP<ov::opset1::Multiply>({flatten, repeat_interleave_sin}, {{"auto_broadcast", "numpy"}});
        auto add = makeOP<ov::opset1::Add>({multiply, multiply_1}, {{"auto_broadcast", "numpy"}});

        auto slice_5 =
            makeOP<ov::opset1::StridedSlice>({transpose, {0, 0, 0, rotary_ndims}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                             {{"begin_mask", {1, 1, 1, 0}},
                                              {"end_mask", {1, 1, 1, 0}},
                                              {"new_axis_mask", {}},
                                              {"shrink_axis_mask", {}},
                                              {"ellipsis_mask", {}}});
        auto concat = makeOP<ov::opset1::Concat>({add, slice_5}, {{"axis", -1}});
        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{input, cos, sin});
    }
    manager.register_pass<ov::pass::RoPEFusion>(true);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{seq_len, 1, 4096});
        auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                           ov::PartialShape{seq_len, 1, 1, (rotary_ndims / 2)});
        auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32,
                                                           ov::PartialShape{seq_len, 1, 1, (rotary_ndims / 2)});
        auto rope = makeOP<ov::op::internal::RoPE>({input, cos, sin},
                                                   {{"config.slice_start", 0},
                                                    {"config.slice_stop", 0},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", rotary_ndims},
                                                    {"config.is_chatglm", true},
                                                    {"config.support_2d_rope", true},
                                                    {"config.is_qwen", false},
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, cos, sin});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
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

        model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x, t_cos, t_sin});
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
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{x, t_cos, t_sin});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x, t_cos, t_sin});
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
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{x, t_cos, t_sin});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x, t_cos, t_sin});
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
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{x, t_cos, t_sin});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{aten_cat_Concat_55}, ov::ParameterVector{input, cos_sin});
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
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", num_heads},
                                                    {"config.head_size", ndims},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, gather_cos_sin});
    }
}

TEST_P(ConvertToROPETest, ConvertToROPE_Qwen_PagedAttention) {
    using namespace ov;

    constexpr int head_cnt = 32, head_size = 128;
    int output_idx = GetParam();

    {
        // Parameters
        auto position_ids = std::make_shared<opset1::Parameter>(element::i64, PartialShape{-1, -1});
        auto qkv = std::make_shared<opset1::Parameter>(element::f32, PartialShape{-1, 1, 3 * head_cnt * head_size});

        // Split QKV and reshape to [batch, 1, head_cnt, head_size]
        auto qkv_proj = makeOP<opset1::VariadicSplit>({qkv, 2, {head_cnt * head_size, head_cnt * head_size, -1}});
        auto view = makeOP<opset1::Reshape>({qkv_proj->output(output_idx), {0, 0, head_cnt, head_size}},
                                            {{"special_zero", true}});

        // Slice out rotary dims
        auto slice = makeOP<opset8::Slice>({view, {0}, {128}, {1}, {3}});

        // Prepare rotary embedding table and gather by position
        auto rotary_emp = makeConst(element::f32, {1, 4096, 1, 128}, {1});
        auto pos_i32 = makeOP<opset1::Convert>({position_ids}, {{"destination_type", "i32"}});
        auto pos_reshaped = makeOP<opset1::Reshape>({pos_i32, {-1, 1}}, {{"special_zero", false}});
        auto gathered = makeOP<opset8::Gather>({rotary_emp, pos_reshaped, 1}, {{"batch_dims", 0}});
        auto gathered_reshape = makeOP<opset1::Reshape>({gathered, {-1, 1, 1, 128}}, {{"special_zero", false}});

        // Elementwise multiply
        auto mul = makeOP<opset1::Multiply>({slice, gathered_reshape}, {{"auto_broadcast", "numpy"}});

        // Interleave/stack for rotary
        auto reshaped = makeOP<opset1::Reshape>({slice, {0, 0, 32, 2, 64}}, {{"special_zero", true}});
        auto split = makeOP<opset1::Split>({reshaped, -2}, {{"num_splits", 2}});
        auto neg = makeOP<opset1::Multiply>({split->output(1), -1.0f}, {{"auto_broadcast", "numpy"}});
        auto squeeze0 = makeOP<opset1::Reshape>({neg, {-1, 1, 32, 64}}, {{"special_zero", false}});
        auto squeeze1 = makeOP<opset1::Reshape>({split->output(0), {-1, 1, 32, 64}}, {{"special_zero", false}});
        auto cat = makeOP<opset1::Concat>({squeeze0, squeeze1}, {{"axis", -1}});

        // Second rotary embedding gather and multiply
        auto rotary_emp2 = makeConst(element::f32, {1, 4096, 1, 128}, {1});
        auto gathered2 = makeOP<opset8::Gather>({rotary_emp2, pos_reshaped, 1}, {{"batch_dims", 0}});
        auto gathered2_reshape = makeOP<opset1::Reshape>({gathered2, {-1, 1, 1, 128}}, {{"special_zero", false}});
        auto mul2 = makeOP<opset1::Multiply>({cat, gathered2_reshape}, {{"auto_broadcast", "numpy"}});

        // Final add
        auto add = makeOP<opset1::Add>({mul, mul2}, {{"auto_broadcast", "numpy"}});

        model = std::make_shared<Model>(OutputVector{add}, ParameterVector{position_ids, qkv});
    }

    manager.register_pass<ov::pass::RoPEFusion>(false);

    {
        int slice_start = output_idx == 0 ? 0 : head_cnt * head_size;
        int slice_stop = slice_start + head_cnt * head_size;

        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape{-1, 1, 4096 * 3});
        auto rotary_emp_sin = makeConst(element::f32, {1, 4096, 1, 128}, {1});
        auto rotary_emp_cos = makeConst(element::f32, {1, 4096, 1, 128}, {1});
        auto position_ids = std::make_shared<opset1::Parameter>(element::i64, PartialShape{-1, -1});
        auto pos_i32 = makeOP<opset1::Convert>({position_ids}, {{"destination_type", "i32"}});
        auto pos_reshaped = makeOP<opset1::Reshape>({pos_i32, {-1, 1}}, {{"special_zero", false}});
        auto rope = makeOP<ov::op::internal::RoPE>({input, rotary_emp_sin, rotary_emp_cos, pos_reshaped},
                                                   {{"config.slice_start", slice_start},
                                                    {"config.slice_stop", slice_stop},
                                                    {"config.input_trans0213", false},
                                                    {"config.output_trans0213", false},
                                                    {"config.is_interleaved", false},
                                                    {"config.rotary_ndims", 128},
                                                    {"config.is_chatglm", false},
                                                    {"config.support_2d_rope", false},
                                                    {"config.is_qwen", true},
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", head_cnt},
                                                    {"config.head_size", head_size},
                                                    {"config.gather_position_arg_id", 3}});
        model_ref = std::make_shared<Model>(OutputVector{rope}, ParameterVector{input, position_ids});
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
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

        model = std::make_shared<ov::Model>(ov::OutputVector{aten_cat_Concat_1},
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
                                                    {"config.use_rope_cache", false},
                                                    {"config.head_cnt", 0},
                                                    {"config.head_size", 0},
                                                    {"config.gather_position_arg_id", 0}});
        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, aten_gather_GatherElements});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{aten_cat_Concat}, ov::ParameterVector{input, input1});
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
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", 32},
                                                    {"config.head_size", 128},
                                                    {"config.gather_position_arg_id", 0}});

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, input1});
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

        model = std::make_shared<ov::Model>(ov::OutputVector{aten_cat_Concat}, ov::ParameterVector{input0, input1});
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
                                                    {"config.use_rope_cache", true},
                                                    {"config.head_cnt", 32},
                                                    {"config.head_size", 128},
                                                    {"config.gather_position_arg_id", 0}});

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, input1});
    }
}
