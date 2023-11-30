// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <transformations/cpu_opset/common/op/rope.hpp>
#include <transformations/cpu_opset/common/pass/rope_fusion.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "utils/gen_pattern.hpp"
#include "utils/print_model.hpp"

using namespace testing;
using namespace ov::intel_cpu;
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
    manager.register_pass<RoPEFusion>();

    {
        auto hidden_states =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, num_head, ndims});
        auto param_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});
        auto param_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});
        auto add_Add = makeOP<RoPENode>({hidden_states, param_cos, param_sin},
                                        {{"config.slice_start", 0},
                                         {"config.slice_stop", 0},
                                         {"config.input_trans0213", true},
                                         {"config.is_interleaved", false},
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
    manager.register_pass<RoPEFusion>();

    {
        auto hidden_states =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, num_head, ndims});
        auto seq_len = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
        auto gather_id = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, seq_length});
        auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);

        auto add_Add = makeOP<RoPENode>({hidden_states, cos_sin_cache[0], cos_sin_cache[1], gather_id},
                                        {{"config.slice_start", 0},
                                         {"config.slice_stop", 0},
                                         {"config.input_trans0213", true},
                                         {"config.is_interleaved", false},
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
    manager.register_pass<RoPEFusion>();
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims * 3});
        auto param_cos =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_len, rotary_ndims});
        auto param_sin =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_len, rotary_ndims});
        auto rope = makeOP<RoPENode>({input, param_cos, param_sin},
                                     {{"config.slice_start", 0},
                                      {"config.slice_stop", ndims},
                                      {"config.input_trans0213", true},
                                      {"config.is_interleaved", false},
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
    manager.register_pass<RoPEFusion>();
    {
        auto cos_sin = makeCosSinCache(max_position_embeddings, rotary_ndims);
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims * 3});
        auto gather_idx =
            std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, 1, seq_len, rotary_ndims});
        auto batch_limit = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});

        auto rope = makeOP<RoPENode>({input, cos_sin[0], cos_sin[1], gather_idx},
                                     {{"config.slice_start", 0},
                                      {"config.slice_stop", ndims},
                                      {"config.input_trans0213", true},
                                      {"config.is_interleaved", false},
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
    manager.register_pass<RoPEFusion>();
    {
        auto input =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, num_heads, ndims});
        auto cos_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, seq_len, rotary_ndims});
        auto rope = makeOP<RoPENode>({input, cos_sin, cos_sin},
                                     {{"config.slice_start", 0},
                                      {"config.slice_stop", 0},
                                      {"config.input_trans0213", false},
                                      {"config.is_interleaved", true},
                                      {"config.rotary_ndims", rotary_ndims},
                                      {"config.gather_position_arg_id", 0}});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rope}, ov::ParameterVector{input, cos_sin});
    }
}