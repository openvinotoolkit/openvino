// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/rotary_pos_emb.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;
using namespace ov;

namespace ov {
namespace test {

ov::OutputVector RoPETestLlama2StridedSlice::makeCosSinCache(int max_position_embeddings, int rotary_ndims) {
    std::vector<float> lut_sin(max_position_embeddings * rotary_ndims, 0.0f);
    std::vector<float> lut_cos(max_position_embeddings * rotary_ndims, 0.0f);

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (int i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
        auto xita_i = 1.0 / std::pow(10000.0, static_cast<double>(i) / rotary_ndims);
        float* psin = lut_sin.data();
        float* pcos = lut_cos.data();
        for (int m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
            auto vsin = std::sin(xita_i * m);
            auto vcos = std::cos(xita_i * m);
            pcos[k] = pcos[k + rotary_ndims / 2] = vcos;
            psin[k] = psin[k + rotary_ndims / 2] = vsin;
        }
    }
    auto shape = ov::Shape({1, 1, static_cast<size_t>(max_position_embeddings), static_cast<size_t>(rotary_ndims)});
    auto Cos = makeConst(ov::element::f32, shape, lut_cos);
    auto Sin = makeConst(ov::element::f32, shape, lut_sin);
    return {Cos, Sin};
}

std::shared_ptr<ov::Model> RoPETestLlama2StridedSlice::buildROPE_Llama2(int batch,
                                                                        int seq_length,
                                                                        int max_position_embeddings,
                                                                        int num_head,
                                                                        int ndims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{batch, -1, num_head, ndims});
    auto pos_id_end = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{});
    auto pos_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{1, -1});

    auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);
    auto Constant582 = cos_sin_cache[0];
    auto Constant585 = cos_sin_cache[1];

    // concat KV length
    auto transpose_Transpose = makeOP<ov::op::v1::Transpose>({input, {0, 2, 1, 3}});
    auto slice_Unsqueeze_426 = makeOP<ov::op::v0::Unsqueeze>({pos_id_end, 0});
    auto ScatterUpdate_152236 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_426, {0}});
    auto slice_Slice = makeOP<ov::op::v1::StridedSlice>({Constant582, {0, 0, 0}, ScatterUpdate_152236, {1, 1, 1}},
                                                        {{"begin_mask", {1, 1, 0}},
                                                         {"end_mask", {1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto squeeze_Squeeze = makeOP<ov::op::v0::Squeeze>({slice_Slice, 1});
    auto squeeze_Squeeze_435 = makeOP<ov::op::v0::Squeeze>({squeeze_Squeeze, 0});
    auto index_441_Gather = makeOP<ov::op::v8::Gather>({squeeze_Squeeze_435, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze = makeOP<ov::op::v0::Unsqueeze>({index_441_Gather, 1});
    auto mul_Multiply =
        makeOP<ov::op::v1::Multiply>({transpose_Transpose, unsqueeze_Unsqueeze}, {{"auto_broadcast", "numpy"}});
    auto size_ShapeOf_448 = makeOP<ov::op::v3::ShapeOf>({transpose_Transpose}, {{"output_type", "i32"}});
    auto size_Gather_450 = makeOP<ov::op::v8::Gather>({size_ShapeOf_448, 3, 0}, {{"batch_dims", 0}});
    auto floor_divide_Divide =
        makeOP<ov::op::v1::Divide>({size_Gather_450, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floor_divide_Floor = makeOP<ov::op::v0::Floor>({floor_divide_Divide});
    auto slice_Unsqueeze_452 = makeOP<ov::op::v0::Unsqueeze>({floor_divide_Floor, 0});
    auto ScatterUpdate_152312 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_452, {0}});
    auto slice_Slice_459 = makeOP<ov::op::v1::StridedSlice>(
        {transpose_Transpose, ScatterUpdate_152312, {0ll, 0ll, 0ll, LLONG_MAX}, {1, 1, 1, 1}},
        {{"begin_mask", {1, 1, 1, 0}},
         {"end_mask", {1, 1, 1, 0}},
         {"new_axis_mask", {}},
         {"shrink_axis_mask", {}},
         {"ellipsis_mask", {}}});
    auto Constant_182988 = makeConst(element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto neg_Multiply = makeOP<ov::op::v1::Multiply>({slice_Slice_459, Constant_182988}, {{"auto_broadcast", "numpy"}});
    auto ScatterUpdate_152368 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_452, {0}});
    auto slice_Slice2 =
        makeOP<ov::op::v1::StridedSlice>({transpose_Transpose, {0, 0, 0, 0}, ScatterUpdate_152368, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::op::v0::Concat>({neg_Multiply, slice_Slice2}, {{"axis", -1}});
    auto ScatterUpdate_152421 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_426, {0}});
    auto slice_Slice_433 = makeOP<ov::op::v1::StridedSlice>({Constant585, {0, 0, 0}, ScatterUpdate_152421, {1, 1, 1}},
                                                            {{"begin_mask", {1, 1, 0}},
                                                             {"end_mask", {1, 1, 0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
    auto squeeze_Squeeze_436 = makeOP<ov::op::v0::Squeeze>({slice_Slice_433, 1});
    auto squeeze_Squeeze_437 = makeOP<ov::op::v0::Squeeze>({squeeze_Squeeze_436, 0});
    auto index_446_Gather = makeOP<ov::op::v8::Gather>({squeeze_Squeeze_437, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze_447 = makeOP<ov::op::v0::Unsqueeze>({index_446_Gather, 1});
    auto mul_Multiply_463 =
        makeOP<ov::op::v1::Multiply>({cat_Concat, unsqueeze_Unsqueeze_447}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::op::v1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, ov::ParameterVector{input, pos_id_end, pos_ids});
}

ov::Tensor RoPETestLlama2StridedSlice::create_i32_tensor(const ov::Shape& shape, int start, int step) {
    auto tensor = ov::Tensor(ov::element::i32, shape);
    auto* ptr = static_cast<int32_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        ptr[i] = start;
        start += step;
    }
    return tensor;
}

void RoPETestLlama2StridedSlice::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& funcInputs = function->inputs();

    const int position_id_start = 15;
    auto& input_shape = targetInputStaticShapes[0];
    auto seq_length = input_shape[1];

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = -1;
    in_data.range = 2;
    in_data.resolution = 32768;
    ov::Tensor t_input = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, in_data);
    ov::Tensor t_position_id_end = create_i32_tensor(ov::Shape({}), position_id_start + seq_length);
    ov::Tensor t_position_ids = create_i32_tensor(ov::Shape({1, seq_length}), position_id_start);

    inputs.clear();
    inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), t_position_id_end});
    inputs.insert({funcInputs[2].get_node_shared_ptr(), t_position_ids});
}

void RoPETestLlama2StridedSlice::SetUp() {
    targetDevice = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const size_t max_position_embeddings = 2048;
    const size_t ndims = 128;
    const size_t num_head = 32;

    InputShape inpShape = {{batch, seq_length, num_head, ndims}, {{batch, seq_length, num_head, ndims}}};
    init_input_shapes({inpShape});
    function = buildROPE_Llama2(batch, seq_length, max_position_embeddings, num_head, ndims);
}

std::string RoPETestLlama2StridedSlice::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    std::string targetDevice = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

std::shared_ptr<ov::Model> RoPETestChatGLMStridedSlice::buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, batch, 4096 + 256 + 256});
    auto cos_sin_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{32768, 32, 2});
    auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{-1, -1});

    auto __module_transformer_index_67_Gather =
        makeOP<opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});
    auto __module_transformer_transpose_Transpose =
        makeOP<opset1::Transpose>({__module_transformer_index_67_Gather, {1, 0, 2, 3}});
    auto size_ShapeOf_110 =
        makeOP<opset3::ShapeOf>({__module_transformer_transpose_Transpose}, {{"output_type", "i32"}});
    auto __getitem___Gather = makeOP<opset8::Gather>({size_ShapeOf_110, -2, 0}, {{"batch_dims", 0}});
    auto mul_Multiply = makeOP<opset1::Multiply>({__getitem___Gather, 2}, {{"auto_broadcast", "numpy"}});
    auto slice_Unsqueeze_112 = makeOP<opset1::Unsqueeze>({mul_Multiply, 0});

    auto floordiv_Divide =
        makeOP<opset1::Divide>({mul_Multiply, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floordiv_Floor = makeOP<opset1::Floor>({floordiv_Divide});
    auto ListConstruct_126_Reshape_2 = makeOP<opset1::Reshape>({floordiv_Floor, {-1}}, {{"special_zero", false}});

    auto ListUnpack_321 = makeOP<opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
    auto view_Reshape = makeOP<opset1::Reshape>({ListUnpack_321->output(0), {0, 0, 32, 128}}, {{"special_zero", true}});

    auto ScatterUpdate_229053 = makeOP<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_112, {0}});
    auto slice_Slice_357 =
        makeOP<opset1::StridedSlice>({view_Reshape, {0, 0, 0, 0}, ScatterUpdate_229053, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto size_ShapeOf_346 = makeOP<opset3::ShapeOf>({view_Reshape}, {{"output_type", "i32"}});
    auto size_Gather_348 = makeOP<opset8::Gather>({size_ShapeOf_346, 0, 0}, {{"batch_dims", 0}});
    auto ListConstruct_372_Reshape = makeOP<opset1::Reshape>({size_Gather_348, {-1}}, {{"special_zero", false}});
    auto size_Gather_351 = makeOP<opset8::Gather>({size_ShapeOf_346, {2}, 0}, {{"batch_dims", 0}});
    auto ListConstruct_372_Concat =
        makeOP<opset1::Concat>({ListConstruct_372_Reshape, {-1}, size_Gather_351, ListConstruct_126_Reshape_2, {2}},
                               {{"axis", 0}});
    auto reshape_Reshape_373 =
        makeOP<opset1::Reshape>({slice_Slice_357, ListConstruct_372_Concat}, {{"special_zero", false}});
    auto select_Gather_381 = makeOP<opset8::Gather>({reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});
    auto slice_Unsqueeze_367 = makeOP<opset1::Unsqueeze>({size_Gather_348, 0});
    auto slice_Slice_369 =
        makeOP<opset1::StridedSlice>({__module_transformer_transpose_Transpose, {0}, slice_Unsqueeze_367, {1}},
                                     {{"begin_mask", {0}},
                                      {"end_mask", {0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto size_ShapeOf_374 = makeOP<opset3::ShapeOf>({reshape_Reshape_373}, {{"output_type", "i32"}});
    auto size_Gather_376 = makeOP<opset8::Gather>({size_ShapeOf_374, {3}, 0}, {{"batch_dims", 0}});
    auto ListConstruct_379_Concat =
        makeOP<opset1::Concat>({ListConstruct_372_Reshape, {-1}, {1}, size_Gather_376, {2}}, {{"axis", 0}});
    auto view_Reshape_380 =
        makeOP<opset1::Reshape>({slice_Slice_369, ListConstruct_379_Concat}, {{"special_zero", false}});
    auto select_Gather_382 = makeOP<opset8::Gather>({view_Reshape_380, 0, -1}, {{"batch_dims", 0}});
    auto mul_Multiply_383 =
        makeOP<opset1::Multiply>({select_Gather_381, select_Gather_382}, {{"auto_broadcast", "numpy"}});
    auto select_Gather_384 = makeOP<opset8::Gather>({reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});
    auto select_Gather_385 = makeOP<opset8::Gather>({view_Reshape_380, 1, -1}, {{"batch_dims", 0}});
    auto mul_Multiply_386 =
        makeOP<opset1::Multiply>({select_Gather_384, select_Gather_385}, {{"auto_broadcast", "numpy"}});
    auto sub_Subtract_389 =
        makeOP<opset1::Subtract>({mul_Multiply_383, mul_Multiply_386}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_62716 = makeOP<opset1::Unsqueeze>({sub_Subtract_389, -1});
    auto mul_Multiply_391 =
        makeOP<opset1::Multiply>({select_Gather_384, select_Gather_382}, {{"auto_broadcast", "numpy"}});
    auto mul_Multiply_393 =
        makeOP<opset1::Multiply>({select_Gather_381, select_Gather_385}, {{"auto_broadcast", "numpy"}});
    auto add_Add_396 = makeOP<opset1::Add>({mul_Multiply_391, mul_Multiply_393}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_62717 = makeOP<opset1::Unsqueeze>({add_Add_396, -1});
    auto stack_401 = makeOP<opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
    auto flatten_ShapeOf_402 = makeOP<opset3::ShapeOf>({stack_401}, {{"output_type", "i32"}});
    auto flatten_Slice_417 = makeOP<opset1::StridedSlice>({flatten_ShapeOf_402, {0}, {3}, {1}},
                                                          {{"begin_mask", {0}},
                                                           {"end_mask", {0}},
                                                           {"new_axis_mask", {}},
                                                           {"shrink_axis_mask", {}},
                                                           {"ellipsis_mask", {}}});
    auto flatten_Concat_420 = makeOP<opset1::Concat>({flatten_Slice_417, {-1}}, {{"axis", 0}});
    auto flatten_Reshape_421 = makeOP<opset1::Reshape>({stack_401, flatten_Concat_420}, {{"special_zero", true}});
    auto ScatterUpdate_229067 = makeOP<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_112, {0}});
    auto slice_Slice_363 =
        makeOP<opset1::StridedSlice>({view_Reshape, ScatterUpdate_229067, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto cat_Concat_425 = makeOP<opset1::Concat>({flatten_Reshape_421, slice_Slice_363}, {{"axis", -1}});
    return std::make_shared<ov::Model>(ov::NodeVector{cat_Concat_425},
                                       ov::ParameterVector{input, cos_sin_cache, position_ids});
}

ov::Tensor RoPETestChatGLMStridedSlice::create_i32_tensor(const ov::Shape& shape, int start, int step) {
    auto tensor = ov::Tensor(ov::element::i32, shape);
    auto* ptr = static_cast<int32_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        ptr[i] = start;
        start += step;
    }
    return tensor;
}

void RoPETestChatGLMStridedSlice::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& funcInputs = function->inputs();

    auto& input_shape = targetInputStaticShapes[0];
    auto seq_length = input_shape[0];

    ov::Tensor t_input = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, 2, -1.0f, 32768);
    ov::Tensor t_cos_sin_cache =
        utils::create_and_fill_tensor(funcInputs[1].get_element_type(), {32768, 32, 2}, 2, -1.0f, 32768);
    ov::Tensor t_position_ids = create_i32_tensor(ov::Shape({1, seq_length}), 15);

    inputs.clear();
    inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), t_cos_sin_cache});
    inputs.insert({funcInputs[2].get_node_shared_ptr(), t_position_ids});
}

void RoPETestChatGLMStridedSlice::SetUp() {
    targetDevice = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const int num_head = 32;
    const int rotary_dims = 64;

    InputShape inpShape = {{-1, batch, 4096 + 256 + 256}, {{seq_length, batch, 4096 + 256 + 256}}};
    init_input_shapes({inpShape});
    function = buildROPE_ChatGLM(batch, num_head, rotary_dims);
}

std::string RoPETestChatGLMStridedSlice::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    std::string targetDevice = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

std::shared_ptr<ov::Model> RoPETestQwen7bStridedSlice::buildROPE_QWen7b(bool specialReshape) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, -1, 4096 + 4096 + 4096});
    auto cos_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{1, -1, 1, 128});
    auto sin_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{1, -1, 1, 128});

    auto ListUnpack_389_VariadicSplit = makeOP<opset1::VariadicSplit>({input, 2, {4096, 4096, -1}});
    auto view_Reshape =
        makeOP<opset1::Reshape>({ListUnpack_389_VariadicSplit->output(0), {0, 0, 32, 128}}, {{"special_zero", true}});
    auto size_ShapeOf_414 = makeOP<opset3::ShapeOf>({view_Reshape}, {{"output_type", "i32"}});
    auto size_Gather_416 = makeOP<opset8::Gather>({size_ShapeOf_414, 1, 0}, {{"batch_dims", 0}});
    auto neg_Multiply = makeOP<opset1::Multiply>({size_Gather_416, -1}, {{"auto_broadcast", "numpy"}});
    auto slice_Unsqueeze_422 = makeOP<opset1::Unsqueeze>({neg_Multiply, 0});
    auto ScatterUpdate_261437 = makeOP<opset3::ScatterUpdate>({{0, 0}, {1}, slice_Unsqueeze_422, {0}});
    auto slice_Slice_425 = makeOP<opset1::StridedSlice>({cos_cache, ScatterUpdate_261437, {0ll, LLONG_MAX}, {1, 1}},
                                                        {{"begin_mask", {1, 0}},
                                                         {"end_mask", {1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto slice_Slice_431 = makeOP<opset1::StridedSlice>({slice_Slice_425, {0, 0, 0}, {0ll, 0ll, LLONG_MAX}, {1, 1, 1}},
                                                        {{"begin_mask", {1, 1, 0}},
                                                         {"end_mask", {1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto slice_Slice_437 =
        makeOP<opset1::StridedSlice>({slice_Slice_431, {0, 0, 0, 0}, {0ll, 0ll, 0ll, LLONG_MAX}, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto size_ShapeOf_462 = makeOP<opset3::ShapeOf>({slice_Slice_437}, {{"output_type", "i32"}});
    auto size_Gather_464 = makeOP<opset8::Gather>({size_ShapeOf_462, {3}, 0}, {{"batch_dims", 0}});
    auto ScatterUpdate_261533 = makeOP<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, size_Gather_464, {0}});
    auto slice_Slice_470 =
        makeOP<opset1::StridedSlice>({view_Reshape, {0, 0, 0, 0}, ScatterUpdate_261533, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto mul_Multiply = makeOP<opset1::Multiply>({slice_Slice_470, slice_Slice_437}, {{"auto_broadcast", "numpy"}});
    auto size_ShapeOf_478 = makeOP<opset3::ShapeOf>({slice_Slice_470}, {{"output_type", "i32"}});
    auto Gather_239390 = makeOP<opset8::Gather>({size_ShapeOf_478, {0, 1, 2}, 0}, {{"batch_dims", 0}});
    auto size_Gather_489 = makeOP<opset8::Gather>({size_ShapeOf_478, 3, 0}, {{"batch_dims", 0}});
    auto floor_divide_Divide =
        makeOP<opset1::Divide>({size_Gather_489, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floor_divide_Floor = makeOP<opset1::Floor>({floor_divide_Divide});
    auto ListConstruct_493_Reshape_3 = makeOP<opset1::Reshape>({floor_divide_Floor, {-1}}, {{"special_zero", false}});
    auto ListConstruct_493_Concat =
        makeOP<opset1::Concat>({Gather_239390, {2}, ListConstruct_493_Reshape_3}, {{"axis", 0}});
    std::shared_ptr<ov::Node> reshape_Reshape = nullptr;
    if (specialReshape) {
        reshape_Reshape = makeOP<opset1::Reshape>({slice_Slice_470, {0, 0, 32, 2, 64}}, {{"special_zero", true}});
    } else {
        reshape_Reshape =
            makeOP<opset1::Reshape>({slice_Slice_470, ListConstruct_493_Concat}, {{"special_zero", false}});
    }
    auto ListUnpack_496_Split = makeOP<opset1::Split>({reshape_Reshape, -2}, {{"num_splits", 2}});
    auto ListUnpack_496_Squeeze_0 = makeOP<opset1::Squeeze>({ListUnpack_496_Split->output(1), -2});
    auto Constant_296840_compressed = makeConst(element::f16,
                                                ov::Shape({
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                }),
                                                {-1});
    auto Constant_296840 = makeOP<opset1::Convert>({Constant_296840_compressed}, {{"destination_type", "f32"}});
    auto neg_Multiply_499 =
        makeOP<opset1::Multiply>({ListUnpack_496_Squeeze_0, Constant_296840}, {{"auto_broadcast", "numpy"}});
    auto ListUnpack_496_Squeeze = makeOP<opset1::Squeeze>({ListUnpack_496_Split->output(0), -2});
    auto cat_Concat = makeOP<opset1::Concat>({neg_Multiply_499, ListUnpack_496_Squeeze}, {{"axis", -1}});
    auto slice_Slice_449 = makeOP<opset1::StridedSlice>({sin_cache, ScatterUpdate_261437, {0ll, LLONG_MAX}, {1, 1}},
                                                        {{"begin_mask", {1, 0}},
                                                         {"end_mask", {1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto slice_Slice_455 = makeOP<opset1::StridedSlice>({slice_Slice_449, {0, 0, 0}, {0ll, 0ll, LLONG_MAX}, {1, 1, 1}},
                                                        {{"begin_mask", {1, 1, 0}},
                                                         {"end_mask", {1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto slice_Slice_461 =
        makeOP<opset1::StridedSlice>({slice_Slice_455, {0, 0, 0, 0}, {0ll, 0ll, 0ll, LLONG_MAX}, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto mul_Multiply_503 = makeOP<opset1::Multiply>({cat_Concat, slice_Slice_461}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<opset1::Add>({mul_Multiply, mul_Multiply_503}, {{"auto_broadcast", "numpy"}});
    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, ov::ParameterVector{input, cos_cache, sin_cache});
}

void RoPETestQwen7bStridedSlice::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& funcInputs = function->inputs();

    auto& input_shape = targetInputStaticShapes[0];

    ov::Tensor t_input = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, 2, -1.0f, 32768);
    ov::Tensor t_cos_cache =
        utils::create_and_fill_tensor(funcInputs[1].get_element_type(), {1, 4096, 1, 128}, 2, -1.0f, 32768);
    ov::Tensor t_sin_cache =
        utils::create_and_fill_tensor(funcInputs[1].get_element_type(), {1, 4096, 1, 128}, 2, -1.0f, 32768);

    inputs.clear();
    inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), t_cos_cache});
    inputs.insert({funcInputs[2].get_node_shared_ptr(), t_sin_cache});
}

void RoPETestQwen7bStridedSlice::SetUp() {
    bool specialReshape;
    std::tie(specialReshape, targetDevice) = this->GetParam();
    const int batch = 2;
    const int seq_length = 7;
    InputShape inpShape = {{batch, -1, 4096 + 4096 + 4096}, {{batch, seq_length, 4096 + 4096 + 4096}}};
    init_input_shapes({inpShape});
    function = buildROPE_QWen7b(specialReshape);
}

std::string RoPETestQwen7bStridedSlice::getTestCaseName(
    const testing::TestParamInfo<std::tuple<bool, std::string>>& obj) {
    bool specialReshape;
    std::string targetDevice;
    std::tie(specialReshape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "specialReshape=" << specialReshape << "_"
           << "targetDevice=" << targetDevice;
    return result.str();
}

std::shared_ptr<ov::Model> RoPETestGPTJStridedSlice::buildROPE_GPTJ(int num_head,
                                                                    int hidden_dims,
                                                                    int rotary_dims,
                                                                    bool hasShapeOf) {
    auto int32_max = std::numeric_limits<std::int32_t>::max();
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, -1, num_head, hidden_dims});
    auto sincos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, -1, rotary_dims});

    auto slice_Slice_965 = makeOP<ov::op::v1::StridedSlice>({input, {0, 0, 0, 0}, {0, 0, 0, rotary_dims}, {1, 1, 1, 1}},
                                                            {{"begin_mask", {1, 1, 1, 0}},
                                                             {"end_mask", {1, 1, 1, 0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
    slice_Slice_965->set_friendly_name("slice_Slice_965");

    auto varsplit = makeOP<ov::op::v1::VariadicSplit>({sincos, -1, {rotary_dims / 2, -1}});
    varsplit->set_output_size(2);
    varsplit->set_friendly_name("varsplit");
    auto unsqueeze_sin = makeOP<opset1::Unsqueeze>({varsplit->output(0), 2});
    auto unsqueeze_cos = makeOP<opset1::Unsqueeze>({varsplit->output(1), 2});
    std::vector<int32_t> gather_idx(rotary_dims, 1);
    int32_t v = 0;
    for (size_t i = 0; i < gather_idx.size(); i += 2, v++) {
        gather_idx[i] = v;
        gather_idx[i + 1] = v;
    }

    auto const_idx = makeConst(ov::element::i32, ov::Shape({static_cast<size_t>(rotary_dims)}), gather_idx);
    auto constant_155588 = makeConst(element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto repeat_interleave_sin = makeOP<opset8::Gather>({unsqueeze_sin, const_idx, 3}, {{"batch_dims", 0}});
    auto repeat_interleave_cos = makeOP<opset8::Gather>({unsqueeze_cos, const_idx, 3}, {{"batch_dims", 0}});
    repeat_interleave_sin->set_friendly_name("repeat_interleave_sin");
    repeat_interleave_cos->set_friendly_name("repeat_interleave_cos");
    // x interleave (-x[:,:,:, 1::2], x[:,:,:, 0::2])
    auto slice_Slice_1174 =
        makeOP<ov::op::v1::StridedSlice>({slice_Slice_965, {0, 0, 0, 1}, {0, 0, 0, int32_max}, {1, 1, 1, 2}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto neg_Multiply_1177 =
        makeOP<opset1::Multiply>({slice_Slice_1174, constant_155588}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_65524 = makeOP<opset1::Unsqueeze>({neg_Multiply_1177, -1});

    auto slice_Slice_1168 =
        makeOP<ov::op::v1::StridedSlice>({slice_Slice_965, {0, 0, 0, 0}, {0, 0, 0, int32_max}, {1, 1, 1, 2}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto Unsqueeze_65525 = makeOP<opset1::Unsqueeze>({slice_Slice_1168, -1});
    auto stack_1182 = makeOP<opset1::Concat>({Unsqueeze_65524, Unsqueeze_65525}, {{"axis", -1}});
    auto flatten_Reshape_1198 =
        makeOP<opset1::Reshape>({stack_1182, {0, 0, num_head, rotary_dims}}, {{"special_zero", true}});
    // x*cos [B,L,H,ndims]
    auto mul_cos = makeOP<opset1::Multiply>({slice_Slice_965, repeat_interleave_cos}, {{"auto_broadcast", "numpy"}});
    mul_cos->set_friendly_name("mul_cos");
    auto mul_sin =
        makeOP<opset1::Multiply>({flatten_Reshape_1198, repeat_interleave_sin}, {{"auto_broadcast", "numpy"}});
    // *cos + *sin
    auto rotary_emb = makeOP<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_971 =
        makeOP<ov::op::v1::StridedSlice>({input, {0, 0, 0, rotary_dims}, {0, 0, 0, int32_max}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat_1211 = makeOP<opset1::Concat>({rotary_emb, slice_Slice_971}, {{"axis", -1}});
    auto permute_Transpose_1213 = makeOP<opset1::Transpose>({cat_Concat_1211, {0, 2, 1, 3}});
    ov::NodeVector model_output = {permute_Transpose_1213};
    if (hasShapeOf) {
        auto shapeOf = makeOP<opset1::ShapeOf>({rotary_emb}, {{"output_type", "i32"}});
        auto gather = makeOP<opset8::Gather>({shapeOf, {1}, 0}, {{"batch_dims", 0}});
        model_output.push_back(gather);
    }
    return std::make_shared<ov::Model>(model_output, ov::ParameterVector{input, sincos});
}

void RoPETestGPTJStridedSlice::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& funcInputs = function->inputs();

    auto& input_shape = targetInputStaticShapes[0];
    auto& sincos_shape = targetInputStaticShapes[1];
    ov::Tensor t_input = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, 2, -1.0f, 32768);
    ov::Tensor t_cos_sin_cache =
        utils::create_and_fill_tensor(funcInputs[1].get_element_type(), sincos_shape, 2, -1.0f, 32768);

    inputs.clear();
    inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), t_cos_sin_cache});
}

std::string RoPETestGPTJStridedSlice::getTestCaseName(
    const testing::TestParamInfo<std::tuple<bool, std::string>>& obj) {
    bool hasShapeOf;
    std::string targetDevice;
    std::tie(hasShapeOf, targetDevice) = obj.param;
    std::ostringstream result;
    result << "hasShapeOf=" << hasShapeOf << "_"
           << "targetDevice=" << targetDevice;
    return result.str();
}

void RoPETestGPTJStridedSlice::SetUp() {
    bool hasShapeOf;
    std::tie(hasShapeOf, targetDevice) = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const int num_head = 16;
    const int hidden_dims = 256;
    const int rotary_dims = 64;

    InputShape input = {{batch, seq_length, num_head, hidden_dims}, {{batch, seq_length, num_head, hidden_dims}}};
    InputShape sincos = {{batch, seq_length, rotary_dims}, {{batch, seq_length, rotary_dims}}};
    init_input_shapes({input, sincos});
    function = buildROPE_GPTJ(num_head, hidden_dims, rotary_dims, hasShapeOf);
}

ov::OutputVector RoPETestRotateHalfWithoutTranspose::makeCosSinCache(int max_position_embeddings, int rotary_ndims) {
    std::vector<float> lut_sin(max_position_embeddings * rotary_ndims, 0.0f);
    std::vector<float> lut_cos(max_position_embeddings * rotary_ndims, 0.0f);

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (int i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
        auto xita_i = 1.0 / std::pow(10000.0, static_cast<double>(i) / rotary_ndims);
        float* psin = lut_sin.data();
        float* pcos = lut_cos.data();
        for (int m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
            auto vsin = std::sin(xita_i * m);
            auto vcos = std::cos(xita_i * m);
            pcos[k] = pcos[k + rotary_ndims / 2] = vcos;
            psin[k] = psin[k + rotary_ndims / 2] = vsin;
        }
    }
    auto shape = ov::Shape({1, static_cast<size_t>(max_position_embeddings), 1, static_cast<size_t>(rotary_ndims)});
    auto Cos = makeConst(ov::element::f32, shape, lut_cos);
    auto Sin = makeConst(ov::element::f32, shape, lut_sin);
    return {Cos, Sin};
}

std::shared_ptr<ov::Model> RoPETestRotateHalfWithoutTranspose::buildROPE_RotateHalfWithoutTranspose(
    int batch,
    int seq_length,
    int max_position_embeddings,
    int num_head,
    int ndims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{batch, num_head, -1, ndims});
    auto pos_id_end = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{});
    auto pos_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{1, -1});

    auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);
    auto Constant582 = cos_sin_cache[0];
    auto Constant585 = cos_sin_cache[1];

    // concat KV length
    auto slice_Unsqueeze_426 = makeOP<ov::op::v0::Unsqueeze>({pos_id_end, 0});
    auto ScatterUpdate_152236 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0}, {1}, slice_Unsqueeze_426, {0}});
    auto slice_Slice = makeOP<ov::op::v1::StridedSlice>({Constant582, {0, 0, 0}, ScatterUpdate_152236, {1, 1, 1}},
                                                        {{"begin_mask", {1, 0, 1}},
                                                         {"end_mask", {1, 0, 1}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto squeeze_Squeeze = makeOP<ov::op::v0::Squeeze>({slice_Slice, 2});
    auto squeeze_Squeeze_435 = makeOP<ov::op::v0::Squeeze>({squeeze_Squeeze, 0});
    auto index_441_Gather = makeOP<ov::op::v8::Gather>({squeeze_Squeeze_435, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze = makeOP<ov::op::v0::Unsqueeze>({index_441_Gather, 1});
    auto mul_Multiply = makeOP<ov::op::v1::Multiply>({input, unsqueeze_Unsqueeze}, {{"auto_broadcast", "numpy"}});
    auto size_ShapeOf_448 = makeOP<ov::op::v3::ShapeOf>({input}, {{"output_type", "i32"}});
    auto size_Gather_450 = makeOP<ov::op::v8::Gather>({size_ShapeOf_448, 3, 0}, {{"batch_dims", 0}});
    auto floor_divide_Divide =
        makeOP<ov::op::v1::Divide>({size_Gather_450, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floor_divide_Floor = makeOP<ov::op::v0::Floor>({floor_divide_Divide});
    auto slice_Unsqueeze_452 = makeOP<ov::op::v0::Unsqueeze>({floor_divide_Floor, 0});
    auto ScatterUpdate_152312 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_452, {0}});
    auto slice_Slice_459 =
        makeOP<ov::op::v1::StridedSlice>({input, ScatterUpdate_152312, {0ll, 0ll, 0ll, LLONG_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto Constant_182988 = makeConst(element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto neg_Multiply = makeOP<ov::op::v1::Multiply>({slice_Slice_459, Constant_182988}, {{"auto_broadcast", "numpy"}});
    auto ScatterUpdate_152368 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_452, {0}});
    auto slice_Slice2 = makeOP<ov::op::v1::StridedSlice>({input, {0, 0, 0, 0}, ScatterUpdate_152368, {1, 1, 1, 1}},
                                                         {{"begin_mask", {1, 1, 1, 0}},
                                                          {"end_mask", {1, 1, 1, 0}},
                                                          {"new_axis_mask", {}},
                                                          {"shrink_axis_mask", {}},
                                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::op::v0::Concat>({neg_Multiply, slice_Slice2}, {{"axis", -1}});
    auto ScatterUpdate_152421 = makeOP<ov::op::v3::ScatterUpdate>({{0, 0, 0}, {1}, slice_Unsqueeze_426, {0}});
    auto slice_Slice_433 = makeOP<ov::op::v1::StridedSlice>({Constant585, {0, 0, 0}, ScatterUpdate_152421, {1, 1, 1}},
                                                            {{"begin_mask", {1, 0, 1}},
                                                             {"end_mask", {1, 0, 1}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
    auto squeeze_Squeeze_436 = makeOP<ov::op::v0::Squeeze>({slice_Slice_433, 2});
    auto squeeze_Squeeze_437 = makeOP<ov::op::v0::Squeeze>({squeeze_Squeeze_436, 0});
    auto index_446_Gather = makeOP<ov::op::v8::Gather>({squeeze_Squeeze_437, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze_447 = makeOP<ov::op::v0::Unsqueeze>({index_446_Gather, 1});
    auto mul_Multiply_463 =
        makeOP<ov::op::v1::Multiply>({cat_Concat, unsqueeze_Unsqueeze_447}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::op::v1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, ov::ParameterVector{input, pos_id_end, pos_ids});
}

ov::Tensor RoPETestRotateHalfWithoutTranspose::create_i32_tensor(const ov::Shape& shape, int start, int step) {
    auto tensor = ov::Tensor(ov::element::i32, shape);
    auto* ptr = static_cast<int32_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        ptr[i] = start;
        start += step;
    }
    return tensor;
}

void RoPETestRotateHalfWithoutTranspose::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& funcInputs = function->inputs();

    const int position_id_start = 15;
    auto& input_shape = targetInputStaticShapes[0];
    auto seq_length = input_shape[2];

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = -1;
    in_data.range = 2;
    in_data.resolution = 32768;
    ov::Tensor t_input = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, in_data);
    ov::Tensor t_position_id_end = create_i32_tensor(ov::Shape({}), position_id_start + seq_length);
    ov::Tensor t_position_ids = create_i32_tensor(ov::Shape({1, seq_length}), position_id_start);

    inputs.clear();
    inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), t_position_id_end});
    inputs.insert({funcInputs[2].get_node_shared_ptr(), t_position_ids});
}

void RoPETestRotateHalfWithoutTranspose::SetUp() {
    targetDevice = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const size_t max_position_embeddings = 2048;
    const size_t ndims = 128;
    const size_t num_head = 32;

    InputShape inpShape = {{batch, num_head, seq_length, ndims}, {{batch, num_head, seq_length, ndims}}};
    init_input_shapes({inpShape});
    function = buildROPE_RotateHalfWithoutTranspose(batch, seq_length, max_position_embeddings, num_head, ndims);
}

std::string RoPETestRotateHalfWithoutTranspose::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    std::string targetDevice = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void RoPETestLlama2Slice::SetUp() {
    targetDevice = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const size_t max_position_embeddings = 2048;
    const size_t ndims = 128;
    const size_t num_head = 32;

    InputShape inpShape = {{batch, seq_length, num_head, ndims}, {{batch, seq_length, num_head, ndims}}};
    init_input_shapes({inpShape});
    function = RoPETestLlama2Slice::buildROPE_Llama2(batch, seq_length, max_position_embeddings, num_head, ndims);
}

std::shared_ptr<ov::Model> RoPETestLlama2Slice::buildROPE_Llama2(int batch,
                                                                 int seq_length,
                                                                 int max_position_embeddings,
                                                                 int num_head,
                                                                 int ndims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{batch, -1, num_head, ndims});
    auto pos_id_end = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{});
    auto pos_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{1, -1});

    auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);
    auto Constant582 = cos_sin_cache[0];
    auto Constant585 = cos_sin_cache[1];

    // concat KV length
    auto transpose_Transpose = makeOP<ov::op::v1::Transpose>({input, {0, 2, 1, 3}});
    auto slice_Unsqueeze_426 = makeOP<ov::op::v0::Unsqueeze>({pos_id_end, 0});
    auto slice_Slice = makeOP<ov::op::v8::Slice>({Constant582, {0}, slice_Unsqueeze_426, {1}, {2}});
    auto squeeze_Squeeze = makeOP<ov::op::v0::Squeeze>({slice_Slice, 1});
    auto squeeze_Squeeze_435 = makeOP<ov::op::v0::Squeeze>({squeeze_Squeeze, 0});
    auto index_441_Gather = makeOP<ov::op::v8::Gather>({squeeze_Squeeze_435, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze = makeOP<ov::op::v0::Unsqueeze>({index_441_Gather, 1});
    auto mul_Multiply =
        makeOP<ov::op::v1::Multiply>({transpose_Transpose, unsqueeze_Unsqueeze}, {{"auto_broadcast", "numpy"}});
    auto size_ShapeOf_448 = makeOP<ov::op::v3::ShapeOf>({transpose_Transpose}, {{"output_type", "i32"}});
    auto size_Gather_450 = makeOP<ov::op::v8::Gather>({size_ShapeOf_448, 3, 0}, {{"batch_dims", 0}});
    auto floor_divide_Divide =
        makeOP<ov::op::v1::Divide>({size_Gather_450, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floor_divide_Floor = makeOP<ov::op::v0::Floor>({floor_divide_Divide});
    auto slice_Unsqueeze_452 = makeOP<ov::op::v0::Unsqueeze>({floor_divide_Floor, 0});
    auto slice_Slice_459 = makeOP<ov::op::v8::Slice>({transpose_Transpose, slice_Unsqueeze_452, {LLONG_MAX}, {1}, {3}});
    auto Constant_182988 = makeConst(element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto neg_Multiply = makeOP<ov::op::v1::Multiply>({slice_Slice_459, Constant_182988}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice2 = makeOP<ov::op::v8::Slice>({transpose_Transpose, {0}, slice_Unsqueeze_452, {1}, {3}});
    auto cat_Concat = makeOP<ov::op::v0::Concat>({neg_Multiply, slice_Slice2}, {{"axis", -1}});
    auto slice_Slice_433 = makeOP<ov::op::v8::Slice>({Constant585, {0}, slice_Unsqueeze_426, {1}, {2}});
    auto squeeze_Squeeze_436 = makeOP<ov::op::v0::Squeeze>({slice_Slice_433, 1});
    auto squeeze_Squeeze_437 = makeOP<ov::op::v0::Squeeze>({squeeze_Squeeze_436, 0});
    auto index_446_Gather = makeOP<ov::op::v8::Gather>({squeeze_Squeeze_437, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze_447 = makeOP<ov::op::v0::Unsqueeze>({index_446_Gather, 1});
    auto mul_Multiply_463 =
        makeOP<ov::op::v1::Multiply>({cat_Concat, unsqueeze_Unsqueeze_447}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::op::v1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, ov::ParameterVector{input, pos_id_end, pos_ids});
}

void RoPETestChatGLMSlice::SetUp() {
    targetDevice = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const int num_head = 32;
    const int rotary_dims = 64;

    InputShape inpShape = {{-1, batch, 4096 + 256 + 256}, {{seq_length, batch, 4096 + 256 + 256}}};
    init_input_shapes({inpShape});
    function = RoPETestChatGLMSlice::buildROPE_ChatGLM(batch, num_head, rotary_dims);
}

std::shared_ptr<ov::Model> RoPETestChatGLMSlice::buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, batch, 4096 + 256 + 256});
    auto cos_sin_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{32768, 32, 2});
    auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{-1, -1});

    auto __module_transformer_index_67_Gather =
        makeOP<opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});
    auto __module_transformer_transpose_Transpose =
        makeOP<opset1::Transpose>({__module_transformer_index_67_Gather, {1, 0, 2, 3}});
    auto size_ShapeOf_110 =
        makeOP<opset3::ShapeOf>({__module_transformer_transpose_Transpose}, {{"output_type", "i32"}});
    auto __getitem___Gather = makeOP<opset8::Gather>({size_ShapeOf_110, -2, 0}, {{"batch_dims", 0}});
    auto mul_Multiply = makeOP<opset1::Multiply>({__getitem___Gather, 2}, {{"auto_broadcast", "numpy"}});
    auto slice_Unsqueeze_112 = makeOP<opset1::Unsqueeze>({mul_Multiply, 0});

    auto floordiv_Divide =
        makeOP<opset1::Divide>({mul_Multiply, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floordiv_Floor = makeOP<opset1::Floor>({floordiv_Divide});
    auto ListConstruct_126_Reshape_2 = makeOP<opset1::Reshape>({floordiv_Floor, {-1}}, {{"special_zero", false}});

    auto ListUnpack_321 = makeOP<opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
    auto view_Reshape = makeOP<opset1::Reshape>({ListUnpack_321->output(0), {0, 0, 32, 128}}, {{"special_zero", true}});

    auto slice_Slice_357 = makeOP<opset8::Slice>({view_Reshape, {0}, slice_Unsqueeze_112, {1}, {3}});
    auto size_ShapeOf_346 = makeOP<opset3::ShapeOf>({view_Reshape}, {{"output_type", "i32"}});
    auto size_Gather_348 = makeOP<opset8::Gather>({size_ShapeOf_346, 0, 0}, {{"batch_dims", 0}});
    auto ListConstruct_372_Reshape = makeOP<opset1::Reshape>({size_Gather_348, {-1}}, {{"special_zero", false}});
    auto size_Gather_351 = makeOP<opset8::Gather>({size_ShapeOf_346, {2}, 0}, {{"batch_dims", 0}});
    auto ListConstruct_372_Concat =
        makeOP<opset1::Concat>({ListConstruct_372_Reshape, {-1}, size_Gather_351, ListConstruct_126_Reshape_2, {2}},
                               {{"axis", 0}});
    auto reshape_Reshape_373 =
        makeOP<opset1::Reshape>({slice_Slice_357, ListConstruct_372_Concat}, {{"special_zero", false}});
    auto select_Gather_381 = makeOP<opset8::Gather>({reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});
    auto slice_Unsqueeze_367 = makeOP<opset1::Unsqueeze>({size_Gather_348, 0});
    auto slice_Slice_369 = makeOP<opset8::Slice>({__module_transformer_transpose_Transpose, {0}, slice_Unsqueeze_367, {1}, {0}});
    auto size_ShapeOf_374 = makeOP<opset3::ShapeOf>({reshape_Reshape_373}, {{"output_type", "i32"}});
    auto size_Gather_376 = makeOP<opset8::Gather>({size_ShapeOf_374, {3}, 0}, {{"batch_dims", 0}});
    auto ListConstruct_379_Concat =
        makeOP<opset1::Concat>({ListConstruct_372_Reshape, {-1}, {1}, size_Gather_376, {2}}, {{"axis", 0}});
    auto view_Reshape_380 =
        makeOP<opset1::Reshape>({slice_Slice_369, ListConstruct_379_Concat}, {{"special_zero", false}});
    auto select_Gather_382 = makeOP<opset8::Gather>({view_Reshape_380, 0, -1}, {{"batch_dims", 0}});
    auto mul_Multiply_383 =
        makeOP<opset1::Multiply>({select_Gather_381, select_Gather_382}, {{"auto_broadcast", "numpy"}});
    auto select_Gather_384 = makeOP<opset8::Gather>({reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});
    auto select_Gather_385 = makeOP<opset8::Gather>({view_Reshape_380, 1, -1}, {{"batch_dims", 0}});
    auto mul_Multiply_386 =
        makeOP<opset1::Multiply>({select_Gather_384, select_Gather_385}, {{"auto_broadcast", "numpy"}});
    auto sub_Subtract_389 =
        makeOP<opset1::Subtract>({mul_Multiply_383, mul_Multiply_386}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_62716 = makeOP<opset1::Unsqueeze>({sub_Subtract_389, -1});
    auto mul_Multiply_391 =
        makeOP<opset1::Multiply>({select_Gather_384, select_Gather_382}, {{"auto_broadcast", "numpy"}});
    auto mul_Multiply_393 =
        makeOP<opset1::Multiply>({select_Gather_381, select_Gather_385}, {{"auto_broadcast", "numpy"}});
    auto add_Add_396 = makeOP<opset1::Add>({mul_Multiply_391, mul_Multiply_393}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_62717 = makeOP<opset1::Unsqueeze>({add_Add_396, -1});
    auto stack_401 = makeOP<opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
    auto flatten_ShapeOf_402 = makeOP<opset3::ShapeOf>({stack_401}, {{"output_type", "i32"}});
    auto flatten_Slice_417 = makeOP<opset8::Slice>({flatten_ShapeOf_402, {0}, {3}, {1}, {0}});
    auto flatten_Concat_420 = makeOP<opset1::Concat>({flatten_Slice_417, {-1}}, {{"axis", 0}});
    auto flatten_Reshape_421 = makeOP<opset1::Reshape>({stack_401, flatten_Concat_420}, {{"special_zero", true}});
    auto slice_Slice_363 = makeOP<opset8::Slice>({view_Reshape, slice_Unsqueeze_112, {INT_MAX}, {1}, {3}});
    auto cat_Concat_425 = makeOP<opset1::Concat>({flatten_Reshape_421, slice_Slice_363}, {{"axis", -1}});
    return std::make_shared<ov::Model>(ov::NodeVector{cat_Concat_425},
                                       ov::ParameterVector{input, cos_sin_cache, position_ids});
}

void RoPETestQwen7bSlice::SetUp() {
    bool specialReshape;
    std::tie(specialReshape, targetDevice) = this->GetParam();
    const int batch = 2;
    const int seq_length = 7;
    InputShape inpShape = {{batch, -1, 4096 + 4096 + 4096}, {{batch, seq_length, 4096 + 4096 + 4096}}};
    init_input_shapes({inpShape});
    function = RoPETestQwen7bSlice::buildROPE_Qwen7b(specialReshape);
}

std::shared_ptr<ov::Model> RoPETestQwen7bSlice::buildROPE_Qwen7b(bool specialReshape) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, -1, 4096 + 4096 + 4096});
    auto cos_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{1, -1, 1, 128});
    auto sin_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{1, -1, 1, 128});

    auto ListUnpack_389_VariadicSplit = makeOP<opset1::VariadicSplit>({input, 2, {4096, 4096, -1}});
    auto view_Reshape =
        makeOP<opset1::Reshape>({ListUnpack_389_VariadicSplit->output(0), {0, 0, 32, 128}}, {{"special_zero", true}});
    auto size_ShapeOf_414 = makeOP<opset3::ShapeOf>({view_Reshape}, {{"output_type", "i32"}});
    auto size_Gather_416 = makeOP<opset8::Gather>({size_ShapeOf_414, 1, 0}, {{"batch_dims", 0}});
    auto neg_Multiply = makeOP<opset1::Multiply>({size_Gather_416, -1}, {{"auto_broadcast", "numpy"}});
    auto slice_Unsqueeze_422 = makeOP<opset1::Unsqueeze>({neg_Multiply, 0});
    auto slice_Slice_425 = makeOP<opset8::Slice>({cos_cache, slice_Unsqueeze_422, {LLONG_MAX}, {1}, {1}});
    auto size_ShapeOf_462 = makeOP<opset3::ShapeOf>({slice_Slice_425}, {{"output_type", "i32"}});
    auto size_Gather_464 = makeOP<opset8::Gather>({size_ShapeOf_462, {3}, 0}, {{"batch_dims", 0}});
    auto slice_Slice_470 = makeOP<opset8::Slice>({view_Reshape, {0}, size_Gather_464, {1}, {3}});
    auto mul_Multiply = makeOP<opset1::Multiply>({slice_Slice_470, slice_Slice_425}, {{"auto_broadcast", "numpy"}});
    auto size_ShapeOf_478 = makeOP<opset3::ShapeOf>({slice_Slice_470}, {{"output_type", "i32"}});
    auto Gather_239390 = makeOP<opset8::Gather>({size_ShapeOf_478, {0, 1, 2}, 0}, {{"batch_dims", 0}});
    auto size_Gather_489 = makeOP<opset8::Gather>({size_ShapeOf_478, 3, 0}, {{"batch_dims", 0}});
    auto floor_divide_Divide =
        makeOP<opset1::Divide>({size_Gather_489, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floor_divide_Floor = makeOP<opset1::Floor>({floor_divide_Divide});
    auto ListConstruct_493_Reshape_3 = makeOP<opset1::Reshape>({floor_divide_Floor, {-1}}, {{"special_zero", false}});
    auto ListConstruct_493_Concat =
        makeOP<opset1::Concat>({Gather_239390, {2}, ListConstruct_493_Reshape_3}, {{"axis", 0}});
    std::shared_ptr<ov::Node> reshape_Reshape = nullptr;
    if (specialReshape) {
        reshape_Reshape = makeOP<opset1::Reshape>({slice_Slice_470, {0, 0, 32, 2, 64}}, {{"special_zero", true}});
    } else {
        reshape_Reshape =
            makeOP<opset1::Reshape>({slice_Slice_470, ListConstruct_493_Concat}, {{"special_zero", false}});
    }
    auto ListUnpack_496_Split = makeOP<opset1::Split>({reshape_Reshape, -2}, {{"num_splits", 2}});
    auto ListUnpack_496_Squeeze_0 = makeOP<opset1::Squeeze>({ListUnpack_496_Split->output(1), -2});
    auto Constant_296840_compressed = makeConst(element::f16,
                                                ov::Shape({
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                }),
                                                {-1});
    auto Constant_296840 = makeOP<opset1::Convert>({Constant_296840_compressed}, {{"destination_type", "f32"}});
    auto neg_Multiply_499 =
        makeOP<opset1::Multiply>({ListUnpack_496_Squeeze_0, Constant_296840}, {{"auto_broadcast", "numpy"}});
    auto ListUnpack_496_Squeeze = makeOP<opset1::Squeeze>({ListUnpack_496_Split->output(0), -2});
    auto cat_Concat = makeOP<opset1::Concat>({neg_Multiply_499, ListUnpack_496_Squeeze}, {{"axis", -1}});
    auto slice_Slice_449 = makeOP<opset8::Slice>({sin_cache, slice_Unsqueeze_422, {LLONG_MAX}, {1}, {1}});
    auto mul_Multiply_503 = makeOP<opset1::Multiply>({cat_Concat, slice_Slice_449}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<opset1::Add>({mul_Multiply, mul_Multiply_503}, {{"auto_broadcast", "numpy"}});
    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, ov::ParameterVector{input, cos_cache, sin_cache});
}

void RoPETestGPTJSlice::SetUp() {
    bool hasShapeOf;
    std::tie(hasShapeOf, targetDevice) = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const int num_head = 16;
    const int hidden_dims = 256;
    const int rotary_dims = 64;

    InputShape input = {{batch, seq_length, num_head, hidden_dims}, {{batch, seq_length, num_head, hidden_dims}}};
    InputShape sincos = {{batch, seq_length, rotary_dims}, {{batch, seq_length, rotary_dims}}};
    init_input_shapes({input, sincos});
    function = RoPETestGPTJSlice::buildROPE_GPTJ(num_head, hidden_dims, rotary_dims, hasShapeOf);
}

std::shared_ptr<ov::Model> RoPETestGPTJSlice::buildROPE_GPTJ(int num_head,
                                                             int hidden_dims,
                                                             int rotary_dims,
                                                             bool hasShapeOf) {
    auto int32_max = std::numeric_limits<std::int32_t>::max();
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, -1, num_head, hidden_dims});
    auto sincos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{-1, -1, rotary_dims});

    auto slice_Slice_965 = makeOP<ov::op::v8::Slice>({input, {0}, {rotary_dims}, {1}, {3}});
    slice_Slice_965->set_friendly_name("slice_Slice_965");

    auto varsplit = makeOP<ov::op::v1::VariadicSplit>({sincos, -1, {rotary_dims / 2, -1}});
    varsplit->set_output_size(2);
    varsplit->set_friendly_name("varsplit");
    auto unsqueeze_sin = makeOP<opset1::Unsqueeze>({varsplit->output(0), 2});
    auto unsqueeze_cos = makeOP<opset1::Unsqueeze>({varsplit->output(1), 2});
    std::vector<int32_t> gather_idx(rotary_dims, 1);
    int32_t v = 0;
    for (size_t i = 0; i < gather_idx.size(); i += 2, v++) {
        gather_idx[i] = v;
        gather_idx[i + 1] = v;
    }

    auto const_idx = makeConst(ov::element::i32, ov::Shape({static_cast<size_t>(rotary_dims)}), gather_idx);
    auto constant_155588 = makeConst(element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto repeat_interleave_sin = makeOP<opset8::Gather>({unsqueeze_sin, const_idx, 3}, {{"batch_dims", 0}});
    auto repeat_interleave_cos = makeOP<opset8::Gather>({unsqueeze_cos, const_idx, 3}, {{"batch_dims", 0}});
    repeat_interleave_sin->set_friendly_name("repeat_interleave_sin");
    repeat_interleave_cos->set_friendly_name("repeat_interleave_cos");
    // x interleave (-x[:,:,:, 1::2], x[:,:,:, 0::2])
    auto slice_Slice_1174 = makeOP<ov::op::v8::Slice>({slice_Slice_965, {1}, {int32_max}, {2}, {3}});
    auto neg_Multiply_1177 =
        makeOP<opset1::Multiply>({slice_Slice_1174, constant_155588}, {{"auto_broadcast", "numpy"}});
    auto Unsqueeze_65524 = makeOP<opset1::Unsqueeze>({neg_Multiply_1177, -1});

    auto slice_Slice_1168 = makeOP<ov::op::v8::Slice>({slice_Slice_965, {0}, {int32_max}, {2}, {3}});
    auto Unsqueeze_65525 = makeOP<opset1::Unsqueeze>({slice_Slice_1168, -1});
    auto stack_1182 = makeOP<opset1::Concat>({Unsqueeze_65524, Unsqueeze_65525}, {{"axis", -1}});
    auto flatten_Reshape_1198 =
        makeOP<opset1::Reshape>({stack_1182, {0, 0, num_head, rotary_dims}}, {{"special_zero", true}});
    // x*cos [B,L,H,ndims]
    auto mul_cos = makeOP<opset1::Multiply>({slice_Slice_965, repeat_interleave_cos}, {{"auto_broadcast", "numpy"}});
    mul_cos->set_friendly_name("mul_cos");
    auto mul_sin =
        makeOP<opset1::Multiply>({flatten_Reshape_1198, repeat_interleave_sin}, {{"auto_broadcast", "numpy"}});
    // *cos + *sin
    auto rotary_emb = makeOP<opset1::Add>({mul_cos, mul_sin}, {{"auto_broadcast", "numpy"}});

    auto slice_Slice_971 = makeOP<ov::op::v8::Slice>({input, {rotary_dims}, {int32_max}, {1}, {3}});
    auto cat_Concat_1211 = makeOP<opset1::Concat>({rotary_emb, slice_Slice_971}, {{"axis", -1}});
    auto permute_Transpose_1213 = makeOP<opset1::Transpose>({cat_Concat_1211, {0, 2, 1, 3}});
    ov::NodeVector model_output = {permute_Transpose_1213};
    if (hasShapeOf) {
        auto shapeOf = makeOP<opset1::ShapeOf>({rotary_emb}, {{"output_type", "i32"}});
        auto gather = makeOP<opset8::Gather>({shapeOf, {1}, 0}, {{"batch_dims", 0}});
        model_output.push_back(gather);
    }
    return std::make_shared<ov::Model>(model_output, ov::ParameterVector{input, sincos});
}

std::shared_ptr<ov::Model> RoPETestChatGLM2DRoPEStridedSlice::buildROPE_ChatGLM(int batch, int head_cnt, int rotary_dims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{batch, -1, 4096 + 256 + 256});
    auto cos_sin_cache = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{32768, 32, 2});
    auto position_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{-1, -1});

    auto __module_transformer_index_67_Gather =
        makeOP<opset8::Gather>({cos_sin_cache, position_ids, 0}, {{"batch_dims", 0}});

    auto ListUnpack_321 = makeOP<opset1::VariadicSplit>({input, -1, {4096, 256, 256}});
    auto view_Reshape = makeOP<opset1::Reshape>({ListUnpack_321->output(0), {0, 0, 32, 128}}, {{"special_zero", true}});

    auto permute_Transpose = makeOP<opset1::Transpose>({view_Reshape, {0, 2, 1, 3}}, {});

    auto slice_Slice_357 =
        makeOP<opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 0}, {0, 0, 0, 64}, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});

    auto aten_view_Reshape_1 = makeOP<opset1::Reshape>({ListUnpack_321->output(1), {0, 0, 2, 128}}, {{"special_zero", true}});
    auto aten_transpose_1 = makeOP<opset8::Transpose>({aten_view_Reshape_1, {0, 2, 1, 3}});
    auto shape_of_105249 = makeOP<opset8::ShapeOf>({aten_transpose_1}, {{"output_type", "i32"}});
    auto gather_105252 = makeOP<opset8::Gather>({shape_of_105249, {2}, {0}}, {{"batch_dims", 0}});
    auto scatter_update_63441 = makeOP<opset8::ScatterUpdate>({{0, 0}, {1}, gather_105252, {0}});
    // connected to cos_sin_cache
    auto slice_Slice_369 =
        makeOP<opset1::StridedSlice>({__module_transformer_index_67_Gather, {0, 0}, scatter_update_63441, {1, 1}},
                                     {{"begin_mask", {1, 0}},
                                      {"end_mask", {1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto list_construct_concat_1 = makeOP<opset1::Concat>({{-1}, {1}, gather_105252, {32}, {2}}, {{"axis", 0}});

    auto reshape_Reshape_373 =
        makeOP<opset1::Reshape>({slice_Slice_357, {0, 32, 0, 32, 2}}, {{"special_zero", true}});
    auto select_Gather_384 = makeOP<opset8::Gather>({reshape_Reshape_373, 0, -1}, {{"batch_dims", 0}});//x_even
    auto select_Gather_381 = makeOP<opset8::Gather>({reshape_Reshape_373, 1, -1}, {{"batch_dims", 0}});//x_odd

    auto view_Reshape_380 =
        makeOP<opset1::Reshape>({slice_Slice_369, list_construct_concat_1}, {{"special_zero", false}});
    auto select_Gather_385 = makeOP<opset8::Gather>({view_Reshape_380, 0, -1}, {{"batch_dims", 0}});//cos_tab
    auto select_Gather_382 = makeOP<opset8::Gather>({view_Reshape_380, 1, -1}, {{"batch_dims", 0}});//sin_tab

    auto mul_Multiply_386 =
        makeOP<opset1::Multiply>({select_Gather_381, select_Gather_382}, {{"auto_broadcast", "numpy"}});//x_odd_sin
    auto mul_Multiply_383 =
        makeOP<opset1::Multiply>({select_Gather_384, select_Gather_385}, {{"auto_broadcast", "numpy"}});//x_even_cos
    auto sub_Subtract_389 =
        makeOP<opset1::Subtract>({mul_Multiply_383, mul_Multiply_386}, {{"auto_broadcast", "numpy"}});

    auto mul_Multiply_391 =
        makeOP<opset1::Multiply>({select_Gather_381, select_Gather_385}, {{"auto_broadcast", "numpy"}});//x_odd_cos
    auto mul_Multiply_393 =
        makeOP<opset1::Multiply>({select_Gather_384, select_Gather_382}, {{"auto_broadcast", "numpy"}});//x_even_sin
    auto add_Add_396 = makeOP<opset1::Add>({mul_Multiply_391, mul_Multiply_393}, {{"auto_broadcast", "numpy"}});

    auto Unsqueeze_62716 = makeOP<opset1::Unsqueeze>({sub_Subtract_389, -1}, {});
    auto Unsqueeze_62717 = makeOP<opset1::Unsqueeze>({add_Add_396, -1}, {});

    auto stack_401 = makeOP<opset1::Concat>({Unsqueeze_62716, Unsqueeze_62717}, {{"axis", -1}});
    auto flatten_Reshape_421 = makeOP<opset1::Reshape>({stack_401, {0, 32, 0, 64}}, {{"special_zero", true}});
    auto slice_Slice_363 =
        makeOP<opset1::StridedSlice>({permute_Transpose, {0, 0, 0, 64}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto cat_Concat_425 = makeOP<opset1::Concat>({flatten_Reshape_421, slice_Slice_363}, {{"axis", -1}});
    return std::make_shared<ov::Model>(ov::NodeVector{cat_Concat_425},
                                       ov::ParameterVector{input, cos_sin_cache, position_ids});
}

ov::Tensor RoPETestChatGLM2DRoPEStridedSlice::create_i32_tensor(const ov::Shape& shape, int start, int step) {
    auto tensor = ov::Tensor(ov::element::i32, shape);
    auto* ptr = static_cast<int32_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_size(); i++) {
        ptr[i] = start;
        start += step;
    }
    return tensor;
}

void RoPETestChatGLM2DRoPEStridedSlice::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    const auto& funcInputs = function->inputs();

    auto& input_shape = targetInputStaticShapes[0];
    auto batch = input_shape[0];
    auto seq_length = input_shape[1];

    ov::Tensor t_input = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, 2, -1.0f, 32768);
    ov::Tensor t_cos_sin_cache =
        utils::create_and_fill_tensor(funcInputs[1].get_element_type(), {32768, 32, 2}, 2, -1.0f, 32768);
    ov::Tensor t_position_ids = create_i32_tensor(ov::Shape({batch, seq_length}), 15);

    inputs.clear();
    inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), t_cos_sin_cache});
    inputs.insert({funcInputs[2].get_node_shared_ptr(), t_position_ids});
}

void RoPETestChatGLM2DRoPEStridedSlice::SetUp() {
    targetDevice = this->GetParam();

    const int batch = 2;
    const int seq_length = 7;
    const int num_head = 32;
    const int rotary_dims = 64;

    InputShape inpShape = {{batch, -1, 4096 + 256 + 256}, {{batch, seq_length, 4096 + 256 + 256}}};
    init_input_shapes({inpShape});
    function = buildROPE_ChatGLM(-1, num_head, rotary_dims);
}

std::string RoPETestChatGLM2DRoPEStridedSlice::getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
    std::string targetDevice = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

}  // namespace test
}  // namespace ov
