// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <string>
#include <tuple>

#include "common_test_utils/common_utils.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace CPUTestUtils;
using namespace ov::gen_pattern;
using namespace ov;

namespace ov {
namespace test {

static std::shared_ptr<ov::Model> buildCausalMaskPreprocess(const int max_seq_len) {
    std::vector<int32_t> triu(max_seq_len * max_seq_len, 1);
    auto* ptr = triu.data();
    for (int y = 0; y < max_seq_len; y++, ptr += max_seq_len) {
        for (int x = 0; x <= y; x++)
            ptr[x] = 0;
    }
    auto const_triu = makeConst(ov::element::i32,
                                ov::Shape({1, 1, static_cast<size_t>(max_seq_len), static_cast<size_t>(max_seq_len)}),
                                triu);
    auto attention_mask = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{-1, -1});
    auto batch_size = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
    auto cache_positions = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto kvLen = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape{1});

    auto ShapeOf_i32 = [](std::shared_ptr<ov::Node> data) {
        return makeOP<ov::opset3::ShapeOf>({data}, {{"output_type", "i32"}});
    };

    auto ListConstruct_Concat =
        makeOP<ov::opset1::Concat>({batch_size, {1}, {1}, {1}}, {{"axis", 0}});  //  tensor_array<i32[4]>
    auto repeat_Tile =
        makeOP<ov::opset1::Tile>({const_triu, ListConstruct_Concat});  //  tensor_array<u8[?,1,8192,8192]>
    auto to_Convert =
        makeOP<ov::opset1::Convert>({repeat_Tile}, {{"destination_type", "f32"}});  //  tensor_array<f32[?,1,8192,8192]>
    auto Constant_107277 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-FLT_MAX});
    auto mul_Multiply_1 =
        makeOP<ov::opset1::Multiply>({to_Convert, Constant_107277},
                                     {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,1,8192,8192]>
    auto SliceAssign_201_Reshape_0 =
        makeOP<ov::opset1::Reshape>({mul_Multiply_1, {-1}}, {{"special_zero", false}});  //  tensor_array<f32[?]>
    auto SliceAssign_201_ShapeOf = ShapeOf_i32(mul_Multiply_1);                          //  tensor_array<i32[4]>
    auto SliceAssign_201_ReduceProd =
        makeOP<ov::opset1::ReduceProd>({SliceAssign_201_ShapeOf, 0}, {{"keep_dims", false}});  //  tensor_array<i32[]>
    auto SliceAssign_201_Range = makeOP<ov::opset4::Range>({0, SliceAssign_201_ReduceProd, 1},
                                                           {{"output_type", "i32"}});  //  tensor_array<i32[?]>
    auto SliceAssign_201_Reshape =
        makeOP<ov::opset1::Reshape>({SliceAssign_201_Range, {-1, 1, max_seq_len, max_seq_len}},
                                    {{"special_zero", true}});  //  tensor_array<i32[?,1,8192,8192]>
    auto ShapeOf_49034 = ShapeOf_i32(attention_mask);           //  tensor_array<i32[2]>
    auto Gather_41642 =
        makeOP<ov::opset8::Gather>({ShapeOf_49034, {1}, 0}, {{"batch_dims", 0}});  //  tensor_array<i32[1]>
    auto ScatterUpdate_93502 =
        makeOP<ov::opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, Gather_41642, {0}});  //  tensor_array<i32[4]>
    auto SliceAssign_201_Slice =
        makeOP<ov::opset1::StridedSlice>({SliceAssign_201_Reshape, {0, 0, 0, 0}, ScatterUpdate_93502, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});  //  tensor_array<i32[?,1,8192,..8192]>
    auto SliceAssign_201_Reshape_1 = makeOP<ov::opset1::Reshape>({SliceAssign_201_Slice, {-1, 1}},
                                                                 {{"special_zero", false}});  //  tensor_array<i32[?,1]>
    auto causal_mask_boolean_1 =
        makeOP<ov::opset1::StridedSlice>({mul_Multiply_1, {0, 0, 0, 0}, ScatterUpdate_93502, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});  //  tensor_array<f32[?,1,8192,..8192]>
    auto Constant_107278 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {0.000000f});
    auto eq_Equal = makeOP<ov::opset1::Equal>({causal_mask_boolean_1, Constant_107278},
                                              {{"auto_broadcast", "numpy"}});  //  tensor_array<u8[?,1,8192,..8192]>
    auto unsqueeze_Unsqueeze_1 =
        makeOP<ov::opset1::Unsqueeze>({attention_mask, {1, 2}});  //  tensor_array<i32[?,1,1,?]>
    auto eq_Convert = makeOP<ov::opset1::Convert>({unsqueeze_Unsqueeze_1},
                                                  {{"destination_type", "f32"}});  //  tensor_array<f32[?,1,1,?]>
    auto Constant_107279 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {0.000000f});
    auto eq_Equal_1 = makeOP<ov::opset1::Equal>({eq_Convert, Constant_107279},
                                                {{"auto_broadcast", "numpy"}});  //  tensor_array<u8[?,1,1,?]>
    auto mul_LogicalAnd =
        makeOP<ov::opset1::LogicalAnd>({eq_Equal, eq_Equal_1},
                                       {{"auto_broadcast", "numpy"}});  //  tensor_array<u8[?,1,8192,?]>
    auto masked_fill_Select =
        makeOP<ov::opset1::Select>({mul_LogicalAnd, -FLT_MAX, causal_mask_boolean_1},
                                   {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,1,8192,?]>
    auto copy__ShapeOf = ShapeOf_i32(causal_mask_boolean_1);        //  tensor_array<i32[4]>
    auto Constant_47319 = makeConst(ov::element::u8, ov::Shape({}), {0});
    auto copy__Broadcast = makeOP<ov::opset1::Broadcast>({masked_fill_Select, copy__ShapeOf, Constant_47319},
                                                         {{"mode", "numpy"}});  //  tensor_array<f32[?,1,8192,..8192]>
    auto SliceAssign_201_Reshape_2 =
        makeOP<ov::opset1::Reshape>({copy__Broadcast, {-1}}, {{"special_zero", false}});  //  tensor_array<f32[?]>
    auto SliceAssign_201_ScatterNDUpdate = makeOP<ov::opset4::ScatterNDUpdate>(
        {SliceAssign_201_Reshape_0, SliceAssign_201_Reshape_1, SliceAssign_201_Reshape_2},
        {});  //  tensor_array<f32[?]>
    auto SliceAssign_201_Reshape_3 =
        makeOP<ov::opset1::Reshape>({SliceAssign_201_ScatterNDUpdate, {-1, 1, max_seq_len, max_seq_len}},
                                    {{"special_zero", true}});  //  tensor_array<f32[?,1,8192,8192]>
    auto ScatterUpdate_93554 =
        makeOP<ov::opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, kvLen, {0}}, {});  //  tensor_array<i32[4]>
    auto slice_Slice_14 =
        makeOP<ov::opset1::StridedSlice>({SliceAssign_201_Reshape_3, {0, 0, 0, 0}, ScatterUpdate_93554, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});  //  tensor_array<f32[?,1,8192,..8192]>
    auto index_Gather = makeOP<ov::opset8::Gather>({slice_Slice_14, cache_positions, 2},
                                                   {{"batch_dims", 0}});  //  tensor_array<f32[?,1,?,..8192]>
    auto result = index_Gather;

    return std::make_shared<ov::Model>(ov::NodeVector{result},
                                       ov::ParameterVector{attention_mask, batch_size, cache_positions, kvLen});
}

class CausalMaskPreprocessCausalMaskPreprocess : public SubgraphBaseTest {
public:
    ov::Tensor create_i32_tensor(const ov::Shape& shape, int start, int step = 1) {
        auto tensor = ov::Tensor(ov::element::i32, shape);
        auto* ptr = static_cast<int32_t*>(tensor.data());
        for (size_t i = 0; i < tensor.get_size(); i++) {
            ptr[i] = start;
            start += step;
        }
        return tensor;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto& funcInputs = function->inputs();

        auto& sp_attention_mask = targetInputStaticShapes[0];
        auto& sp_cache_positions = targetInputStaticShapes[2];
        auto batch = sp_attention_mask[0];
        auto kvLen = sp_attention_mask[1];
        auto qLen = sp_cache_positions[0];

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 32768;
        ov::Tensor t_attention_mask = ov::Tensor(ov::element::i32, sp_attention_mask);
        auto* ptr = static_cast<int32_t*>(t_attention_mask.data());
        for (size_t n = 0; n < batch; n++, ptr += kvLen) {
            for (size_t i = n*4; i < kvLen; i++) {
                ptr[i] = 1;
            }
        }
        ov::Tensor t_batch_size = create_i32_tensor(ov::Shape({1}), batch, 0);
        ov::Tensor t_cache_positions = create_i32_tensor(ov::Shape({qLen}), 0, 1);
        ov::Tensor t_kvLen = create_i32_tensor(ov::Shape({1}), kvLen, 0);

        inputs.clear();
        inputs.insert({funcInputs[0].get_node_shared_ptr(), t_attention_mask});
        inputs.insert({funcInputs[1].get_node_shared_ptr(), t_batch_size});
        inputs.insert({funcInputs[2].get_node_shared_ptr(), t_cache_positions});
        inputs.insert({funcInputs[3].get_node_shared_ptr(), t_kvLen});
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const size_t batch = 2;
        const size_t qLen = 16;
        const size_t kvLen = 16;
        const size_t max_seq_len = 2048;

        InputShape sp_attention_mask = {{-1, -1}, {{batch, kvLen}}};
        InputShape sp_batch_size = {{1}, {{1}}};
        InputShape sp_cache_positions = {{-1}, {{qLen}}};
        InputShape sp_kvLen = {{1}, {{1}}};
        init_input_shapes({sp_attention_mask, sp_batch_size, sp_cache_positions, sp_kvLen});
        function = buildCausalMaskPreprocess(max_seq_len);
    }
};

TEST_F(CausalMaskPreprocessCausalMaskPreprocess, smoke_CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "CausalMaskPreprocess", 1);
}

}  // namespace test
}  // namespace ov