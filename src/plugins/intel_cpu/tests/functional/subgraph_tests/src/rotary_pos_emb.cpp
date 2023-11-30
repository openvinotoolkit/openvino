// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <debug.h>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ov_models/builders.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <string>
#include <tuple>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ie_precision.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "utils/gen_pattern.hpp"

using namespace CPUTestUtils;
using namespace ov::gen_pattern;
using namespace ov;

namespace ov {
namespace test {

static ov::OutputVector makeCosSinCache(int max_position_embeddings, int rotary_ndims) {
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

static std::shared_ptr<ov::Model> buildROPE_Llama2(const int batch,
                                                   const int seq_length,
                                                   const int max_position_embeddings,
                                                   const int num_head,
                                                   const int ndims) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, PartialShape{batch, -1, num_head, ndims});
    auto pos_id_end = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{});
    auto pos_ids = std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape{1, -1});

    auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);
    auto Constant582 = cos_sin_cache[0];
    auto Constant585 = cos_sin_cache[1];

    // concat KV length
    auto transpose_Transpose = makeOP<opset1::Transpose>({input, {0, 2, 1, 3}});
    auto slice_Unsqueeze_426 = makeOP<opset1::Unsqueeze>({pos_id_end, 0});
    auto ScatterUpdate_152236 = makeOP<opset3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_426, {0}});
    auto slice_Slice = makeOP<opset1::StridedSlice>({Constant582, {0, 0, 0}, ScatterUpdate_152236, {1, 1, 1}},
                                                    {{"begin_mask", {1, 1, 0}},
                                                     {"end_mask", {1, 1, 0}},
                                                     {"new_axis_mask", {}},
                                                     {"shrink_axis_mask", {}},
                                                     {"ellipsis_mask", {}}});
    auto squeeze_Squeeze = makeOP<opset1::Squeeze>({slice_Slice, 1});
    auto squeeze_Squeeze_435 = makeOP<opset1::Squeeze>({squeeze_Squeeze, 0});
    auto index_441_Gather = makeOP<opset8::Gather>({squeeze_Squeeze_435, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze = makeOP<opset1::Unsqueeze>({index_441_Gather, 1});
    auto mul_Multiply =
        makeOP<opset1::Multiply>({transpose_Transpose, unsqueeze_Unsqueeze}, {{"auto_broadcast", "numpy"}});
    auto size_ShapeOf_448 = makeOP<opset3::ShapeOf>({transpose_Transpose}, {{"output_type", "i32"}});
    auto size_Gather_450 = makeOP<opset8::Gather>({size_ShapeOf_448, 3, 0}, {{"batch_dims", 0}});
    auto floor_divide_Divide =
        makeOP<opset1::Divide>({size_Gather_450, 2}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto floor_divide_Floor = makeOP<opset1::Floor>({floor_divide_Divide});
    auto slice_Unsqueeze_452 = makeOP<opset1::Unsqueeze>({floor_divide_Floor, 0});
    auto ScatterUpdate_152312 = makeOP<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_452, {0}});
    auto slice_Slice_459 = makeOP<opset1::StridedSlice>(
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
    auto neg_Multiply = makeOP<opset1::Multiply>({slice_Slice_459, Constant_182988}, {{"auto_broadcast", "numpy"}});
    auto ScatterUpdate_152368 = makeOP<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_452, {0}});
    auto slice_Slice2 =
        makeOP<opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 0}, ScatterUpdate_152368, {1, 1, 1, 1}},
                                     {{"begin_mask", {1, 1, 1, 0}},
                                      {"end_mask", {1, 1, 1, 0}},
                                      {"new_axis_mask", {}},
                                      {"shrink_axis_mask", {}},
                                      {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<opset1::Concat>({neg_Multiply, slice_Slice2}, {{"axis", -1}});
    auto ScatterUpdate_152421 = makeOP<opset3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_426, {0}});
    auto slice_Slice_433 = makeOP<opset1::StridedSlice>({Constant585, {0, 0, 0}, ScatterUpdate_152421, {1, 1, 1}},
                                                        {{"begin_mask", {1, 1, 0}},
                                                         {"end_mask", {1, 1, 0}},
                                                         {"new_axis_mask", {}},
                                                         {"shrink_axis_mask", {}},
                                                         {"ellipsis_mask", {}}});
    auto squeeze_Squeeze_436 = makeOP<opset1::Squeeze>({slice_Slice_433, 1});
    auto squeeze_Squeeze_437 = makeOP<opset1::Squeeze>({squeeze_Squeeze_436, 0});
    auto index_446_Gather = makeOP<opset8::Gather>({squeeze_Squeeze_437, pos_ids, 0}, {{"batch_dims", 0}});
    auto unsqueeze_Unsqueeze_447 = makeOP<opset1::Unsqueeze>({index_446_Gather, 1});
    auto mul_Multiply_463 =
        makeOP<opset1::Multiply>({cat_Concat, unsqueeze_Unsqueeze_447}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<opset1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::NodeVector{add_Add}, ov::ParameterVector{input, pos_id_end, pos_ids});
}

class RoPECPUTest : public SubgraphBaseTest {
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

        const int position_id_start = 15;
        auto& input_shape = targetInputStaticShapes[0];
        auto seq_length = input_shape[1];

        ov::Tensor t_input =
            utils::create_and_fill_tensor(funcInputs[0].get_element_type(), input_shape, 2, -1.0f, 32768);
        ov::Tensor t_position_id_end = create_i32_tensor(ov::Shape({}), position_id_start + seq_length);
        ov::Tensor t_position_ids = create_i32_tensor(ov::Shape({1, seq_length}), position_id_start);

        inputs.clear();
        inputs.insert({funcInputs[0].get_node_shared_ptr(), t_input});
        inputs.insert({funcInputs[1].get_node_shared_ptr(), t_position_id_end});
        inputs.insert({funcInputs[2].get_node_shared_ptr(), t_position_ids});
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const int batch = 2;
        const int seq_length = 7;
        const size_t max_position_embeddings = 2048;
        const size_t ndims = 128;
        const size_t num_head = 32;

        InputShape inpShape = {{batch, seq_length, num_head, ndims}, {{batch, seq_length, num_head, ndims}}};
        init_input_shapes({inpShape});
        function = buildROPE_Llama2(batch, seq_length, max_position_embeddings, num_head, ndims);
    }
};

TEST_F(RoPECPUTest, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
