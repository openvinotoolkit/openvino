// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/subgraph/msda_pattern.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>
#include <transformations/utils/print_model.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/msda.hpp"
#include "transformations/common_optimizations/multi_scale_deformable_attn_fusion.hpp"

namespace ov {
namespace test {

using namespace ov::opset10;

std::shared_ptr<ov::Node> build_grid_sample_block(const std::shared_ptr<ov::Node>& attn_Reshape,
                                                  const std::shared_ptr<ov::Node>& attn_Sub,
                                                  std::initializer_list<int> slice_start,
                                                  std::initializer_list<int> slice_end,
                                                  std::initializer_list<int> scale_size,
                                                  std::initializer_list<int> sub_indices,
                                                  std::initializer_list<int> sub_axis) {
    using namespace ov::opset10;

    auto Constant_124164 = Constant::create(element::i64, Shape{slice_start.size()}, slice_start);
    auto Constant_125832 = Constant::create(element::i64, Shape{slice_end.size()}, slice_end);
    auto Constant_124170 = Constant::create(element::i64, Shape{2}, {1, 1});
    auto attn_Slice = std::make_shared<StridedSlice>(attn_Reshape,
                                                     Constant_124164,
                                                     Constant_125832,
                                                     Constant_124170,
                                                     std::vector<int64_t>{1, 0},
                                                     std::vector<int64_t>{1, 0});
    auto attn_Reshape_4 =
        std::make_shared<Reshape>(attn_Slice, Constant::create(element::i64, Shape{3}, {0, 0, 256}), true);
    auto attn_Transpose =
        std::make_shared<Transpose>(attn_Reshape_4, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    // [?,256,16700] [?,256,4200] [?,256,1050] [?,256,273] =>
    // {-1,32,100,167}  {-1,32,50,84} {-1,32,25,42} {-1,32,13,21}
    auto attn_Reshape_5 =
        std::make_shared<Reshape>(attn_Transpose,
                                  Constant::create(element::i64, Shape{scale_size.size()}, scale_size),
                                  true);

    // [1,22223,8,1,4,2] ?? [1,22223,8,4,2] ??
    auto attn_Gather_9 =
        std::make_shared<Gather>(attn_Sub,
                                 Constant::create(element::i64, Shape{sub_indices.size()}, sub_indices),
                                 Constant::create(element::i64, Shape{sub_axis.size()}, sub_axis),
                                 0);
    auto attn_squeeze =
        std::make_shared<Squeeze>(attn_Gather_9, Constant::create(element::i64, Shape{1}, sub_axis));  // FIXME???
    auto attn_Transpose_1 =
        std::make_shared<Transpose>(attn_squeeze, Constant::create(element::i64, Shape{5}, {0, 2, 1, 3, 4}));
    auto attn_Reshape_6 =
        std::make_shared<Reshape>(attn_Transpose_1, Constant::create(element::i64, Shape{4}, {-1, 22223, 4, 2}), true);
    GridSample::Attributes attributes = {false,
                                         GridSample::InterpolationMode::BILINEAR,
                                         GridSample::PaddingMode::ZEROS};
    auto attn_GridSample = std::make_shared<GridSample>(attn_Reshape_5, attn_Reshape_6, attributes);
    auto attn_Unsqueeze_31 = std::make_shared<Reshape>(attn_GridSample,
                                                       Constant::create(element::i64, Shape{5}, {-1, 32, 22223, 1, 4}),
                                                       false);

    return attn_Unsqueeze_31;
}

std::shared_ptr<ov::Node> build_concated_grid_samplers(const std::shared_ptr<ov::Node>& attn_Reshape,
                                                       const std::shared_ptr<ov::Node>& attn_Sub) {
    using namespace ov::opset10;

    // 0
    auto attn_Unsqueeze_31 =
        build_grid_sample_block(attn_Reshape, attn_Sub, {0, 0}, {0, 16700}, {-1, 32, 100, 167}, {0}, {3});

    // 1
    auto attn_Unsqueeze_32 =
        build_grid_sample_block(attn_Reshape, attn_Sub, {0, 16700}, {0, 20900}, {-1, 32, 50, 84}, {1}, {3});

    // 2
    auto attn_Unsqueeze_33 =
        build_grid_sample_block(attn_Reshape, attn_Sub, {0, 20900}, {0, 21950}, {-1, 32, 25, 42}, {2}, {3});

    // 3
    auto attn_Unsqueeze_34 =
        build_grid_sample_block(attn_Reshape, attn_Sub, {0, 21950}, {0, 22223}, {-1, 32, 13, 21}, {3}, {3});

    auto attn_Concat_17 = std::make_shared<Concat>(
        ov::NodeVector{attn_Unsqueeze_31, attn_Unsqueeze_32, attn_Unsqueeze_33, attn_Unsqueeze_34},
        -2);  //  tensor_array<f16[?,32,22223,4,4]>
              //  /encoder/layers.1/self_attn/Concat_17(/encoder/layers.1/self_attn/Unsqueeze_31,
              //  /encoder/layers.1/self_attn/Unsqueeze_32, /encoder/layers.1/self_attn/Unsqueeze_33,
              //  /encoder/layers.1/self_attn/Unsqueeze_34)
    auto attn_Reshape_17 = std::make_shared<Reshape>(
        attn_Concat_17,
        Constant::create(element::i64, Shape{4}, {0, 32, 0, 16}),
        true);  //  tensor_array<f16[?,32,22223,16]>
                //  /encoder/layers.1/self_attn/Reshape_17(/encoder/layers.1/self_attn/Concat_17, Constant_650341)

    return attn_Reshape_17;
}

std::shared_ptr<ov::Node> build_attn_aggregate(const std::shared_ptr<ov::Node>& input_attn_weight,
                                               const std::shared_ptr<ov::Node>& grid_sample) {
    using namespace ov::opset10;
    auto attn_Transpose_8 = std::make_shared<Transpose>(
        input_attn_weight,
        Constant::create(
            element::i64,
            Shape{5},
            {0, 2, 1, 3, 4}));  //  tensor_array<f16[?,8,22223,4,4]>
                                //  /encoder/layers.1/self_attn/Transpose_8(/encoder/layers.1/self_attn/Reshape_3,
                                //  Constant_51230)
    auto attn_Reshape_16 = std::make_shared<Reshape>(
        attn_Transpose_8,
        Constant::create(element::i64, Shape{4}, {-1, 1, 0, 16}),
        true);  //  tensor_array<f16[?,1,22223,16]>
                //  /encoder/layers.1/self_attn/Reshape_16(/encoder/layers.1/self_attn/Transpose_8, Constant_650344)
    auto attn_Mul_3 = std::make_shared<Multiply>(
        grid_sample,
        attn_Reshape_16,
        "numpy");  //  tensor_array<f16[?,32,22223,16]>
                   //  /encoder/layers.1/self_attn/Mul_3(/encoder/layers.1/self_attn/Reshape_17,
                   //  /encoder/layers.1/self_attn/Reshape_16)
    auto attn_ReduceSum = std::make_shared<ReduceSum>(
        attn_Mul_3,
        Constant::create(element::i64, Shape{1}, {-1}),
        false);  //  tensor_array<f16[?,32,22223]>
                 //  /encoder/layers.1/self_attn/ReduceSum(/encoder/layers.1/self_attn/Mul_3, Constant_62256)
    auto attn_Reshape_18 = std::make_shared<Reshape>(
        attn_ReduceSum,
        Constant::create(element::i64, Shape{3}, {-1, 256, 0}),
        true);  //  tensor_array<f16[?,256,22223]>
                //  /encoder/layers.1/self_attn/Reshape_18(/encoder/layers.1/self_attn/ReduceSum, Constant_650345)
    auto attn_output_proj_MatMul_transpose_a = std::make_shared<Transpose>(
        attn_Reshape_18,
        Constant::create(
            element::i64,
            Shape{3},
            {0,
             2,
             1}));  //  tensor_array<f16[?,22223,256]>
                    //  /encoder/layers.1/self_attn/output_proj/MatMul/transpose_a(/encoder/layers.1/self_attn/Reshape_18,
                    //  Constant_48800)
    return attn_output_proj_MatMul_transpose_a;
}

std::shared_ptr<ov::Model> build_model_msda(ov::PartialShape value_shape,
                                            ov::PartialShape offset_shape,
                                            ov::PartialShape weight_shape) {
    using namespace ov::opset10;

    auto input_attn_value = std::make_shared<Parameter>(element::f32, value_shape);
    auto input_attn_offsets = std::make_shared<Parameter>(element::f32, offset_shape);
    auto input_attn_weight = std::make_shared<Parameter>(element::f32, weight_shape);

    auto grid_sample_1 = build_concated_grid_samplers(input_attn_value, input_attn_offsets);
    auto attn_Transpose_1 = build_attn_aggregate(input_attn_weight, grid_sample_1);
    grid_sample_1->set_friendly_name("grid_sample_1");
    attn_Transpose_1->set_friendly_name("attn_output_proj_MatMul_transpose_a_1");

    auto grid_sample_2 = build_concated_grid_samplers(input_attn_value, input_attn_offsets);
    auto attn_Transpose_2 = build_attn_aggregate(input_attn_weight, grid_sample_2);
    grid_sample_2->set_friendly_name("grid_sample_2");
    attn_Transpose_2->set_friendly_name("attn_output_proj_MatMul_transpose_a_2");

    auto attn_output = std::make_shared<Add>(attn_Transpose_1, attn_Transpose_2);
    attn_output->set_friendly_name("attn_output");

    return std::make_shared<ov::Model>(NodeVector{attn_output},
                                       ParameterVector{input_attn_value, input_attn_offsets, input_attn_weight});
}

std::string MSDAPattern::getTestCaseName(testing::TestParamInfo<MSDAPatternShapeParams> obj) {
    MSDAPatternShapeParams shape_params;
    std::tie(shape_params) = obj.param;
    std::ostringstream results;

    results << "input=" << shape_params.value_shape << "_";
    results << "offset=" << shape_params.offset_shape << "_";
    results << "weights=" << shape_params.weight_shape;
    return results.str();
}

void MSDAPattern::SetUp() {
    // MSDAPatternShapeParams shape_params;
    // shape_params = GetParam();
    ov::PartialShape value_shape = PartialShape{-1, 22223, 8, 32};
    ov::PartialShape offset_shape = PartialShape{-1, 22223, 8, 4, 4, 2};
    ov::PartialShape weight_shape = PartialShape{-1, 22223, 8, 4, 4};
    InputShape input_value_shape = {value_shape, {Shape{1, 22223, 8, 32}}};
    InputShape input_offset_shape = {offset_shape, {Shape{1, 22223, 8, 4, 4, 2}}};
    InputShape input_weight_shape = {weight_shape, {Shape{1, 22223, 8, 4, 4}}};
    init_input_shapes({input_value_shape, input_offset_shape, input_weight_shape});
    targetDevice =
        ov::test::utils::DEVICE_GPU;
    function = build_model_msda(value_shape, offset_shape, weight_shape);
    // functionRefs = build_model_msda(shape_params);
    // function = functionRefs->clone();
    // ov::pass::Manager manager;

    // manager.register_pass<ov::pass::MultiScaleDeformableAttnFusion>();
    // manager.run_passes(function);
}

}  // namespace test
}  // namespace ov
