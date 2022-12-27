// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_model_to_fp16_element_type.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <transformations/convert_precision.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, ConvertModelToFP16ElementType) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto convert_ins1 = std::make_shared<ov::opset8::Convert>(const_weights, ov::element::f32);
        ov::mark_as_decompression(convert_ins1);
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              convert_ins1,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertModelToFP16ElementType>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f16,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertModelToFP16ElementTypeNoConvertion) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ConvertModelToFP16ElementType>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 12, 12});
        auto const_weights = ov::opset8::Constant::create(
            ov::element::f32,
            ov::Shape{1, 3, 3, 3},
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto conv = std::make_shared<ov::opset8::Convolution>(input,
                                                              const_weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

template <element::Type_t T>
bool has_type(std::shared_ptr<Model> model) {
    for (auto& node : model->get_ordered_ops()) {
        for (auto& input : node->inputs()) {
            if (input.get_element_type() == element::Type(T)) {
                return true;
            }
        }
        for (auto& output : node->outputs()) {
            if (output.get_element_type() == element::Type(T)) {
                return true;
            }
        }
    }
    return false;
}

TEST(TransformationTests, ConvertModelToFP16ElementType_keep_sensitive_nodes_in_fp32) {
    // with keep_precision_sensitive_in_fp32 = true should keep precision sensitive nodes in FP32
    std::shared_ptr<ov::Model> model(nullptr), model_ref(nullptr);
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f32, Shape{360, 640});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f32, Shape{720, 1280});
        auto shapeof = std::make_shared<opset8::ShapeOf>(input_2);

        // decompression Converts are needed for ConvertModelToFP16ElementType to be triggered
        auto compressed_const = opset8::Constant::create(element::f16, Shape{}, {2.0f});
        auto decompress_convert = std::make_shared<opset8::Convert>(compressed_const, element::f32);
        mark_as_decompression(decompress_convert);
        auto add_decompressed_const = std::make_shared<opset8::Add>(input_1, decompress_convert);

        auto convert_to_float = std::make_shared<opset8::Convert>(shapeof, element::f32);
        auto const_denominator = opset8::Constant::create(element::f32, Shape{}, {2.0f});
        auto div = std::make_shared<opset8::Divide>(convert_to_float, const_denominator);
        auto new_shape = std::make_shared<opset8::Convert>(div, element::i64);

        auto reshape = std::make_shared<opset8::Reshape>(add_decompressed_const, new_shape, false);
        model = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});

        pass::Manager manager;
        bool keep_precision_sensitive_in_fp32 = true;
        manager.register_pass<ov::pass::ConvertModelToFP16ElementType>(keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }
    {
        auto input_1 = std::make_shared<opset8::Parameter>(element::f16, Shape{360, 640});
        auto input_2 = std::make_shared<opset8::Parameter>(element::f16, Shape{720, 1280});
        auto shapeof_1 = std::make_shared<opset8::ShapeOf>(input_2);

        // after ConvertModelToFP16ElementType Const->Convert are constant-folded into a single f16 Const
        auto compressed_const = opset8::Constant::create(element::f16, Shape{}, {2.0f});
        auto add_compressed_const = std::make_shared<opset8::Add>(input_1, compressed_const);

        // shape subgraph will be constant folded
        auto new_shape_const = opset8::Constant::create(element::i64, Shape{2}, {360, 640});

        auto reshape = std::make_shared<opset8::Reshape>(add_compressed_const, new_shape_const, false);
        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input_1, input_2});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConvertModelToFP16ElementType_do_not_keep_in_fp32) {
    // negative test: check that without keeping sensitive nodes in FP32 the whole ov::Model is converted to f16
    // including ShapeOf subgraph and we get wrong output shape [1, 3, 287, 511] instead of correct one [1, 3, 288, 512]
    std::shared_ptr<ov::Model> model(nullptr);
    std::shared_ptr<opset8::Interpolate> interpolate(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 720, 1280});
        auto sizes = opset8::Constant::create(element::i64, Shape{4}, {1, 3, 288, 512});
        auto scales_const = opset8::Constant::create(element::f32, Shape{4}, {1.0f, 1.0f, 0.4f, 0.4f});

        // decompression Converts are needed for ConvertModelToFP16ElementType to be triggered
        auto compressed_const = opset8::Constant::create(element::f16, Shape{}, {0.0f});
        auto decompress_convert = std::make_shared<opset8::Convert>(compressed_const, element::f32);
        mark_as_decompression(decompress_convert);
        // scales still are {1., 1., 0.4, 0.4}
        auto scales = std::make_shared<opset8::Add>(scales_const, decompress_convert);

        opset8::Interpolate::InterpolateAttrs attrs;

        attrs.mode = opset8::Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = opset8::Interpolate::ShapeCalcMode::SCALES;
        attrs.nearest_mode = opset8::Interpolate::NearestMode::FLOOR;
        attrs.pads_begin = std::vector<size_t>{0};
        attrs.pads_end = std::vector<size_t>{0};
        attrs.antialias = false;
        attrs.coordinate_transformation_mode = opset8::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
        attrs.cube_coeff = -0.75f;

        interpolate = std::make_shared<opset8::Interpolate>(input, sizes, scales, attrs);
        model = std::make_shared<ov::Model>(NodeVector{interpolate}, ParameterVector{input});

        pass::Manager manager;
        // didn't keep in FP32 intentionally
        bool keep_precision_sensitive_in_fp32 = false;
        manager.register_pass<ov::pass::ConvertModelToFP16ElementType>(keep_precision_sensitive_in_fp32);
        manager.run_passes(model);
    }

    ASSERT_FALSE(has_type<element::Type_t::f32>(model));
    ASSERT_TRUE(interpolate->input_value(2).get_element_type() == element::Type_t::f16);
    ASSERT_TRUE(interpolate->output(0).get_partial_shape() == ov::PartialShape({1, 3, 287, 511}));
}
