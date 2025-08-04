// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/interpolate.hpp"

#include "common_test_utils/test_enums.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/interpolate.hpp"

namespace ov {
namespace test {
std::string InterpolateLayerTest::getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj) {
    using ov::test::utils::operator<<;

    InterpolateSpecificParams interpolate_params;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    ov::Shape target_shape;
    std::string target_device;
    std::map<std::string, std::string> additional_config;
    std::tie(interpolate_params, model_type, shapes, target_shape, target_device, additional_config) = obj.param;
    std::vector<size_t> pad_begin, pad_end;
    std::vector<int64_t> axes;
    std::vector<float> scales;
    bool antialias;
    ov::op::v4::Interpolate::InterpolateMode mode;
    ov::op::v4::Interpolate::ShapeCalcMode shape_calc_mode;
    ov::op::v4::Interpolate::CoordinateTransformMode coordinate_transform_mode;
    ov::op::v4::Interpolate::NearestMode nearest_mode;
    double cube_coef;
    std::tie(mode, shape_calc_mode, coordinate_transform_mode, nearest_mode, antialias, pad_begin, pad_end, cube_coef, axes, scales) = interpolate_params;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "TS=" << ov::test::utils::vec2str(target_shape) << "_";
    result << "InterpolateMode=" << mode << "_";
    result << "ShapeCalcMode=" << shape_calc_mode << "_";
    result << "CoordinateTransformMode=" << coordinate_transform_mode << "_";
    result << "NearestMode=" << nearest_mode << "_";
    result << "cube_coef=" << cube_coef << "_";
    result << "Antialias=" << antialias << "_";
    result << "PB=" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE=" << ov::test::utils::vec2str(pad_end) << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "Scales=" << ov::test::utils::vec2str(scales) << "_";
    result << "netType=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void InterpolateLayerTest::SetUp() {
    InterpolateSpecificParams interpolate_params;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    ov::Shape target_shape;
    std::map<std::string, std::string> additional_config;
    std::tie(interpolate_params, model_type, shapes, target_shape, targetDevice, additional_config) = this->GetParam();
    std::vector<size_t> pad_begin, pad_end;
    std::vector<int64_t> axes;
    std::vector<float> scales;
    bool antialias;
    ov::op::v4::Interpolate::InterpolateMode mode;
    ov::op::v4::Interpolate::ShapeCalcMode shape_calc_mode;
    ov::op::v4::Interpolate::CoordinateTransformMode coordinate_transform_mode;
    ov::op::v4::Interpolate::NearestMode nearest_mode;

    configuration.insert(additional_config.begin(), additional_config.end());

    double cube_coef;
    std::tie(mode, shape_calc_mode, coordinate_transform_mode, nearest_mode, antialias, pad_begin, pad_end, cube_coef, axes, scales) = interpolate_params;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto sizes_input = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);

    auto scales_input = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{scales.size()}, scales);

    ov::op::v4::Interpolate::InterpolateAttrs interpolate_attributes{mode, shape_calc_mode, pad_begin,
        pad_end, coordinate_transform_mode, nearest_mode, antialias, cube_coef};

    std::shared_ptr<ov::op::v4::Interpolate> interpolate;
    if (axes.empty()) {
        interpolate = std::make_shared<ov::op::v4::Interpolate>(param,
                                                                sizes_input,
                                                                scales_input,
                                                                interpolate_attributes);
    } else {
        auto axesInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);

        interpolate = std::make_shared<ov::op::v4::Interpolate>(param,
                                                                sizes_input,
                                                                scales_input,
                                                                axesInput,
                                                                interpolate_attributes);
    }
    auto result = std::make_shared<ov::op::v0::Result>(interpolate);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "interpolate");

    if (model_type == ov::element::f32) {
        abs_threshold = 1e-6;
    }
}

std::string Interpolate11LayerTest::getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj) {
    return InterpolateLayerTest::getTestCaseName(obj);
}

namespace {
std::shared_ptr<ov::op::v0::Constant> make_scales_or_sizes_input(ov::op::util::InterpolateBase::ShapeCalcMode shape_calc_mode,
                                                                        const std::vector<size_t>& sizes,
                                                                        const std::vector<float>& scales) {
    if (shape_calc_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SIZES)
        return std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{sizes.size()}, sizes);
    else
        return std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{scales.size()}, scales);
}
}
void Interpolate11LayerTest::SetUp() {
    InterpolateSpecificParams interpolate_params;
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    ov::Shape target_shape;
    std::map<std::string, std::string> additional_config;
    std::tie(interpolate_params, model_type, shapes, target_shape, targetDevice, additional_config) = this->GetParam();
    std::vector<size_t> pad_begin, pad_end;
    std::vector<int64_t> axes;
    std::vector<float> scales;
    bool antialias;
    ov::op::v4::Interpolate::InterpolateMode mode;
    ov::op::v4::Interpolate::ShapeCalcMode shape_calc_mode;
    ov::op::v4::Interpolate::CoordinateTransformMode coordinate_transform_mode;
    ov::op::v4::Interpolate::NearestMode nearest_mode;

    configuration.insert(additional_config.begin(), additional_config.end());

    double cube_coef;
    std::tie(mode, shape_calc_mode, coordinate_transform_mode, nearest_mode, antialias, pad_begin, pad_end, cube_coef, axes, scales) = interpolate_params;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto scales_orsizes_input = make_scales_or_sizes_input(shape_calc_mode, target_shape, scales);

    ov::op::util::InterpolateBase::InterpolateAttrs interpolate_attributes{mode, shape_calc_mode, pad_begin,
        pad_end, coordinate_transform_mode, nearest_mode, antialias, cube_coef};

    std::shared_ptr<ov::op::v11::Interpolate> interpolate{};
    if (axes.empty()) {
        interpolate = std::make_shared<ov::op::v11::Interpolate>(param,
                                                                     scales_orsizes_input,
                                                                     interpolate_attributes);
    } else {
        auto axesInput = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);

        interpolate = std::make_shared<ov::op::v11::Interpolate>(param,
                                                                     scales_orsizes_input,
                                                                     axesInput,
                                                                     interpolate_attributes);
    }

    auto result = std::make_shared<ov::op::v0::Result>(interpolate);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "interpolate");
}
}  // namespace test
}  // namespace ov
