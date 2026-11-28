// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/normalize_interpolate_rank3.hpp"

#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {
using Interpolate = ov::op::util::InterpolateBase;
using Constant = ov::op::v0::Constant;

struct Config {
    std::string name;
    bool v11 = false;
    ov::PartialShape shape{2, 18, 101};
    Interpolate::InterpolateAttrs attrs;
    std::vector<int64_t> axes{2};
    std::vector<double> values{160};
    std::vector<double> expected_values{1, 160};
    ov::element::Type value_type = ov::element::dynamic;
    bool implicit_axes = false;
    bool runtime_values = false;
    bool runtime_axes = false;

    explicit Config(std::string test_name) : name(std::move(test_name)) {
        attrs.mode = Interpolate::InterpolateMode::LINEAR_ONNX;
        attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SCALES;
        attrs.pads_begin = attrs.pads_end = {0, 0, 0};
    }
};

std::shared_ptr<ov::Model> getModel(const Config& config) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, config.shape);
    data->set_friendly_name("data");
    data->output(0).set_names({"data"});
    ov::ParameterVector parameters{data};
    const bool scales_mode = config.attrs.shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES;
    const auto value_type = config.value_type.is_static() ? config.value_type : (scales_mode ? ov::element::f32 : ov::element::i64);
    ov::Output<ov::Node> shape_values = Constant::create(value_type, {config.values.size()}, config.values);
    if (config.runtime_values) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(value_type, ov::Shape{config.values.size()});
        parameter->output(0).set_names({"resize_values"});
        parameters.push_back(parameter);
        shape_values = parameter;
    }
    ov::Output<ov::Node> axes_input = Constant::create(ov::element::i64, {config.axes.size()}, config.axes);
    if (config.runtime_axes) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{config.axes.size()});
        parameter->output(0).set_names({"resize_axes"});
        parameters.push_back(parameter);
        axes_input = parameter;
    }
    shape_values.get_tensor().set_names({"resize_values"});
    axes_input.get_tensor().set_names({"resize_axes"});

    std::shared_ptr<ov::Node> output;
    if (config.v11) {
        output = config.implicit_axes ? std::make_shared<ov::op::v11::Interpolate>(data, shape_values, config.attrs)
                                      : std::make_shared<ov::op::v11::Interpolate>(data, shape_values, axes_input, config.attrs);
    } else {
        auto unused = Constant::create(scales_mode ? ov::element::i64 : ov::element::f32, {config.values.size()}, std::vector<int>(config.values.size(), 1));
        unused->output(0).set_names({"unused_resize_values"});
        ov::Output<ov::Node> sizes = scales_mode ? unused : shape_values;
        ov::Output<ov::Node> scales = scales_mode ? shape_values : unused;
        output = config.implicit_axes ? std::make_shared<ov::op::v4::Interpolate>(data, sizes, scales, config.attrs)
                                      : std::make_shared<ov::op::v4::Interpolate>(data, sizes, scales, axes_input, config.attrs);
    }
    output->set_friendly_name("interpolate");
    output->output(0).set_names({"interpolated_data"});
    return std::make_shared<ov::Model>(ov::OutputVector{output}, parameters);
}

std::shared_ptr<ov::Model> getModelRef(const Config& config) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, config.shape);
    data->set_friendly_name("data");
    data->output(0).set_names({"data"});
    const auto singleton_axis = Constant::create(ov::element::i64, {1}, {2});
    const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data, singleton_axis);
    const auto axes = Constant::create(ov::element::i64, {2}, {2, 3});
    const bool scales_mode = config.attrs.shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES;
    const auto value_type = config.value_type.is_static() ? config.value_type : (scales_mode ? ov::element::f32 : ov::element::i64);
    const auto values = Constant::create(value_type, {2}, config.expected_values);
    auto attrs = config.attrs;
    attrs.pads_begin = attrs.pads_end = {0, 0, 0, 0};
    std::shared_ptr<ov::Node> interpolate;
    if (config.v11) {
        interpolate = std::make_shared<ov::op::v11::Interpolate>(unsqueeze, values, axes, attrs);
    } else {
        const auto unused = Constant::create(scales_mode ? ov::element::i64 : ov::element::f32, {2}, {1, 1});
        interpolate = std::make_shared<ov::op::v4::Interpolate>(unsqueeze, scales_mode ? unused : values, scales_mode ? values : unused, axes, attrs);
    }
    const auto squeeze = std::make_shared<ov::op::v0::Squeeze>(interpolate, singleton_axis);
    squeeze->set_friendly_name("interpolate");
    squeeze->output(0).set_names({"interpolated_data"});
    return std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{data});
}

std::vector<Config> positive_cases() {
    std::vector<Config> cases;
    for (bool v11 : {false, true}) {
        const auto prefix = v11 ? "V11_" : "V4_";
        Config last_axis(std::string(prefix) + "LastAxis");
        last_axis.v11 = v11;
        auto dynamic = last_axis;
        dynamic.name = std::string(prefix) + "DynamicDimensions";
        dynamic.shape = {-1, 18, -1};
        dynamic.values = {0.75};
        dynamic.expected_values = {1, 0.75};
        dynamic.attrs.coordinate_transformation_mode = Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
        cases.push_back(dynamic);
        auto all_axes = last_axis;
        all_axes.name = std::string(prefix) + "ImplicitAllAxes";
        all_axes.implicit_axes = true;
        all_axes.values = {1, 1, 160};
        cases.push_back(all_axes);
        auto sizes = dynamic;
        sizes.name = std::string(prefix) + "SizesDynamicDimensions";
        sizes.attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
        sizes.values = {16160};
        sizes.expected_values = {1, 16160};
        cases.push_back(sizes);
        if (v11)
            continue;
        all_axes.name = "V4_PermutedAllAxes";
        all_axes.implicit_axes = false;
        all_axes.axes = {2, 0, 1};
        all_axes.values = {160, 1, 1};
        cases.push_back(all_axes);
        auto typed_scales = last_axis;
        typed_scales.name = "V4_ConstantScales_f16";
        typed_scales.value_type = ov::element::f16;
        typed_scales.values = {0.75};
        typed_scales.expected_values = {1, 0.75};
        cases.push_back(typed_scales);
        auto typed_sizes = last_axis;
        typed_sizes.name = "V4_ConstantSizes_i32";
        typed_sizes.attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
        typed_sizes.value_type = ov::element::i32;
        typed_sizes.values = {16777217};
        typed_sizes.expected_values = {1, 16777217};
        cases.push_back(typed_sizes);
    }
    return cases;
}

std::vector<Config> negative_cases() {
    std::vector<Config> cases;
    Config runtime_scales("V4_RuntimeScales");
    runtime_scales.runtime_values = true;
    cases.push_back(runtime_scales);
    Config runtime_sizes("V11_RuntimeSizes");
    runtime_sizes.v11 = true;
    runtime_sizes.runtime_values = true;
    runtime_sizes.attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
    cases.push_back(runtime_sizes);
    Config runtime_axes("V11_RuntimeAxes");
    runtime_axes.v11 = true;
    runtime_axes.runtime_axes = true;
    cases.push_back(runtime_axes);
    Config empty_axes("V4_EmptyAxesScales");
    empty_axes.axes = {};
    empty_axes.values = {1, 1, 160};
    cases.push_back(empty_axes);
    Config base("SharedGuards");
    auto add = [&](const std::string& name, const Config& value) {
        cases.push_back(value);
        cases.back().name = name;
    };
    auto config = base;
    config.shape = ov::PartialShape::dynamic(4);
    config.axes = {3};
    add("Rank4", config);
    config = base;
    config.attrs.mode = Interpolate::InterpolateMode::NEAREST;
    add("OtherModeNearest", config);
    config = base;
    config.attrs.antialias = true;
    add("Antialias", config);
    config = base;
    config.attrs.pads_begin[2] = 1;
    add("PadBegin", config);
    config = base;
    config.attrs.pads_end[2] = 1;
    add("PadEnd", config);
    config = base;
    config.axes = {0, 1, 2};
    config.values = {2, 1, 160};
    add("ResizedBatch", config);
    config.values = {1, 2, 160};
    add("ResizedChannel", config);
    config = base;
    config.axes = {1};
    add("NotLastAxis", config);
    config = base;
    config.axes = {0, 0, 2};
    config.values = {1, 1, 160};
    add("DuplicateAxis", config);
    config = base;
    config.values = {0};
    add("ZeroScale", config);
    config = base;
    config.attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
    config.axes = {0, 1, 2};
    config.values = {2, 18, 16160};
    config.shape = {-1, 18, 101};
    add("UnprovenBatchSize", config);
    config = base;
    config.attrs.shape_calculation_mode = Interpolate::ShapeCalcMode::SIZES;
    config.values = {0};
    add("ZeroSize", config);
    return cases;
}

std::vector<Config> callback_cases() {
    Config v4("V4");
    Config v11("V11");
    v11.v11 = true;
    return {v4, v11};
}

class NormalizeInterpolateRank3Test : public TransformationTestsF, public testing::WithParamInterface<Config> {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::TENSOR_NAMES);
        manager.register_pass<ov::intel_gpu::NormalizeInterpolateRank3>();
        manager.register_pass<ov::intel_gpu::NormalizeInterpolateRank3>();
    }
};

class NormalizeInterpolateRank3Positive : public NormalizeInterpolateRank3Test {};
class NormalizeInterpolateRank3Negative : public NormalizeInterpolateRank3Test {};
class NormalizeInterpolateRank3Callback : public NormalizeInterpolateRank3Test {};

TEST_P(NormalizeInterpolateRank3Positive, NormalizesInterpolate) {
    model = getModel(GetParam());
    model_ref = getModelRef(GetParam());
}

TEST_P(NormalizeInterpolateRank3Callback, HonorsTransformationCallback) {
    model = getModel(GetParam());
    manager.get_pass_config()->set_callback<ov::intel_gpu::NormalizeInterpolateRank3>([](const std::shared_ptr<const ov::Node>&) {
        return true;
    });
}

TEST_P(NormalizeInterpolateRank3Negative, LeavesUnsupportedCaseUnchanged) {
    model = getModel(GetParam());
}

const auto case_name = [](const testing::TestParamInfo<Config>& info) {
    return info.param.name;
};

INSTANTIATE_TEST_SUITE_P(smoke, NormalizeInterpolateRank3Positive, testing::ValuesIn(positive_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(smoke, NormalizeInterpolateRank3Negative, testing::ValuesIn(negative_cases()), case_name);
INSTANTIATE_TEST_SUITE_P(smoke, NormalizeInterpolateRank3Callback, testing::ValuesIn(callback_cases()), case_name);
}  // namespace
