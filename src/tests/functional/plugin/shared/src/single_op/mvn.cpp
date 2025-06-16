// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/mvn.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"

namespace ov {
namespace test {
std::string Mvn1LayerTest::getTestCaseName(const testing::TestParamInfo<mvn1Params>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    ov::AxisSet axes;
    bool across_channels, normalize_variance;
    double eps;
    std::string target_device;
    std::tie(shapes, model_type, axes, across_channels, normalize_variance, eps, target_device) = obj.param;
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
    result << "ModelType=" << model_type.get_type_name() << "_";
    if (!axes.empty()) {
        result << "ReductionAxes=" << ov::test::utils::vec2str(axes.to_vector()) << "_";
    } else {
        result << "across_channels=" << (across_channels ? "TRUE" : "FALSE") << "_";
    }
    result << "normalize_variance=" << (normalize_variance ? "TRUE" : "FALSE") << "_";
    result << "Epsilon=" << eps << "_";
    result << "TargetDevice=" << target_device;
    return result.str();
}

void Mvn1LayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    ov::AxisSet axes;
    bool across_channels, normalize_variance;
    double eps;
    std::tie(shapes, model_type, axes, across_channels, normalize_variance, eps, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    std::shared_ptr<ov::op::v0::MVN> mvn;

    if (axes.empty()) {
        mvn = std::make_shared<ov::op::v0::MVN>(param, across_channels, normalize_variance, eps);

        // OpenVINO MVN implementation implicitly adds 0th dimension to reduction axes set which is not valid behavior
        ov::AxisSet axes;
        const size_t startAxis = across_channels ? 1 : 2;
        const size_t numOfDims = param->output(0).get_partial_shape().size();
        for (size_t i = startAxis; i < numOfDims; i++)
            axes.insert(i);
        mvn->set_reduction_axes(axes);
    } else {
        mvn = std::make_shared<ov::op::v0::MVN>(param, axes, normalize_variance, eps);
    }

    if (model_type == ov::element::f32) {
        abs_threshold = 5e-7;
    }

    auto result = std::make_shared<ov::op::v0::Result>(mvn);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "MVN1");
}

std::string Mvn6LayerTest::getTestCaseName(const testing::TestParamInfo<mvn6Params>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    ov::element::Type axis_type;
    std::vector<int> axes;
    bool normalize_variance;
    float eps;
    std::string eps_mode;
    std::string target_device;
    std::tie(shapes, model_type, axis_type, axes, normalize_variance, eps, eps_mode, target_device) = obj.param;
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
    result << "ModelType=" << model_type.get_type_name() << "_";
    result << "AxType=" << axis_type.get_type_name() << "_";
    result << "Ax=" << ov::test::utils::vec2str(axes) << "_";
    result << "NormVariance=" << (normalize_variance ? "TRUE" : "FALSE") << "_";
    result << "Eps=" << eps << "_";
    result << "EM=" << eps_mode << "_";
    result << "TargetDevice=" << target_device;
    return result.str();
}

void Mvn6LayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    ov::element::Type axis_type;
    std::vector<int> axes;
    bool normalize_variance;
    float eps;
    std::string eps_mode;
    std::tie(shapes, model_type, axis_type, axes, normalize_variance, eps, eps_mode, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto axes_node = ov::op::v0::Constant::create(axis_type, ov::Shape{axes.size()}, axes);

    ov::op::MVNEpsMode nEpsMode = ov::op::MVNEpsMode::INSIDE_SQRT;
    if (eps_mode == "outside_sqrt")
        nEpsMode = ov::op::MVNEpsMode::OUTSIDE_SQRT;
    auto mvn = std::make_shared<ov::op::v6::MVN>(param, axes_node, normalize_variance, eps, nEpsMode);

    auto result = std::make_shared<ov::op::v0::Result>(mvn);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "MVN6");
}
}  // namespace test
}  // namespace ov
