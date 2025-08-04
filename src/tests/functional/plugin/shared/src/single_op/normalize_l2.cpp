// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/normalize_l2.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/normalize_l2.hpp"

namespace ov {
namespace test {
std::string NormalizeL2LayerTest::getTestCaseName(const testing::TestParamInfo<NormalizeL2LayerTestParams>& obj) {
    std::vector<int64_t> axes;
    float eps;
    ov::op::EpsMode eps_mode;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string targetDevice;
    std::tie(axes, eps, eps_mode, shapes, model_type, targetDevice) = obj.param;

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
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "eps=" << eps << "_";
    result << "eps_mode=" << eps_mode << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NormalizeL2LayerTest::SetUp() {
    std::vector<InputShape> shapes;
    std::vector<int64_t> axes;
    float eps;
    ov::op::EpsMode eps_mode;
    ov::element::Type model_type;
    std::tie(axes, eps, eps_mode, shapes, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto norm_axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
    auto norm = std::make_shared<ov::op::v0::NormalizeL2>(param, norm_axes, eps, eps_mode);

    auto result = std::make_shared<ov::op::v0::Result>(norm);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "NormalizeL2");
}
}  // namespace test
}  // namespace ov
