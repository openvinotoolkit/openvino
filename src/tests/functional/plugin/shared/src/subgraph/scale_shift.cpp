// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/scaleshift.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace test {
std::string ScaleShiftLayerTest::getTestCaseName(const testing::TestParamInfo<ScaleShiftParamsTuple> &obj) {
    std::vector<ov::Shape> inputShapes;
    ov::element::Type type;
    std::string targetName;
    std::vector<float> scale, shift;
    std::tie(inputShapes, type, targetName, scale, shift) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    results << "Scale=" << ov::test::utils::vec2str(scale) << "_";
    results << "Shift=" << ov::test::utils::vec2str(shift) << "_";
    results << "netPRC=" << type.get_type_name() << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void ScaleShiftLayerTest::SetUp() {
    std::vector<ov::Shape> inputShapes;
    ov::element::Type type;
    std::vector<float> scale, shift;
    std::tie(inputShapes, type, targetDevice, scale, shift) = this->GetParam();
    auto paramsShape = ov::Shape{1};
    if (inputShapes.size() > 1)
        paramsShape = inputShapes[1];

    ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(type, inputShapes[0])};
    auto mul_const = std::make_shared<ov::op::v0::Constant>(type, paramsShape, scale);
    auto mul = std::make_shared<ov::op::v1::Multiply>(paramsIn[0], mul_const);
    auto add_const = std::make_shared<ov::op::v0::Constant>(type, paramsShape, shift);
    auto add = std::make_shared<ov::op::v1::Add>(mul, add_const);
    function = std::make_shared<ov::Model>(add, paramsIn, "scale_shift");
}
} // namespace test
} // namespace ov
