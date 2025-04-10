// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/shape_of.hpp"

namespace ov {
namespace test {
std::string ShapeOfLayerTest::getTestCaseName(testing::TestParamInfo<shapeOfParams> obj) {
    std::vector<InputShape> input_shapes;
    ov::element::Type model_type, out_type;
    std::string target_device;
    std::tie(model_type, out_type, input_shapes, target_device) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "modelType=" << model_type.to_string() << "_";
    result << "outType=" << out_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ShapeOfLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type model_type;
    std::tie(model_type, outType, input_shapes, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(param, outType);
    function = std::make_shared<ov::Model>(shape_of->outputs(), ov::ParameterVector{param}, "ShapeOf");
}
}  // namespace test
}  // namespace ov
