// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/concat.hpp"

namespace ov {
namespace test {

std::string ConcatLayerTest::getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj) {
    int axis;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string targetName;
    std::tie(axis, shapes, model_type, targetName) = obj.param;
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
    result << "axis=" << axis << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConcatLayerTest::SetUp() {
    int axis;
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::tie(axis, shapes, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector params;
    ov::NodeVector params_nodes;
    for (const auto& shape : inputDynamicShapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(param);
        params_nodes.push_back(param);
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(params_nodes, axis);
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    function = std::make_shared<ov::Model>(result, params, "concat");
}
}  // namespace test
}  // namespace ov
