// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/einsum.hpp"
#include "openvino/op/einsum.hpp"

namespace ov {
namespace test {

std::string EinsumLayerTest::getTestCaseName(const testing::TestParamInfo<EinsumLayerTestParamsSet>& obj) {
    EinsumEquationWithInput equation_with_input;
    ov::element::Type model_type;
    std::string targetDevice;
    std::tie(model_type, equation_with_input, targetDevice) = obj.param;
    std::string equation;
    std::vector<InputShape> shapes;
    std::tie(equation, shapes) = equation_with_input;

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
    result << "PRC=" << model_type.get_type_name() << "_";
    result << "Eq=" << equation << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void EinsumLayerTest::SetUp() {
    EinsumEquationWithInput equation_with_input;
    ov::element::Type model_type;
    std::tie(model_type, equation_with_input, targetDevice) = this->GetParam();
    std::string equation;
    std::vector<InputShape> shapes;
    std::tie(equation, shapes) = equation_with_input;
    init_input_shapes(shapes);

    ov::ParameterVector params;
    ov::OutputVector param_outs;
    for (const auto& shape : inputDynamicShapes) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(param);
        param_outs.push_back(param);
    }

    auto einsum = std::make_shared<ov::op::v7::Einsum>(param_outs, equation);

    auto result = std::make_shared<ov::op::v0::Result>(einsum);
    function = std::make_shared<ov::Model>(result, params, "einsum");
}
}  // namespace test
}  // namespace ov
