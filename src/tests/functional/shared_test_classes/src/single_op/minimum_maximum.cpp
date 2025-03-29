// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/minimum_maximum.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/maximum.hpp"


namespace ov {
namespace test {
using ov::test::utils::InputLayerType;
using ov::test::utils::MinMaxOpType;

std::string MaxMinLayerTest::getTestCaseName(const testing::TestParamInfo<MaxMinParamsTuple> &obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string target_name;
    InputLayerType second_input_type;
    MinMaxOpType op_type;
    std::tie(shapes, op_type, model_type, second_input_type, target_name) = obj.param;
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
    result << "OpType=" << op_type << "_";
    result << "SecondaryInputType=" << second_input_type << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_name << "_";
    return result.str();
}

void MaxMinLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    InputLayerType second_input_type;
    MinMaxOpType op_type;
    std::tie(shapes, op_type, model_type, second_input_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
    ov::NodeVector inputs {params[0]};

    if (InputLayerType::PARAMETER == second_input_type) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
        params.push_back(param);
        inputs.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, targetStaticShapes[0][1]);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        inputs.push_back(constant);
    }

    std::shared_ptr<ov::Node> min_max_op;
    switch (op_type) {
    case MinMaxOpType::MINIMUM:
        min_max_op = std::make_shared<ov::op::v1::Minimum>(inputs[0], inputs[1]);
        break;
    case MinMaxOpType::MAXIMUM:
        min_max_op = std::make_shared<ov::op::v1::Maximum>(inputs[0], inputs[1]);
        break;
    default:
        throw std::logic_error("Unsupported operation type");
    }

    auto result = std::make_shared<ov::op::v0::Result>(min_max_op);
    function = std::make_shared<ov::Model>(result, params, "MinMax");
}
} // namespace test
} // namespace ov
