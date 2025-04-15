// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/select.hpp"

namespace ov {
namespace test {
std::string SelectLayerTest::getTestCaseName(const testing::TestParamInfo<selectTestParams> &obj) {
    auto shapes_ss = [](const InputShape& shape) {
        std::stringstream ss;
        ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
        for (size_t j = 0lu; j < shape.second.size(); j++)
            ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
        return ss;
    };

    std::vector<InputShape> input_shapes;
    ov::element::Type model_type;
    ov::op::AutoBroadcastSpec broadcast;
    std::string target_device;
    std::tie(input_shapes, model_type, broadcast, target_device) = obj.param;
    std::ostringstream result;
    result << "COND=BOOL" << shapes_ss(input_shapes[0]).str() <<
        "_THEN=" << model_type.to_string() << shapes_ss(input_shapes[1]).str() <<
        "_ELSE=" << model_type.to_string() << shapes_ss(input_shapes[2]).str();
    result << "_broadcastSpec=" << broadcast.m_type;
    result << "_trgDev=" << target_device;
    return result.str();
}

void SelectLayerTest::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type model_type;
    ov::op::AutoBroadcastSpec broadcast;
    std::tie(input_shapes, model_type, broadcast, targetDevice) = this->GetParam();

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[0]);
    ov::ParameterVector params{param};
    for (size_t i = 1; i < inputDynamicShapes.size(); i++) {
        param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[i]);
        params.push_back(param);
    }
    auto select = std::make_shared<ov::op::v1::Select>(params[0], params[1], params[2], broadcast);
    function = std::make_shared<ov::Model>(select->outputs(), params, "Select");
}
}  // namespace test
}  // namespace ov
