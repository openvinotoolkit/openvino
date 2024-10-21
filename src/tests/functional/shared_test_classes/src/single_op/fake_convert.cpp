// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/fake_convert.hpp"

namespace ov {
namespace test {
std::string FakeConvertLayerTest::getTestCaseName(const testing::TestParamInfo<fakeConvertParamsTuple>& obj) {
    ov::element::Type destination_type, model_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    std::tie(destination_type, shapes, model_type, target_device) = obj.param;

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
    result << "trgPRC=" << destination_type.get_type_name() << "_";
    result << "netPRC=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void FakeConvertLayerTest::SetUp() {
    ov::element::Type destination_type, model_type;
    std::vector<InputShape> shapes;
    std::tie(destination_type, shapes, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (const auto& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    std::shared_ptr<ov::op::v13::FakeConvert> fc;
    if (inputDynamicShapes.size() == 2) {
        fc = std::make_shared<ov::op::v13::FakeConvert>(params[0], params[1], destination_type);
    } else {
        fc = std::make_shared<ov::op::v13::FakeConvert>(params[0], params[1], params[2], destination_type);
    }
    auto result = std::make_shared<ov::op::v0::Result>(fc);
    function = std::make_shared<ov::Model>(result, params, "fakeConvert");
}
}  // namespace test
}  // namespace ov
