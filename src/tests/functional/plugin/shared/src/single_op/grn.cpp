// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grn.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/grn.hpp"

namespace ov {
namespace test {
std::string GrnLayerTest::getTestCaseName(const testing::TestParamInfo<grnParams>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    float bias;
    std::tie(model_type, shapes, bias, target_device) = obj.param;

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
    result << "modelType=" << model_type.get_type_name() << '_';
    result << "bias="   << bias << '_';
    result << "trgDev=" << '_';
    return result.str();
}

void GrnLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    float bias;
    std::tie(model_type, shapes, bias, targetDevice) = GetParam();
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto grn = std::make_shared<ov::op::v0::GRN>(param, bias);
    auto result = std::make_shared<ov::op::v0::Result>(grn);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "Grn");
}
}  // namespace test
}  // namespace ov
