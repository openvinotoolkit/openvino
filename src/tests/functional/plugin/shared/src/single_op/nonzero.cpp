// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/nonzero.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/non_zero.hpp"

namespace ov {
namespace test {

std::string NonZeroLayerTest::getTestCaseName(const testing::TestParamInfo<NonZeroLayerTestParamsSet>& obj) {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::string target_device;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes, model_type, target_device, additional_config) = obj.param;

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
    result << "inPRC=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void NonZeroLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes, model_type, targetDevice, additional_config) = GetParam();
    configuration.insert(additional_config.cbegin(), additional_config.cend());
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto non_zero = std::make_shared<ov::op::v3::NonZero>(param);

    auto result = std::make_shared<ov::op::v0::Result>(non_zero);

    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "non_zero");
}
}  // namespace test
}  // namespace ov
