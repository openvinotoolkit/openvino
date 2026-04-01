// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/identity.hpp"
#include "openvino/op/identity.hpp"

namespace ov::test {
std::string IdentityLayerTest::getTestCaseName(const testing::TestParamInfo<identityParams>& obj) {
    const auto& [modelType, input_shapes, targetDevice] = obj.param;
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
    result << "modelType=" << modelType.to_string() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void IdentityLayerTest::SetUp() {
    const auto& [model_type, input_shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    init_input_shapes({input_shapes});

    const auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    param->set_friendly_name("param");

    const auto op = std::make_shared<ov::op::v16::Identity>(param);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(op)};
    function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "Identity");
}
}  // namespace ov::test
