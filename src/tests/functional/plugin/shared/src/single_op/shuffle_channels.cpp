// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/shuffle_channels.hpp"
#include "openvino/op/shuffle_channels.hpp"

namespace ov {
namespace test {
std::string ShuffleChannelsLayerTest::getTestCaseName(const testing::TestParamInfo<shuffleChannelsLayerTestParamsSet>& obj) {
    const auto& [test_params, model_type, input_shapes, target_device] = obj.param;

    const auto& [axis, group] = test_params;

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
    result << "Axis=" << std::to_string(axis) << "_";
    result << "Group=" << std::to_string(group) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev="  << target_device;
    return result.str();
}

void ShuffleChannelsLayerTest::SetUp() {
    std::string target_device;
    const auto& [test_params, model_type, input_shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    const auto& [axis, group] = test_params;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto shuffle_channels = std::make_shared<ov::op::v0::ShuffleChannels>(param, axis, group);
    function = std::make_shared<ov::Model>(shuffle_channels->outputs(), ov::ParameterVector{param}, "ShuffleChannels");
}
}  // namespace test
}  // namespace ov
