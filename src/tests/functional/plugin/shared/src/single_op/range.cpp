// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/range.hpp"

#include <cstring>

#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"


namespace ov {
namespace test {
std::string RangeLayerTest::getTestCaseName(const testing::TestParamInfo<RangeParams>& obj) {
    const auto& [start, stop, step, model_type, target_device] = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "modelType=" << model_type.to_string() << separator;
    result << "trgDev=" << target_device;
    return result.str();
}

void RangeLayerTest::SetUp() {
    const auto& [start, stop, step, model_type, _targetDevice] = GetParam();
    targetDevice = _targetDevice;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape())};
    params[0]->set_friendly_name("start");
    params[1]->set_friendly_name("stop");
    params[2]->set_friendly_name("step");
    auto range = std::make_shared<ov::op::v4::Range>(params[0], params[1], params[2], model_type);
    function = std::make_shared<ov::Model>(range->outputs(), params, "Range");
}

// Feed start, stop and step as the input values, so each instance runs the case in its name.
void RangeLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& [start, stop, step, model_type, _targetDevice] = GetParam();
    const float values[] = {start, stop, step};
    const auto& params = function->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        // The Constant converts the value to the model type.
        const ov::op::v0::Constant value(model_type, ov::Shape{}, values[i]);
        ov::Tensor tensor(model_type, targetInputStaticShapes[i]);
        std::memcpy(tensor.data(), value.get_data_ptr(), tensor.get_byte_size());
        inputs.insert({params[i], tensor});
    }
}
}  // namespace test
}  // namespace ov
