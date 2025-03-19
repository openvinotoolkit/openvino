// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/range.hpp"


namespace ov {
namespace test {
std::string RangeLayerTest::getTestCaseName(const testing::TestParamInfo<RangeParams>& obj) {
    ov::element::Type model_type;
    float start, stop, step;
    std::string target_device;
    std::tie(start, stop, step, model_type, target_device) = obj.param;

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
    ov::element::Type model_type;
    float start, stop, step;
    tie(start, stop, step, model_type, targetDevice) = GetParam();

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape())};
    params[0]->set_friendly_name("start");
    params[1]->set_friendly_name("stop");
    params[2]->set_friendly_name("step");
    auto range = std::make_shared<ov::op::v4::Range>(params[0], params[1], params[2], model_type);
    function = std::make_shared<ov::Model>(range->outputs(), params, "Range");
}
}  // namespace test
}  // namespace ov
