// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/reorg_yolo.hpp"

namespace ov {
namespace test {
std::string ReorgYoloLayerTest::getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple> &obj) {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    size_t stride;
    std::string target_device;
    std::tie(input_shape, stride, model_type, target_device) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "stride=" << stride << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ReorgYoloLayerTest::SetUp() {
    std::vector<size_t> input_shape;
    ov::element::Type model_type;
    size_t stride;
    std::tie(input_shape, stride, model_type, targetDevice) = this->GetParam();
    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto reorg_yolo = std::make_shared<ov::op::v0::ReorgYolo>(param, stride);
    function = std::make_shared<ov::Model>(reorg_yolo->outputs(), ov::ParameterVector{param}, "ReorgYolo");
}
}  // namespace test
}  // namespace ov
