// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/region_yolo.hpp"
#include "openvino/op/region_yolo.hpp"

namespace ov {
namespace test {
std::string RegionYoloLayerTest::getTestCaseName(const testing::TestParamInfo<regionYoloParamsTuple> &obj) {
    const auto& [input_shape,
                 classes,
                 coords,
                 num_regions,
                 do_softmax,
                 mask,
                 start_axis,
                 end_axis,
                 model_type,
                 target_device] = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "classes=" << classes << "_";
    result << "coords=" << coords << "_";
    result << "num=" << num_regions << "_";
    result << "doSoftmax=" << do_softmax << "_";
    result << "axis=" << start_axis << "_";
    result << "endAxis=" << end_axis << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void RegionYoloLayerTest::SetUp() {
    const auto& [input_shape,
                 classes,
                 coords,
                 num_regions,
                 do_softmax,
                 mask,
                 start_axis,
                 end_axis,
                 model_type,
                 _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));
    auto region_yolo = std::make_shared<ov::op::v0::RegionYolo>(param, coords, classes, num_regions, do_softmax, mask, start_axis, end_axis);
    function = std::make_shared<ov::Model>(region_yolo->outputs(), ov::ParameterVector{param}, "RegionYolo");
}
}  // namespace test
}  // namespace ov
