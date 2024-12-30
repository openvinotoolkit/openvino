// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/fake_convert.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ov {
namespace test {
std::string FakeConvertLayerTest::getTestCaseName(const testing::TestParamInfo<FakeConvertParams>& obj) {
    FakeConvertParams params = obj.param;

    std::vector<InputShape> data_shapes;
    Shape scale_shape, shift_shape;
    element::Type_t data_prec, dst_prec;
    bool default_shift;
    std::string target_device;
    std::tie(data_shapes, scale_shape, shift_shape, data_prec, dst_prec, default_shift, target_device) = params;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : data_shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : data_shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << ")_scaleShape=" << ov::test::utils::vec2str(scale_shape) << "_";
    result << "shiftShape=" << ov::test::utils::vec2str(shift_shape) << "_";
    result << "dataPrecision=" << element::Type(data_prec) << "_";
    result << "destinationPrecision=" << element::Type(dst_prec) << "_";
    if (default_shift)
        result << "defaultShift=true";
    else
        result << "defaultShift=false";
    return result.str();
}

void FakeConvertLayerTest::SetUp() {
    FakeConvertParams params = this->GetParam();

    std::vector<InputShape> data_shapes;
    Shape scale_shape, shift_shape;
    element::Type_t data_prec, dst_prec;
    bool default_shift;
    std::tie(data_shapes, scale_shape, shift_shape, data_prec, dst_prec, default_shift, targetDevice) = params;

    init_input_shapes(data_shapes);

    const auto data = std::make_shared<opset1::Parameter>(data_prec, inputDynamicShapes.front());
    const auto scale = std::make_shared<opset1::Constant>(data_prec, scale_shape);
    const auto shift = std::make_shared<opset1::Constant>(data_prec, shift_shape);

    const auto fake_convert = default_shift ? std::make_shared<opset13::FakeConvert>(data, scale, dst_prec)
                                            : std::make_shared<opset13::FakeConvert>(data, scale, shift, dst_prec);
    function = std::make_shared<ov::Model>(NodeVector{fake_convert}, ParameterVector{data});
}
}  // namespace test
}  // namespace ov
