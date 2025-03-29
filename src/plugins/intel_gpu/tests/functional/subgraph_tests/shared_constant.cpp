// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"

namespace {
// Validate scenario where a single Constant has multiple users (like one constant is used for Convolution, ConvolutionBackpropData, Multiply, etc.)
class SharedConstant : virtual public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto type = ov::element::f32;
        ov::Shape constShape{4, 1, 3, 3};
        ov::Shape convInputShape{1, 1, 5, 5};
        ov::Shape convBackpropInputShape{1, 4, 5, 5};
        ov::Shape constGroupConvBackpropShape{2, 2, 3, 3, 3};
        auto constant = ov::op::v0::Constant::create(type, constShape, {1});
        auto input1 = std::make_shared<ov::op::v0::Parameter>(type, convInputShape);
        auto conv = std::make_shared<ov::op::v1::Convolution>(
            input1, constant, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
        auto input2 = std::make_shared<ov::op::v0::Parameter>(type, convBackpropInputShape);
        auto convBprop = std::make_shared<ov::op::v1::ConvolutionBackpropData>(input2, constant, ov::Strides{1, 1},
                ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
        auto input3 = std::make_shared<ov::op::v0::Parameter>(type, convBackpropInputShape);
        auto constantGroupConv = ov::op::v0::Constant::create(type, constGroupConvBackpropShape, {1});
        auto groupConvBprop = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input3, constantGroupConv, ov::Strides{1, 1},
                ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
        auto input4 = std::make_shared<ov::op::v0::Parameter>(type, constShape);
        auto mul = std::make_shared<ov::op::v1::Multiply>(input4, constant);
        auto input5 = std::make_shared<ov::op::v0::Parameter>(type, constGroupConvBackpropShape);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(input5, constantGroupConv);
        // explicitly set the output name, to avoid global conflict
        mul2->set_friendly_name("Multiply_0");
        mul->set_friendly_name("Multiply_1");
        function = std::make_shared<ov::Model>(ov::NodeVector{convBprop, conv, groupConvBprop, mul2, mul},
                ov::ParameterVector{input1, input2, input3, input4, input5});
    }
};

TEST_F(SharedConstant, Inference) {
    run();
}
}  // namespace
