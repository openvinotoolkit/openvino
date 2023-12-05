// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertMaxPool8ToMaxPool1) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 3});
        Strides strides{1}, dilations{1};
        Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_8 = std::make_shared<opset8::MaxPool>(data, strides, dilations, pads_begin, pads_end, kernel);
        auto result = std::make_shared<opset1::Result>(maxpool_8->output(0));

        model = std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{data});
        manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 3});
        Strides strides{1};
        Shape pads_begin{0}, pads_end{0}, kernel{1};
        auto maxpool_1 = std::make_shared<opset1::MaxPool>(data, strides, pads_begin, pads_end, kernel);
        auto result = std::make_shared<opset1::Result>(maxpool_1->output(0));

        model_ref = std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{data});
    }
}
