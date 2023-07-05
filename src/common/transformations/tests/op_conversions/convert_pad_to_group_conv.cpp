// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/op/pad.hpp>
#include <queue>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "pad_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset12;

using NodePtr = std::shared_ptr<ov::Node>;
using PadFactoryPtr = std::shared_ptr<IPadFactory>;
using TestModelFactoryPtr = std::shared_ptr<ITestModelFactory>;
using TestParams = std::tuple<PadFactoryPtr, TestModelFactoryPtr>;

PAD_TEST_BODY(ConvertPadToConv) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});

        manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto weights = Constant::create(element::f32, Shape{3, 1, 1, 1, 1}, {1});
        Strides stride{1, 1};
        CoordinateDiff pad_begin{1, 0}, pad_end{0, 1};
        auto conv = std::make_shared<GroupConvolution>(input, weights, stride, pad_begin, pad_end, stride);

        function_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
}

PAD_TEST_BODY(NegativeConvertPadToConv) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, -1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }
    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

PAD_TEST_BODY(ConvertPadToConvNeg1) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {1, 0, 1, 0});  // Batch dim padding
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

PAD_TEST_BODY(ConvertPadToConvNeg2) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 1, 0, 1});  // Channel dim padding
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

PAD_TEST_BODY(ConvertPadToConvNeg3) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::SYMMETRIC;  // Unsupported mode
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

PAD_TEST_BODY(ConvertPadToConvNeg4) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {1.});  // Unsupported value
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

namespace {

#undef CREATE_MODEL_FACTORY
#define CREATE_MODEL_FACTORY(type_name) std::make_shared<type_name>()

std::vector<TestModelFactoryPtr> model_factories = {CREATE_MODEL_FACTORY(ConvertPadToConv),
                                                    CREATE_MODEL_FACTORY(ConvertPadToConvNeg1),
                                                    CREATE_MODEL_FACTORY(ConvertPadToConvNeg2),
                                                    CREATE_MODEL_FACTORY(ConvertPadToConvNeg3),
                                                    CREATE_MODEL_FACTORY(ConvertPadToConvNeg4),
                                                    CREATE_MODEL_FACTORY(NegativeConvertPadToConv)};

#undef CREATE_PAD_FACTORY
#define CREATE_PAD_FACTORY(type_name, type_str) CreatePadFactory<type_name>(type_str)

std::vector<PadFactoryPtr> pad_factories = {CREATE_PAD_FACTORY(ov::op::v1::Pad, "op_v1_Pad"),
                                            CREATE_PAD_FACTORY(ov::op::v12::Pad, "op_v12_Pad")};

}  // namespace

INSTANTIATE_TEST_SUITE_P(ConvertPadToGroupConvolutionTestSuite,
                         PadTestFixture,
                         ::testing::Combine(::testing::ValuesIn(pad_factories), ::testing::ValuesIn(model_factories)),
                         PadTestFixture::get_test_name);
