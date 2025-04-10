// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset12;

using NodePtr = std::shared_ptr<ov::Node>;

class IPadFactory {
public:
    explicit IPadFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IPadFactory() = default;
    virtual std::shared_ptr<ov::Node> create(const Output<Node>& arg,
                                             const Output<Node>& pads_begin,
                                             const Output<Node>& pads_end,
                                             ov::op::PadMode pad_mode) const = 0;
    virtual std::shared_ptr<ov::Node> create(const Output<Node>& arg,
                                             const Output<Node>& pads_begin,
                                             const Output<Node>& pads_end,
                                             const Output<Node>& arg_pad_value,
                                             ov::op::PadMode pad_mode) const = 0;

    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};
using PadFactoryPtr = std::shared_ptr<IPadFactory>;

template <typename PadT>
class PadFactory : public IPadFactory {
public:
    explicit PadFactory(const std::string& type_name) : IPadFactory(type_name) {}
    NodePtr create(const Output<Node>& arg,
                   const Output<Node>& pads_begin,
                   const Output<Node>& pads_end,
                   ov::op::PadMode pad_mode) const override {
        return std::make_shared<PadT>(arg, pads_begin, pads_end, pad_mode);
    }
    NodePtr create(const Output<Node>& arg,
                   const Output<Node>& pads_begin,
                   const Output<Node>& pads_end,
                   const Output<Node>& arg_pad_value,
                   ov::op::PadMode pad_mode) const override {
        return std::make_shared<PadT>(arg, pads_begin, pads_end, arg_pad_value, pad_mode);
    }
};

template <typename PadT>
PadFactoryPtr CreatePadFactory(const std::string& type_name) {
    return std::make_shared<PadFactory<PadT>>(type_name);
}

struct ITestModelFactory {
    explicit ITestModelFactory(const std::string& a_test_name) : test_name(a_test_name) {}
    virtual ~ITestModelFactory() = default;
    virtual void setup(PadFactoryPtr pad_factory, ov::pass::Manager& manager) = 0;
    std::string test_name;
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Model> model_ref;
};
using TestModelFactoryPtr = std::shared_ptr<ITestModelFactory>;

using TestParams = std::tuple<PadFactoryPtr, TestModelFactoryPtr>;

class ConvertPadGroupConvTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestParams>& obj) {
        PadFactoryPtr pad_factory;
        TestModelFactoryPtr model_factory;
        std::tie(pad_factory, model_factory) = obj.param;

        std::ostringstream test_name;
        test_name << "pad_factory=" << pad_factory->getTypeName() << "/";
        test_name << "model_factory=" << model_factory->test_name;

        return test_name.str();
    }
};

TEST_P(ConvertPadGroupConvTestFixture, CompareFunctions) {
    PadFactoryPtr pad_factory;
    TestModelFactoryPtr model_factory;
    std::tie(pad_factory, model_factory) = this->GetParam();

    model_factory->setup(pad_factory, manager);
    model = model_factory->model;
    model_ref = model_factory->model_ref;
    if (!model_ref)
        model_ref = model->clone();
}

#define TEST_BODY(TestName)                                                         \
    struct TestName : public ITestModelFactory {                                    \
        TestName() : ITestModelFactory(#TestName) {}                                \
        void setup(PadFactoryPtr pad_factory, ov::pass::Manager& manager) override; \
    };                                                                              \
    void TestName::setup(PadFactoryPtr pad_factory, ov::pass::Manager& manager)

TEST_BODY(ConvertPadToConv) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        model = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});

        manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto weights = Constant::create(element::f32, Shape{3, 1, 1, 1, 1}, {1});
        Strides stride{1, 1};
        CoordinateDiff pad_begin{1, 0}, pad_end{0, 1};
        auto conv = std::make_shared<GroupConvolution>(input, weights, stride, pad_begin, pad_end, stride);

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
}

TEST_BODY(NegativeConvertPadToConv) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, -1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        model = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }
    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

TEST_BODY(ConvertPadToConvNeg1) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {1, 0, 1, 0});  // Batch dim padding
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        model = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

TEST_BODY(ConvertPadToConvNeg2) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 1, 0, 1});  // Channel dim padding
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        model = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

TEST_BODY(ConvertPadToConvNeg3) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::SYMMETRIC;  // Unsupported mode
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        model = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ConvertPadToGroupConvolution>();
}

TEST_BODY(ConvertPadToConvNeg4) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = Constant::create(element::f32, Shape{}, {1.});  // Unsupported value
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = pad_factory->create(input, pad_begin, pad_end, pad_value, pad_mode);
        model = std::make_shared<Model>(NodeVector{pad}, ParameterVector{input});
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
                         ConvertPadGroupConvTestFixture,
                         ::testing::Combine(::testing::ValuesIn(pad_factories), ::testing::ValuesIn(model_factories)),
                         ConvertPadGroupConvTestFixture::get_test_name);
