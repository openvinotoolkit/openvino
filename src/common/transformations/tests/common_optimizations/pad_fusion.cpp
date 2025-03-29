// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pad_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
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

class PadFusionTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
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

TEST_P(PadFusionTestFixture, CompareFunctions) {
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

TEST_BODY(PadElimination) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 0, 0});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 0, 0});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::EliminatePad>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1},
                                                  op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadElimination) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::EliminatePad>();
    }
    // Reference function is equal to function
}

TEST_BODY(PadFusionAvgPoolExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{0, 0},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        model = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<AvgPool>(data,
                                                  Strides{1, 1},
                                                  Shape{1, 1},
                                                  Shape{2, 2},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR,
                                                  op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_BODY(NegativePadFusionAvgPoolExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{0, 0},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        model = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

TEST_BODY(PadFusionAvgPoolDontExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR);
        model = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto avg_pool = std::make_shared<AvgPool>(data,
                                                  Strides{1, 1},
                                                  Shape{1, 1},
                                                  Shape{3, 3},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR,
                                                  op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_BODY(NegativePadFusionAvgPoolDontExcludePad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR);
        model = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

TEST_BODY(PadFusionConvolution) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{3, 3},
                                                  Shape{1, 1},
                                                  op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadFusionConvolution) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

TEST_BODY(PadFusionConvolutionBackpropData) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{4, 4},
                                                              CoordinateDiff{3, 3},
                                                              Shape{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(data,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{3, 3},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(PadFusionGroupConvolution) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(pad,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{1, 1},
                                                       Shape{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(data,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{1, 1},
                                                       CoordinateDiff{3, 3},
                                                       Shape{1, 1},
                                                       op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadFusionGroupConvolution) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -2, -2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(pad,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{1, 1},
                                                       Shape{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

TEST_BODY(PadFusionGroupConvolutionBackpropData) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(pad,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{3, 2},
                                                                   CoordinateDiff{4, 3},
                                                                   Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(data,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{2, 1},
                                                                   CoordinateDiff{1, 2},
                                                                   Shape{1, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(PadFusionAvgPoolNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{0, 0},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        model = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto avg_pool = std::make_shared<AvgPool>(data,
                                                  Strides{1, 1},
                                                  Shape{1, 1},
                                                  Shape{2, 2},
                                                  Shape{4, 4},
                                                  false,
                                                  op::RoundingType::FLOOR,
                                                  op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_BODY(PadFusionConvolutionNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{3, 3},
                                                  Shape{1, 1},
                                                  op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(PadFusionConvolutionBackpropDataNonConstPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{4, 4},
                                                              CoordinateDiff{3, 3},
                                                              Shape{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(data,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{3, 3},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(PadFusionGroupConvolutionNonConstPadValue) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(pad,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{0, 0},
                                                       CoordinateDiff{1, 1},
                                                       Shape{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{1, 1, 4, 4, 4});
        auto conv = std::make_shared<GroupConvolution>(data,
                                                       filters,
                                                       Strides{1, 1},
                                                       CoordinateDiff{1, 1},
                                                       CoordinateDiff{3, 3},
                                                       Shape{1, 1},
                                                       op::PadType::EXPLICIT);
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(PadFusionGroupConvolutionBackpropDataNonConstPadValue) {
    Shape data_shape{1, 4, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 3, 1});
        std::shared_ptr<Node> pad_value = Constant::create(element::f16, Shape{}, {0});
        pad_value = std::make_shared<Convert>(pad_value, element::f32);
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(pad,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{3, 2},
                                                                   CoordinateDiff{4, 3},
                                                                   Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);
        auto filters = std::make_shared<Parameter>(element::f32, Shape{2, 2, 1, 5, 5});
        auto conv = std::make_shared<GroupConvolutionBackpropData>(data,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{2, 1},
                                                                   CoordinateDiff{1, 2},
                                                                   Shape{1, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadFusionNonConstantPadMode) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::REFLECT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadFusionNonZeroPadValue) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadFusionPadForBatchSize) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {0});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {1, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad_value = Constant::create(element::i32, Shape{}, {0});
        auto pad = pad_factory->create(data, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadFusionAvgPoolExcludePadNonZeroPads) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        model = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 1, 1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto avg_pool = std::make_shared<AvgPool>(pad,
                                                  Strides{1, 1},
                                                  Shape{0, 0},
                                                  Shape{1, 1},
                                                  Shape{4, 4},
                                                  true,
                                                  op::RoundingType::FLOOR);
        model_ref = std::make_shared<Model>(NodeVector{avg_pool}, ParameterVector{data});
    }
}

TEST_BODY(NegativePadFusionConvolutionBackpropDataTooSmallPad) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);

        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    {
        auto data = std::make_shared<Parameter>(element::f32, data_shape);

        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, 2, 2});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        auto filters = std::make_shared<Parameter>(element::f32, Shape{3, 2, 5, 5});
        auto conv = std::make_shared<ConvolutionBackpropData>(pad,
                                                              filters,
                                                              Strides{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              CoordinateDiff{1, 1},
                                                              Shape{1, 1});

        model_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
    }
}

TEST_BODY(NegativePadPreservation) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<Parameter>(element::i32, data_shape);
        auto pads_begin = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pads_end = Constant::create(element::i32, Shape{4}, {0, 0, -1, -1});
        auto pad = pad_factory->create(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        auto filters = std::make_shared<Parameter>(element::i32, Shape{1, 3, 4, 4});
        auto conv = std::make_shared<Convolution>(pad,
                                                  filters,
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{1, 1},
                                                  Shape{1, 1});
        model = std::make_shared<Model>(NodeVector{conv}, ParameterVector{data, filters});
        manager.register_pass<ov::pass::PadFusion>();
    }
    // Reference function is equal to function
}

namespace {

#undef CREATE_MODEL_FACTORY
#define CREATE_MODEL_FACTORY(type_name) std::make_shared<type_name>()

std::vector<TestModelFactoryPtr> model_factories = {
    CREATE_MODEL_FACTORY(PadElimination),
    CREATE_MODEL_FACTORY(PadFusionAvgPoolExcludePad),
    CREATE_MODEL_FACTORY(PadFusionAvgPoolDontExcludePad),
    CREATE_MODEL_FACTORY(PadFusionConvolution),
    CREATE_MODEL_FACTORY(PadFusionConvolutionBackpropData),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolution),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolutionBackpropData),
    CREATE_MODEL_FACTORY(PadFusionAvgPoolNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionConvolutionNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionConvolutionBackpropDataNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolutionNonConstPadValue),
    CREATE_MODEL_FACTORY(PadFusionGroupConvolutionBackpropDataNonConstPadValue),
    CREATE_MODEL_FACTORY(NegativePadFusionNonConstantPadMode),
    CREATE_MODEL_FACTORY(NegativePadFusionNonZeroPadValue),
    CREATE_MODEL_FACTORY(NegativePadFusionPadForBatchSize),
    CREATE_MODEL_FACTORY(NegativePadFusionAvgPoolExcludePadNonZeroPads),
    CREATE_MODEL_FACTORY(NegativePadFusionConvolutionBackpropDataTooSmallPad),
    CREATE_MODEL_FACTORY(NegativePadPreservation),
    CREATE_MODEL_FACTORY(NegativePadElimination),
    CREATE_MODEL_FACTORY(NegativePadFusionAvgPoolExcludePad),
    CREATE_MODEL_FACTORY(NegativePadFusionAvgPoolDontExcludePad),
    CREATE_MODEL_FACTORY(NegativePadFusionConvolution),
    CREATE_MODEL_FACTORY(NegativePadFusionGroupConvolution)};

#undef CREATE_PAD_FACTORY
#define CREATE_PAD_FACTORY(type_name, type_str) CreatePadFactory<type_name>(type_str)

std::vector<PadFactoryPtr> pad_factories = {CREATE_PAD_FACTORY(ov::op::v1::Pad, "op_v1_Pad"),
                                            CREATE_PAD_FACTORY(ov::op::v12::Pad, "op_v12_Pad")};

}  // namespace

INSTANTIATE_TEST_SUITE_P(PadTestSuite,
                         PadFusionTestFixture,
                         ::testing::Combine(::testing::ValuesIn(pad_factories), ::testing::ValuesIn(model_factories)),
                         PadFusionTestFixture::get_test_name);
