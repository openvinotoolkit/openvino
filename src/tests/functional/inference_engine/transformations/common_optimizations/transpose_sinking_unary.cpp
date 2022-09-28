// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations//transpose_sinking_unary.hpp"

#include <transformations/init_node_info.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

#include "gtest/gtest.h"

using NodePtr = std::shared_ptr<ov::Node>;

class IUnaryFactory {
public:
    IUnaryFactory() = default;
    virtual ~IUnaryFactory() = default;
    virtual NodePtr create(NodePtr parent_node) const = 0;
};

using UnaryFactoryPtr = std::shared_ptr<IUnaryFactory>;

template <typename UnaryT>
class UnaryFactory : public IUnaryFactory {
public:
    UnaryFactory() = default;
    NodePtr create(NodePtr parent_node) const override {
        return std::make_shared<UnaryT>(parent_node);
    }
};

template <>
NodePtr UnaryFactory<ov::opset9::Elu>::create(NodePtr parent_node) const
{
    return std::make_shared<ov::opset9::Elu>(parent_node, 0.1);
}

template <>
NodePtr UnaryFactory<ov::opset9::Clamp>::create(NodePtr parent_node) const
{
    return std::make_shared<ov::opset9::Clamp>(parent_node, 0.1, 0.2);
}

template <>
NodePtr UnaryFactory<ov::opset9::Convert>::create(NodePtr parent_node) const
{
    return std::make_shared<ov::opset9::Convert>(parent_node, ov::element::f64);
}

template <typename UnaryT>
UnaryFactoryPtr CreateUnaryFactory()
{
    return std::make_shared<UnaryFactory<UnaryT>>();
}

// ----------------------------------------------------------------------------

class IPassManagerFactory {
public:
    IPassManagerFactory() = default;
    virtual ~IPassManagerFactory() = default;
    virtual ov::pass::Manager createManager() const = 0;
};

using PassManagerFactoryPtr = std::shared_ptr<IPassManagerFactory>;

template <typename PassT>
class PassManagerFactory : public IPassManagerFactory {
public:
    ov::pass::Manager createManager() const override;
};

template <typename PassT>
ov::pass::Manager PassManagerFactory<PassT>::createManager() const
{
    ov::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<PassT>();
    return manager;
}

template <typename PassT>
PassManagerFactoryPtr CreatePassManagerFactory()
{
    return std::make_shared<PassManagerFactory<PassT>>();
}

// ----------------------------------------------------------------------------

using FloatPtr = std::unique_ptr<float[]>;

using CreateGraphF = std::function< std::shared_ptr<ov::Model> (UnaryFactoryPtr unary_factory) >;

class TransposeSinkingUnaryTestFixture: public CommonTestUtils::TestsCommon,
                                        public ::testing::WithParamInterface<std::tuple<UnaryFactoryPtr,
                                                                                        PassManagerFactoryPtr,
                                                                                        CreateGraphF /* function factory */,
                                                                                        CreateGraphF /* reference_function factory */>> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ov::Model> model, reference_model;
    ov::pass::Manager pass_manager;
};

void TransposeSinkingUnaryTestFixture::SetUp() {
    // TODO: use auto & [ ... ] = this->GetParam() when C++17
    UnaryFactoryPtr unary_factory;
    PassManagerFactoryPtr pass_manager_factory;
    CreateGraphF model_factory;
    CreateGraphF reference_model_factory;
    std::tie(unary_factory, pass_manager_factory, model_factory, reference_model_factory) = this->GetParam();

    model = model_factory(unary_factory);
    reference_model = reference_model_factory(unary_factory);
    pass_manager = pass_manager_factory->createManager();
}

namespace {

std::string GetFinalNodeName(std::shared_ptr<ov::Model> model, int index = 0)
{
    NodePtr result_node = model->get_results()[index];
    return result_node->get_input_node_ptr(0)->get_friendly_name();
}

void execute_test(std::shared_ptr<ov::Model> model,
                  std::shared_ptr<ov::Model> reference_model,
                  ov::pass::Manager pass_manager)
{
    std::shared_ptr<ov::Model> original_model = model->clone();

    pass_manager.run_passes(model);
    ASSERT_NO_THROW(check_rt_info(model));
    
    EXPECT_EQ(GetFinalNodeName(model), GetFinalNodeName(original_model));

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(model, reference_model);
    ASSERT_TRUE(result.valid) << result.message;
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory)
{
        ov::Shape input_shape{1, 96, 55, 55};
        auto input_type = ov::element::f32;

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        auto elu = unary_factory->create(transpose0);
        return std::make_shared<ov::Model>(elu, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(UnaryFactoryPtr unary_factory)
{
        ov::Shape input_shape{1, 96, 55, 55};
        auto input_type = ov::element::f32;

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto elu = unary_factory->create(X);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(elu, ng_order0);

        return std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
}

std::vector<UnaryFactoryPtr> unary_factories = {
    CreateUnaryFactory<ov::opset9::Clamp>(),
    CreateUnaryFactory<ov::opset9::Elu>(),
    CreateUnaryFactory<ov::opset9::SoftPlus>(),
    CreateUnaryFactory<ov::opset9::LogicalNot>(),
    CreateUnaryFactory<ov::opset9::Convert>(),
    CreateUnaryFactory<ov::opset9::Abs>(),
    CreateUnaryFactory<ov::opset9::Acos>(),
    CreateUnaryFactory<ov::opset9::Asin>(),
    CreateUnaryFactory<ov::opset9::Asinh>(),
    CreateUnaryFactory<ov::opset9::Atan>(),
    CreateUnaryFactory<ov::opset9::Ceiling>(),
    CreateUnaryFactory<ov::opset9::Cos>(),
    CreateUnaryFactory<ov::opset9::Cosh>(),
    CreateUnaryFactory<ov::opset9::Erf>(),
    CreateUnaryFactory<ov::opset9::Exp>(),
    CreateUnaryFactory<ov::opset9::Gelu>(),
    CreateUnaryFactory<ov::opset9::HSigmoid>(),
    CreateUnaryFactory<ov::opset9::HSwish>(),
    CreateUnaryFactory<ov::opset9::Log>(),
    CreateUnaryFactory<ov::opset9::Negative>(),
    CreateUnaryFactory<ov::opset9::Relu>(),
    CreateUnaryFactory<ov::opset9::Sigmoid>(),
    CreateUnaryFactory<ov::opset9::Sign>(),
    CreateUnaryFactory<ov::opset9::Sin>(),
    CreateUnaryFactory<ov::opset9::Sinh>(),
    CreateUnaryFactory<ov::opset9::SoftSign>(),
    CreateUnaryFactory<ov::opset9::Sqrt>(),
    CreateUnaryFactory<ov::opset9::Tan>(),
    CreateUnaryFactory<ov::opset9::Tanh>()
};

} // namespace

TEST_P(TransposeSinkingUnaryTestFixture, CompareFunctions) {
    execute_test(model, reference_model, pass_manager);
}

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryForwardTestSuite, TransposeSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CreatePassManagerFactory<ov::pass::TransposeSinkingUnaryForward>()),
                                            ::testing::Values(CreateFunctionTransposeBefore),
                                            ::testing::Values(CreateFunctionTransposeAfter)));

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryBackwardTestSuite, TransposeSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CreatePassManagerFactory<ov::pass::TransposeSinkingUnaryBackward>()),
                                            ::testing::Values(CreateFunctionTransposeAfter),
                                            ::testing::Values(CreateFunctionTransposeBefore)));
