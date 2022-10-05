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
NodePtr UnaryFactory<ov::opset9::Elu>::create(NodePtr parent_node) const {
    return std::make_shared<ov::opset9::Elu>(parent_node, 0.1);
}

template <>
NodePtr UnaryFactory<ov::opset9::Clamp>::create(NodePtr parent_node) const {
    return std::make_shared<ov::opset9::Clamp>(parent_node, 0.1, 0.2);
}

template <>
NodePtr UnaryFactory<ov::opset9::Convert>::create(NodePtr parent_node) const {
    return std::make_shared<ov::opset9::Convert>(parent_node, ov::element::f64);
}

template <typename UnaryT>
UnaryFactoryPtr CreateUnaryFactory() {
    return std::make_shared<UnaryFactory<UnaryT>>();
}

// ----------------------------------------------------------------------------

class IPassFactory {
public:
    IPassFactory() = default;
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager & pass_manager) const = 0;
};

using PassFactoryPtr = std::shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    void registerPass(ov::pass::Manager & pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};

template <typename PassT>
PassFactoryPtr CreatePassFactory() {
    return std::make_shared<PassFactory<PassT>>();
}

// ----------------------------------------------------------------------------

using FloatPtr = std::unique_ptr<float[]>;

using CreateGraphF = std::function< std::shared_ptr<ov::Model> (UnaryFactoryPtr unary_factory,
                                                                size_t num_unary_ops,
                                                                const ov::Shape & input_shape,
                                                                ov::element::Type input_type) >;

using TestParams = std::tuple<UnaryFactoryPtr,
                              PassFactoryPtr,
                              size_t, /* num_unary_ops */
                              CreateGraphF, /* model_factory */
                              CreateGraphF, /* reference_model_factory */
                              ov::Shape, /* input shape */
                              ov::element::Type>; /* input type */

class TransposeSinkingUnaryTestFixture: public ::testing::WithParamInterface<TestParams>,
                                        public TransformationTestsF {};

namespace {

std::string GetFinalNodeName(std::shared_ptr<ov::Model> model, int index = 0) {
    NodePtr result_node = model->get_results()[index];
    return result_node->get_input_node_ptr(0)->get_friendly_name();
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const ov::Shape & input_shape,
                                                         ov::element::Type input_type) {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = unary_factory->create(in_op);
        }

        return std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const ov::Shape & input_shape,
                                                         ov::element::Type input_type) {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = unary_factory->create(in_op);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        return std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeMultConsumers(UnaryFactoryPtr unary_factory,
                                                                size_t num_unary_ops,
                                                                const ov::Shape & input_shape,
                                                                ov::element::Type input_type) {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = unary_factory->create(in_op);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        auto cosh = std::make_shared<ov::opset9::Cosh>(in_op);

        return std::make_shared<ov::Model>(ov::OutputVector{transpose0, cosh}, ov::ParameterVector{X});
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

std::vector<size_t> unary_operations_numbers = {1, 10};

} // namespace

TEST_P(TransposeSinkingUnaryTestFixture, CompareFunctions) {
    UnaryFactoryPtr unary_factory;
    PassFactoryPtr pass_factory;
    size_t num_unary_ops;
    CreateGraphF model_factory;
    CreateGraphF reference_model_factory;
    ov::Shape input_shape;
    ov::element::Type input_type;
    std::tie(unary_factory,
             pass_factory,
             num_unary_ops,
             model_factory,
             reference_model_factory,
             input_shape,
             input_type) = this->GetParam();

    model = model_factory(unary_factory, num_unary_ops, input_shape, input_type);
    model_ref = reference_model_factory(unary_factory, num_unary_ops, input_shape, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryForwardTestSuite, TransposeSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingUnaryForward>()),
                                            ::testing::ValuesIn(unary_operations_numbers),
                                            ::testing::Values(CreateFunctionTransposeBefore),
                                            ::testing::Values(CreateFunctionTransposeAfter),
                                            ::testing::Values(ov::Shape{1, 96, 55, 55}),
                                            ::testing::Values(ov::element::f32)));

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryBackwardTestSuite, TransposeSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingUnaryBackward>()),
                                            ::testing::ValuesIn(unary_operations_numbers),
                                            ::testing::Values(CreateFunctionTransposeAfter),
                                            ::testing::Values(CreateFunctionTransposeBefore),
                                            ::testing::Values(ov::Shape{1, 96, 55, 55}),
                                            ::testing::Values(ov::element::f32)));

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryBackwardMultConsumersTestSuite, TransposeSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingUnaryBackward>()),
                                            ::testing::ValuesIn(unary_operations_numbers),
                                            ::testing::Values(CreateFunctionTransposeMultConsumers),
                                            ::testing::Values(CreateFunctionTransposeMultConsumers),
                                            ::testing::Values(ov::Shape{1, 96, 55, 55}),
                                            ::testing::Values(ov::element::f32)));
