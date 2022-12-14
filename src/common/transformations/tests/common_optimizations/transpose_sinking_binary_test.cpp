// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_binary.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<ov::Model>;
using Output = ov::Output<ov::Node>;

// ----------------------------------------------------------------------------

class IBinaryFactory {
public:
    IBinaryFactory() = default;
    virtual ~IBinaryFactory() = default;
    virtual NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const = 0;
};

using BinaryFactoryPtr = std::shared_ptr<IBinaryFactory>;

template <typename BinaryT>
class BinaryFactory : public IBinaryFactory {
public:
    BinaryFactory() = default;
    NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const override {
        return std::make_shared<BinaryT>(parent_left_node, parent_right_node);
    }
};

template <typename BinaryT>
BinaryFactoryPtr CreateBinaryFactory() {
    return std::make_shared<BinaryFactory<BinaryT>>();
}

// ----------------------------------------------------------------------------

class IPassFactory {
public:
    IPassFactory() = default;
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
};

using PassFactoryPtr = std::shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    void registerPass(ov::pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};

template <typename PassT>
PassFactoryPtr CreatePassFactory() {
    return std::make_shared<PassFactory<PassT>>();
}

std::vector<BinaryFactoryPtr> binary_factories = {CreateBinaryFactory<ov::opset9::Add>(),
                                                  CreateBinaryFactory<ov::opset9::Divide>(),
                                                  CreateBinaryFactory<ov::opset9::Maximum>(),
                                                  CreateBinaryFactory<ov::opset9::Minimum>(),
                                                  CreateBinaryFactory<ov::opset9::Mod>(),
                                                  CreateBinaryFactory<ov::opset9::Multiply>(),
                                                  CreateBinaryFactory<ov::opset9::Power>(),
                                                  CreateBinaryFactory<ov::opset9::SquaredDifference>(),
                                                  CreateBinaryFactory<ov::opset9::Subtract>()};

std::vector<size_t> binary_operations_numbers = {1, 10};

std::vector<size_t> binary_transpose_input_indexes = {0, 1};

}  // namespace

namespace binary {
namespace single_consumer {
namespace forward {
namespace one_input_transpose {

std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type,
                                          size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type,
                                                   size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});

        auto transpose_reversed_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, transpose_reversed);
        else
            in_op = binary_factory->create(transpose_reversed, in_op);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {
std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

        in_op = binary_factory->create(in_op, transpose1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

        auto transpose_reversed_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(transpose1, transpose_reversed_const);

        in_op = binary_factory->create(in_op, transpose_reversed);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace double_transpose
}  // namespace forward

namespace backward {
namespace one_input_transpose {
std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type,
                                          size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type,
                                                   size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, transpose);
        else
            in_op = binary_factory->create(transpose, in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}
}  // namespace one_input_transpose
}  // namespace backward
}  // namespace single_consumer
}  // namespace binary

using CreateGraphBinaryF = std::function<std::shared_ptr<ov::Model>(BinaryFactoryPtr unary_factory,
                                                                    size_t num_binary_ops,
                                                                    ov::element::Type input_type,
                                                                    size_t binary_transpose_input_idx)>;

using TestBinaryParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    size_t,             /* num_binary_ops */
                                    CreateGraphBinaryF, /* model_factory */
                                    CreateGraphBinaryF, /* reference_model_factory */
                                    ov::element::Type,  /* input type */
                                    size_t>;            /* binary_transpose_input_idx */

class TransposeSinkingBinaryTestFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                          public TransformationTestsF {};

TEST_P(TransposeSinkingBinaryTestFixture, CompareFunctions) {
    BinaryFactoryPtr unary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryF model_factory;
    CreateGraphBinaryF reference_model_factory;
    ov::element::Type input_type;
    size_t binary_transpose_input_idx;
    std::tie(unary_factory,
             pass_factory,
             num_binary_ops,
             model_factory,
             reference_model_factory,
             input_type,
             binary_transpose_input_idx) = this->GetParam();

    model = model_factory(unary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    model_ref = reference_model_factory(unary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryForwardTestSuite,
    TransposeSinkingBinaryTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_factories),
        ::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingBinaryElementwiseForward>()),
        ::testing::ValuesIn(binary_operations_numbers),
        ::testing::Values(binary::single_consumer::forward::one_input_transpose::CreateFunction),
        ::testing::Values(binary::single_consumer::forward::one_input_transpose::CreateReferenceFunction),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)));

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryBackwardTestSuite,
    TransposeSinkingBinaryTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_factories),
        ::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingBinaryElementwiseBackward>()),
        ::testing::ValuesIn(binary_operations_numbers),
        ::testing::Values(binary::single_consumer::backward::one_input_transpose::CreateFunction),
        ::testing::Values(binary::single_consumer::backward::one_input_transpose::CreateReferenceFunction),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)));

// --------------------------------------------------------------------------------------

using CreateGraphBinaryTwoTransposeInputsF = std::function<
    std::shared_ptr<ov::Model>(BinaryFactoryPtr unary_factory, size_t num_binary_ops, ov::element::Type input_type)>;

using TestBinaryTwoTransposeInputsParams =
    std::tuple<BinaryFactoryPtr,
               PassFactoryPtr,
               size_t,                               /* num_binary_ops */
               CreateGraphBinaryTwoTransposeInputsF, /* model_factory */
               CreateGraphBinaryTwoTransposeInputsF, /* reference_model_factory */
               ov::element::Type>;                   /* input type */

class TransposeSinkingBinaryTwoTransposeInputsTestFixture
    : public ::testing::WithParamInterface<TestBinaryTwoTransposeInputsParams>,
      public TransformationTestsF {};

TEST_P(TransposeSinkingBinaryTwoTransposeInputsTestFixture, CompareFunctions) {
    BinaryFactoryPtr unary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryTwoTransposeInputsF model_factory;
    CreateGraphBinaryTwoTransposeInputsF reference_model_factory;
    ov::element::Type input_type;

    std::tie(unary_factory, pass_factory, num_binary_ops, model_factory, reference_model_factory, input_type) =
        this->GetParam();

    model = model_factory(unary_factory, num_binary_ops, input_type);
    model_ref = reference_model_factory(unary_factory, num_binary_ops, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryTwoTransposeInputsForwardTestSuite,
    TransposeSinkingBinaryTwoTransposeInputsTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_factories),
                       ::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingBinaryElementwiseForward>()),
                       ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(binary::single_consumer::forward::double_transpose::CreateFunction),
                       ::testing::Values(binary::single_consumer::forward::double_transpose::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32)));
