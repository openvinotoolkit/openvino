// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_binary.hpp"

#include <functional>

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing;
using namespace transpose_sinking::testing::utils;

namespace {

template <typename BinaryT>
class BinaryFactory : public IFactory {
public:
    explicit BinaryFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& inputs) const override {
        return std::make_shared<BinaryT>(inputs[0], inputs[1]);
    }
};

template <typename BinaryT>
FactoryPtr CreateBinaryFactory(const std::string& type_name) {
    return std::make_shared<BinaryFactory<BinaryT>>(type_name);
}

// ----------------------------------------------------------------------------

#undef CREATE_BINARY_FACTORY
#define CREATE_BINARY_FACTORY(type_name) CreateBinaryFactory<type_name>(#type_name)

/*
 * binary operations without PRelu
 * PRelu input(1) is special constant input that is important for some tests. Specially for the
 * Unsqueeze insertion
 */
std::vector<FactoryPtr> binary_elementwise_factories = {CREATE_BINARY_FACTORY(Add),
                                                        CREATE_BINARY_FACTORY(Divide),
                                                        CREATE_BINARY_FACTORY(Maximum),
                                                        CREATE_BINARY_FACTORY(Minimum),
                                                        CREATE_BINARY_FACTORY(Mod),
                                                        CREATE_BINARY_FACTORY(Multiply),
                                                        CREATE_BINARY_FACTORY(Power),
                                                        CREATE_BINARY_FACTORY(SquaredDifference),
                                                        CREATE_BINARY_FACTORY(Subtract)};

std::vector<FactoryPtr> binary_factories = {CREATE_BINARY_FACTORY(Add),
                                            CREATE_BINARY_FACTORY(Divide),
                                            CREATE_BINARY_FACTORY(Maximum),
                                            CREATE_BINARY_FACTORY(Minimum),
                                            CREATE_BINARY_FACTORY(Mod),
                                            CREATE_BINARY_FACTORY(Multiply),
                                            CREATE_BINARY_FACTORY(Power),
                                            CREATE_BINARY_FACTORY(SquaredDifference),
                                            CREATE_BINARY_FACTORY(Subtract),
                                            CREATE_BINARY_FACTORY(PRelu)};

std::vector<size_t> binary_operations_numbers = {1, 10};

std::vector<size_t> binary_transpose_input_indexes = {0, 1};

}  // namespace

namespace transpose_sinking {
namespace testing {
namespace binary {

namespace single_consumer {
namespace forward {
namespace one_input_transpose {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      size_t num_binary_ops,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create({in_op, in_constant});
        else
            in_op = binary_factory->create({in_constant, in_op});
    }

    return std::make_shared<Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               size_t num_binary_ops,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

        auto transpose_reversed_const = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<Transpose>(in_constant, transpose_reversed_const);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create({in_op, transpose_reversed});
        else
            in_op = binary_factory->create({transpose_reversed, in_op});
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {
std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory, size_t num_binary_ops, element::Type input_type) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);

        in_op = binary_factory->create({in_op, transpose1});
    }

    return std::make_shared<Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               size_t num_binary_ops,
                                               element::Type input_type) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

        auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);

        auto transpose_reversed_const = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<Transpose>(transpose1, transpose_reversed_const);

        in_op = binary_factory->create({in_op, transpose_reversed});
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

using CreateGraphBinaryTwoTransposeInputsF =
    std::function<std::shared_ptr<Model>(FactoryPtr binary_factory, size_t num_binary_ops, element::Type input_type)>;

using TestBinaryTwoTransposeInputsParams =
    std::tuple<FactoryPtr,
               PassFactoryPtr,
               size_t,                               /* num_binary_ops */
               CreateGraphBinaryTwoTransposeInputsF, /* model_factory */
               CreateGraphBinaryTwoTransposeInputsF, /* reference_model_factory */
               element::Type>;                       /* input type */

class TransposeSinkingBinaryTwoTransposeInputsTestFixture
    : public ::testing::WithParamInterface<TestBinaryTwoTransposeInputsParams>,
      public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestBinaryTwoTransposeInputsParams>& obj) {
        FactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        size_t num_binary_ops;
        CreateGraphBinaryTwoTransposeInputsF model_factory;
        CreateGraphBinaryTwoTransposeInputsF reference_model_factory;
        element::Type input_type;

        std::tie(binary_factory, pass_factory, num_binary_ops, model_factory, reference_model_factory, input_type) =
            obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "numBinaryOps=" << num_binary_ops << "/";
        test_name << "inputType=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingBinaryTwoTransposeInputsTestFixture, CompareFunctions) {
    FactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryTwoTransposeInputsF model_factory;
    CreateGraphBinaryTwoTransposeInputsF reference_model_factory;
    element::Type input_type;

    std::tie(binary_factory, pass_factory, num_binary_ops, model_factory, reference_model_factory, input_type) =
        this->GetParam();

    model = model_factory(binary_factory, num_binary_ops, input_type);
    model_ref = reference_model_factory(binary_factory, num_binary_ops, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(TransposeSinkingBinaryTwoTransposeInputsForwardTestSuite,
                         TransposeSinkingBinaryTwoTransposeInputsTestFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(TSBinaryForward)),
                                            ::testing::ValuesIn(binary_operations_numbers),
                                            ::testing::Values(CreateFunction),
                                            ::testing::Values(CreateReferenceFunction),
                                            ::testing::Values(element::f32)),
                         TransposeSinkingBinaryTwoTransposeInputsTestFixture::get_test_name);

}  // namespace double_transpose
}  // namespace forward

namespace backward {
namespace one_input_transpose {
std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      size_t num_binary_ops,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create({in_op, in_constant});
        else
            in_op = binary_factory->create({in_constant, in_op});
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               size_t num_binary_ops,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh = std::make_shared<Tanh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<Transpose>(in_constant, ng_order);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create({in_op, transpose});
        else
            in_op = binary_factory->create({transpose, in_op});
    }

    return std::make_shared<Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

using CreateGraphBinaryF = std::function<std::shared_ptr<Model>(FactoryPtr binary_factory,
                                                                size_t num_binary_ops,
                                                                element::Type input_type,
                                                                size_t binary_transpose_input_idx)>;

using TestBinaryParams = std::tuple<FactoryPtr,
                                    PassFactoryPtr,
                                    size_t,             /* num_binary_ops */
                                    CreateGraphBinaryF, /* model_factory */
                                    CreateGraphBinaryF, /* reference_model_factory */
                                    element::Type,      /* input type */
                                    size_t>;            /* binary_transpose_input_idx */

class TransposeSinkingBinaryTestFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                          public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestBinaryParams>& obj) {
        FactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        size_t num_binary_ops;
        CreateGraphBinaryF model_factory;
        CreateGraphBinaryF reference_model_factory;
        element::Type input_type;
        size_t binary_transpose_input_idx;

        std::tie(binary_factory,
                 pass_factory,
                 num_binary_ops,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 binary_transpose_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "numBinaryOps=" << num_binary_ops << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingBinaryTestFixture, CompareFunctions) {
    FactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryF model_factory;
    CreateGraphBinaryF reference_model_factory;
    element::Type input_type;
    size_t binary_transpose_input_idx;
    std::tie(binary_factory,
             pass_factory,
             num_binary_ops,
             model_factory,
             reference_model_factory,
             input_type,
             binary_transpose_input_idx) = this->GetParam();

    model = model_factory(binary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    model_ref = reference_model_factory(binary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TSBinaryForwardTestSuite,
    TransposeSinkingBinaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(TSBinaryForward)),
                       ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TSBinaryBackwardTestSuite,
    TransposeSinkingBinaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(TSBinaryBackward)),
                       ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(single_consumer::backward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::backward::one_input_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryTestFixture::get_test_name);

// --------------------------------------------------------------------------------------

using CreateGraphBinaryIncompatShapesF = std::function<std::shared_ptr<Model>(FactoryPtr unary_factory,
                                                                              element::Type input_type,
                                                                              Shape input_shape,
                                                                              Shape constant_shape,
                                                                              size_t binary_transpose_input_idx)>;

using TestBinaryIncompatShapesParams = std::tuple<FactoryPtr,
                                                  PassFactoryPtr,
                                                  Shape,                            /* input shape */
                                                  Shape,                            /* constant_shape */
                                                  CreateGraphBinaryIncompatShapesF, /* model_factory */
                                                  CreateGraphBinaryIncompatShapesF, /* reference_model_factory */
                                                  element::Type,                    /* input type */
                                                  size_t>;                          /* binary_transpose_input_idx */

class TransposeSinkingBinaryIncompatShapesTestFixture
    : public ::testing::WithParamInterface<TestBinaryIncompatShapesParams>,
      public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestBinaryIncompatShapesParams>& obj) {
        FactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        Shape input_shape;
        Shape constant_shape;
        CreateGraphBinaryIncompatShapesF model_factory;
        CreateGraphBinaryIncompatShapesF reference_model_factory;
        element::Type input_type;
        size_t binary_transpose_input_idx;
        std::tie(binary_factory,
                 pass_factory,
                 input_shape,
                 constant_shape,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 binary_transpose_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "inputShape=" << to_string(input_shape) << "/";
        test_name << "constantShape=" << to_string(constant_shape) << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingBinaryIncompatShapesTestFixture, CompareFunctions) {
    FactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    Shape input_shape;
    Shape constant_shape;
    CreateGraphBinaryIncompatShapesF model_factory;
    CreateGraphBinaryIncompatShapesF reference_model_factory;
    element::Type input_type;
    size_t binary_transpose_input_idx;
    std::tie(binary_factory,
             pass_factory,
             input_shape,
             constant_shape,
             model_factory,
             reference_model_factory,
             input_type,
             binary_transpose_input_idx) = this->GetParam();

    model = model_factory(binary_factory, input_type, input_shape, constant_shape, binary_transpose_input_idx);
    model_ref =
        reference_model_factory(binary_factory, input_type, input_shape, constant_shape, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

namespace binary {
namespace single_consumer {
namespace backward {
namespace incompat_shapes {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      Shape input_shape,
                                      Shape constant_shape,
                                      size_t binary_transpose_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<Constant>(input_type, constant_shape, Shape{1});

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create({X, in_constant});
    else
        binary_op = binary_factory->create({in_constant, X});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary_op, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               Shape input_shape,
                                               Shape constant_shape,
                                               size_t binary_transpose_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto in_constant = std::make_shared<Constant>(input_type, constant_shape, Shape{1});

    std::vector<size_t> dims(input_shape.size() - constant_shape.size());
    std::iota(dims.begin(), dims.end(), 0);
    auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{dims.size()}, dims);
    auto unsqeeze = std::make_shared<Unsqueeze>(in_constant, unsqueeze_const);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(unsqeeze, ng_order1);

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create({transpose0, transpose1});
    else
        binary_op = binary_factory->create({transpose1, transpose0});

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::vector<Shape> constant_shapes = {Shape{96, 55, 55}, Shape{1}};

}  // namespace incompat_shapes
}  // namespace backward

namespace forward {
namespace incompat_shapes {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      Shape input_shape,
                                      Shape constant_shape,
                                      size_t binary_transpose_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<Constant>(input_type, constant_shape, Shape{1});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create({transpose0, in_constant});
    else
        binary_op = binary_factory->create({in_constant, transpose0});

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               Shape input_shape,
                                               Shape constant_shape,
                                               size_t binary_transpose_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<Constant>(input_type, constant_shape, Shape{1});

    std::vector<size_t> dims(input_shape.size() - constant_shape.size());
    std::iota(dims.begin(), dims.end(), 0);
    auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{dims.size()}, dims);
    auto unsqeeze = std::make_shared<Unsqueeze>(in_constant, unsqueeze_const);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
    auto transpose1 = std::make_shared<Transpose>(unsqeeze, ng_order1);

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create({X, transpose1});
    else
        binary_op = binary_factory->create({transpose1, X});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary_op, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::vector<Shape> constant_shapes = {Shape{55, 55, 96}, Shape{1}};

}  // namespace incompat_shapes
}  // namespace forward

}  // namespace single_consumer
}  // namespace binary

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryIncompatShapesBackwardTestSuite,
    TransposeSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(TSBinaryBackward)),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::ValuesIn(binary::single_consumer::backward::incompat_shapes::constant_shapes),
                       ::testing::Values(binary::single_consumer::backward::incompat_shapes::CreateFunction),
                       ::testing::Values(binary::single_consumer::backward::incompat_shapes::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryIncompatShapesForwardTestSuite,
    TransposeSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(TSBinaryForward)),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::ValuesIn(binary::single_consumer::forward::incompat_shapes::constant_shapes),
                       ::testing::Values(binary::single_consumer::forward::incompat_shapes::CreateFunction),
                       ::testing::Values(binary::single_consumer::forward::incompat_shapes::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingPReluIncompatShapesForwardTestSuite,
    TransposeSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(::testing::Values(CREATE_BINARY_FACTORY(PRelu)),
                       ::testing::Values(CREATE_PASS_FACTORY(TSBinaryForward)),
                       ::testing::Values(Shape{1, 3, 16, 16}),
                       ::testing::ValuesIn(std::vector<Shape>{Shape{3}}),
                       ::testing::Values(binary::single_consumer::forward::incompat_shapes::CreateFunction),
                       ::testing::Values(binary::single_consumer::forward::incompat_shapes::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::Values(0)),
    TransposeSinkingBinaryIncompatShapesTestFixture::get_test_name);

}  // namespace one_input_transpose
}  // namespace backward
}  // namespace single_consumer

namespace mult_consumers {
namespace forward {
namespace input_transpose_consumers {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto tanh = std::make_shared<Tanh>(transpose0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({transpose0, in_constant});
    else
        binary = binary_factory->create({in_constant, transpose0});

    return std::make_shared<Model>(ov::OutputVector{binary, tanh}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto tanh = std::make_shared<Tanh>(transpose0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

    auto transpose_reversed_const = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
    auto transpose_reversed = std::make_shared<Transpose>(in_constant, transpose_reversed_const);

    if (!binary_transpose_input_idx)
        binary = binary_factory->create({X, transpose_reversed});
    else
        binary = binary_factory->create({transpose_reversed, X});

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(binary, ng_order1);

    return std::make_shared<Model>(ov::OutputVector{transpose1, tanh}, ov::ParameterVector{X});
}

}  // namespace input_transpose_consumers

namespace output_consumers {

namespace one_binary {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({transpose0, in_constant});
    else
        binary = binary_factory->create({in_constant, transpose0});

    auto tanh1 = std::make_shared<Tanh>(binary);
    auto tanh2 = std::make_shared<Tanh>(binary);

    return std::make_shared<Model>(ov::OutputVector{tanh1, tanh2}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

    auto transpose_reversed_const = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
    auto transpose_reversed = std::make_shared<Transpose>(in_constant, transpose_reversed_const);

    if (!binary_transpose_input_idx)
        binary = binary_factory->create({X, transpose_reversed});
    else
        binary = binary_factory->create({transpose_reversed, X});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary, ng_order0);

    auto tanh1 = std::make_shared<Tanh>(transpose0);
    auto tanh2 = std::make_shared<Tanh>(transpose0);

    return std::make_shared<Model>(ov::OutputVector{tanh1, tanh2}, ov::ParameterVector{X});
}

}  // namespace one_binary

}  // namespace output_consumers

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({transpose0, in_constant});
    else
        binary = binary_factory->create({in_constant, transpose0});

    auto tanh = std::make_shared<Tanh>(X);

    return std::make_shared<Model>(ov::OutputVector{binary, tanh}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh = std::make_shared<Tanh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

    auto transpose_reversed_const = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
    auto transpose_reversed = std::make_shared<Transpose>(in_constant, transpose_reversed_const);

    if (!binary_transpose_input_idx)
        binary = binary_factory->create({X, transpose_reversed});
    else
        binary = binary_factory->create({transpose_reversed, X});

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(binary, ng_order1);

    return std::make_shared<Model>(ov::OutputVector{transpose1, tanh}, ov::ParameterVector{X});
}

}  // namespace input_node_consumers

}  // namespace forward

namespace backward {

namespace output_consumers {

namespace one_binary {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({tanh0, in_constant});
    else
        binary = binary_factory->create({in_constant, tanh0});

    auto tanh = std::make_shared<Tanh>(binary);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0, tanh}, ov::ParameterVector{X});
}

}  // namespace one_binary

namespace multiple_binaries {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};
    const size_t n_binaries = 10;

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    NodePtr in_op = tanh0;
    for (size_t i = 0; i < n_binaries; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create({in_op, in_constant});
        else
            in_op = binary_factory->create({in_constant, in_op});
    }

    auto tanh = std::make_shared<Tanh>(in_op);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0, tanh}, ov::ParameterVector{X});
}

}  // namespace multiple_binaries

}  // namespace output_consumers

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({tanh0, in_constant});
    else
        binary = binary_factory->create({in_constant, tanh0});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary, ng_order0);

    auto tanh1 = std::make_shared<Tanh>(tanh0);

    return std::make_shared<Model>(ov::OutputVector{transpose0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(tanh0, ng_order0);

    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose = std::make_shared<Transpose>(in_constant, ng_order);

    NodePtr binary;
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({transpose0, transpose});
    else
        binary = binary_factory->create({transpose, transpose0});

    auto tanh1 = std::make_shared<Tanh>(tanh0);

    return std::make_shared<Model>(ov::OutputVector{binary, tanh1}, ov::ParameterVector{X});
}

}  // namespace input_node_consumers

namespace output_transpose_mult_consumers {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({X, in_constant});
    else
        binary = binary_factory->create({in_constant, X});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary, ng_order0);

    auto tanh0 = std::make_shared<Tanh>(transpose0);
    auto tanh1 = std::make_shared<Tanh>(transpose0);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose = std::make_shared<Transpose>(in_constant, ng_order);

    NodePtr binary;
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({transpose0, transpose});
    else
        binary = binary_factory->create({transpose, transpose0});

    auto tanh0 = std::make_shared<Tanh>(binary);
    auto tanh1 = std::make_shared<Tanh>(binary);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace output_transpose_mult_consumers

namespace output_transpose_mult_transposes {

std::shared_ptr<Model> CreateFunction(FactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({X, in_constant});
    else
        binary = binary_factory->create({in_constant, X});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(binary, ng_order0);

    auto tanh0 = std::make_shared<Tanh>(transpose0);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(binary, ng_order1);

    auto tanh1 = std::make_shared<Tanh>(transpose1);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(FactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_transpose_input_idx) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose = std::make_shared<Transpose>(in_constant, ng_order);

    NodePtr binary;
    if (!binary_transpose_input_idx)
        binary = binary_factory->create({transpose0, transpose});
    else
        binary = binary_factory->create({transpose, transpose0});

    auto tanh0 = std::make_shared<Tanh>(binary);
    auto tanh1 = std::make_shared<Tanh>(binary);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace output_transpose_mult_transposes

}  // namespace backward

using CreateGraphF = std::function<
    std::shared_ptr<Model>(FactoryPtr binary_factory, element::Type input_type, size_t binary_transpose_input_idx)>;

struct CreateGraphFunctionDesc {
    CreateGraphFunctionDesc() = default;
    CreateGraphFunctionDesc(CreateGraphF a_model_factory,
                            CreateGraphF a_reference_model_factory,
                            std::string a_subtest_name)
        : model_factory(a_model_factory),
          reference_model_factory(a_reference_model_factory),
          subtest_name(a_subtest_name) {}
    CreateGraphF model_factory;
    CreateGraphF reference_model_factory;
    std::string subtest_name;
};

using TestBinaryParams = std::tuple<FactoryPtr,
                                    PassFactoryPtr,
                                    CreateGraphFunctionDesc,
                                    element::Type, /* input type */
                                    size_t>;       /*binary_transpose_input_idx*/

class TransposeBinaryMultiSinkingFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                           public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestBinaryParams>& obj) {
        FactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        CreateGraphFunctionDesc function_desc;
        element::Type input_type;
        size_t binary_transpose_input_idx;

        std::tie(binary_factory, pass_factory, function_desc, input_type, binary_transpose_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << function_desc.subtest_name << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeBinaryMultiSinkingFixture, CompareFunctions) {
    FactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    CreateGraphFunctionDesc function_desc;
    element::Type input_type;
    size_t binary_transpose_input_idx;

    std::tie(binary_factory, pass_factory, function_desc, input_type, binary_transpose_input_idx) = this->GetParam();

    model = function_desc.model_factory(binary_factory, input_type, binary_transpose_input_idx);
    model_ref = function_desc.reference_model_factory(binary_factory, input_type, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

#define SUBTEST(nmspace, subtest_name) \
    CreateGraphFunctionDesc(nmspace::CreateFunction, nmspace::CreateReferenceFunction, subtest_name)

std::vector<CreateGraphFunctionDesc> forward_subtests = {
    SUBTEST(forward::input_transpose_consumers, "forwardInputTransposeConsumers"),
    SUBTEST(forward::output_consumers::one_binary, "forwardOutputConsumers"),
    SUBTEST(forward::input_node_consumers, "forwardInputNodeConsumers")};

std::vector<CreateGraphFunctionDesc> backward_subtests = {
    SUBTEST(backward::input_node_consumers, "backwardInputNodeConsumers"),
    SUBTEST(backward::output_transpose_mult_consumers, "backwardOutputTransposeMultConsumers"),
    SUBTEST(backward::output_transpose_mult_transposes, "outputTransposeMultTransposes")};

#undef SUBTEST

INSTANTIATE_TEST_SUITE_P(TSBinaryForwardMultiConsumersTestSuite,
                         TransposeBinaryMultiSinkingFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(TSBinaryForward)),
                                            ::testing::ValuesIn(forward_subtests),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)),
                         TransposeBinaryMultiSinkingFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSBinaryBackwardMultiConsumersTestSuite,
                         TransposeBinaryMultiSinkingFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(TSBinaryBackward)),
                                            ::testing::ValuesIn(backward_subtests),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)),
                         TransposeBinaryMultiSinkingFixture::get_test_name);

namespace no_sinking {

struct CreateGraphFunctionDesc {
    CreateGraphFunctionDesc() = default;
    CreateGraphFunctionDesc(CreateGraphF a_model_factory, std::string a_subtest_name)
        : model_factory(a_model_factory),
          subtest_name(a_subtest_name) {}
    CreateGraphF model_factory;
    std::string subtest_name;
};

using TestBinaryParams = std::tuple<FactoryPtr,
                                    PassFactoryPtr,
                                    CreateGraphFunctionDesc,
                                    element::Type, /* input type */
                                    size_t>;       /*binary_transpose_input_idx*/

class TransposeBinaryMultiSinkingBinaryMultiConsumersFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                                               public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestBinaryParams>& obj) {
        FactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        CreateGraphFunctionDesc function_desc;
        element::Type input_type;
        size_t binary_transpose_input_idx;

        std::tie(binary_factory, pass_factory, function_desc, input_type, binary_transpose_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << function_desc.subtest_name << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeBinaryMultiSinkingBinaryMultiConsumersFixture, CompareFunctions) {
    FactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    CreateGraphFunctionDesc function_desc;
    element::Type input_type;
    size_t binary_transpose_input_idx;

    std::tie(binary_factory, pass_factory, function_desc, input_type, binary_transpose_input_idx) = this->GetParam();

    model = function_desc.model_factory(binary_factory, input_type, binary_transpose_input_idx);
    model_ref = model->clone();
    pass_factory->registerPass(manager);
}

#define SUBTEST(nmspace, subtest_name) CreateGraphFunctionDesc(nmspace::CreateFunction, subtest_name)

std::vector<CreateGraphFunctionDesc> backward_subtests_binary_consumers = {
    SUBTEST(backward::output_consumers::one_binary, "backwardOutputConsumersOneBinary"),
    SUBTEST(backward::output_consumers::multiple_binaries, "backwardOutputConsumersMultipleBinaries"),
};
#undef SUBTEST

INSTANTIATE_TEST_SUITE_P(TSBinaryBackwardBinaryMultiConsumersTestSuite,
                         TransposeBinaryMultiSinkingBinaryMultiConsumersFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(TSBinaryBackward)),
                                            ::testing::ValuesIn(backward_subtests_binary_consumers),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)),
                         TransposeBinaryMultiSinkingBinaryMultiConsumersFixture::get_test_name);

}  // namespace no_sinking

}  // namespace mult_consumers

}  // namespace binary
}  // namespace testing
}  // namespace transpose_sinking

TEST_F(TransformationTestsF, TSBinaryBackwardPReluSlabSpecial) {
    const Shape input_shape = {2, 3, 5, 5};
    const Shape slope_shape = {3};

    {
        auto X = std::make_shared<Parameter>(element::f32, input_shape);

        auto slope = std::make_shared<Constant>(element::f32, slope_shape, Shape{5, 7, 9});
        auto prelu = std::make_shared<PRelu>(X, slope);

        auto ts_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<Transpose>(prelu, ts_order);

        model = std::make_shared<Model>(ov::OutputVector{transpose}, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, input_shape);

        auto ts_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(X, ts_order0);

        auto slope = std::make_shared<Constant>(element::f32, slope_shape, Shape{1});

        std::vector<size_t> dims = {0, 2, 3};
        auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{dims.size()}, dims);
        auto unsqeeze = std::make_shared<Unsqueeze>(slope, unsqueeze_const);

        auto ts_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<Transpose>(unsqeeze, ts_order1);

        auto prelu = std::make_shared<PRelu>(transpose0, transpose1);

        model_ref = std::make_shared<Model>(ov::OutputVector{prelu}, ov::ParameterVector{X});
    }

    manager.register_pass<TSBinaryBackward>();
}

TEST_F(TransformationTestsF, TSBinaryBackwardPReluSlabNotSpecial) {
    const Shape input_shape = {2, 3, 5, 5};
    const Shape slope_shape = {5};

    {
        auto X = std::make_shared<Parameter>(element::f32, input_shape);

        auto slope = std::make_shared<Constant>(element::f32, slope_shape, Shape{5, 7, 9, 11, 15});
        auto prelu = std::make_shared<PRelu>(X, slope);

        auto ts_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<Transpose>(prelu, ts_order);

        model = std::make_shared<Model>(ov::OutputVector{transpose}, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, input_shape);

        auto ts_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(X, ts_order0);

        auto slope = std::make_shared<Constant>(element::f32, slope_shape, Shape{1});

        std::vector<size_t> dims = {0, 1, 2};
        auto unsqueeze_const = std::make_shared<Constant>(element::i64, Shape{dims.size()}, dims);
        auto unsqeeze = std::make_shared<Unsqueeze>(slope, unsqueeze_const);

        auto ts_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<Transpose>(unsqeeze, ts_order1);

        auto prelu = std::make_shared<PRelu>(transpose0, transpose1);

        model_ref = std::make_shared<Model>(ov::OutputVector{prelu}, ov::ParameterVector{X});
    }

    manager.register_pass<TSBinaryBackward>();
}

TEST_F(TransformationTestsF, TSBinaryBackwardPReluSlabSpecialRank1) {
    const Shape input_shape = {5};
    const Shape slope_shape = {5};

    {
        auto X = std::make_shared<Parameter>(element::f32, input_shape);

        auto slope = std::make_shared<Constant>(element::f32, slope_shape, Shape{5, 7, 9, 11, 15});
        auto prelu = std::make_shared<PRelu>(X, slope);

        auto ts_order = std::make_shared<Constant>(element::u64, Shape{1}, Shape{0});
        auto transpose = std::make_shared<Transpose>(prelu, ts_order);

        model = std::make_shared<Model>(ov::OutputVector{transpose}, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, input_shape);

        auto ts_order0 = std::make_shared<Constant>(element::u64, Shape{1}, Shape{0});
        auto transpose0 = std::make_shared<Transpose>(X, ts_order0);

        auto slope = std::make_shared<Constant>(element::f32, slope_shape, Shape{1});

        auto ts_order1 = std::make_shared<Constant>(element::u64, Shape{1}, Shape{0});
        auto transpose1 = std::make_shared<Transpose>(slope, ts_order1);

        auto prelu = std::make_shared<PRelu>(transpose0, transpose1);

        model_ref = std::make_shared<Model>(ov::OutputVector{prelu}, ov::ParameterVector{X});
    }

    manager.register_pass<TSBinaryBackward>();
}

TEST_F(TransformationTestsF, TSBinaryForwardDynamic) {
    auto X = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto ts_order = std::make_shared<Constant>(element::u64, Shape{0}, Shape{});
    auto transpose = std::make_shared<Transpose>(X, ts_order);

    auto c1 = std::make_shared<Constant>(element::f32, Shape{0}, Shape{});

    auto add = std::make_shared<Add>(transpose, c1);

    model = std::make_shared<Model>(ov::OutputVector{add}, ov::ParameterVector{X});
    model_ref = model->clone();

    manager.register_pass<TSBinaryForward>();
}
