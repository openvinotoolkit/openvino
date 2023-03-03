// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations//transpose_sinking_unary.hpp"

#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

using namespace ov;
using namespace ov::opset10;

namespace {
std::string to_string(const Shape& shape) {
    std::ostringstream result;
    result << "{";
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (idx)
            result << ",";
        result << shape[idx];
    }
    result << "}";
    return result.str();
}
}  // namespace

using NodePtr = std::shared_ptr<ov::Node>;

class IUnaryFactory {
public:
    IUnaryFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IUnaryFactory() = default;
    virtual NodePtr create(NodePtr parent_node) const = 0;

    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

using UnaryFactoryPtr = std::shared_ptr<IUnaryFactory>;

template <typename UnaryT>
class UnaryFactory : public IUnaryFactory {
public:
    UnaryFactory(const std::string& type_name) : IUnaryFactory(type_name) {}
    NodePtr create(NodePtr parent_node) const override {
        return std::make_shared<UnaryT>(parent_node);
    }
};

template <>
NodePtr UnaryFactory<Elu>::create(NodePtr parent_node) const {
    return std::make_shared<Elu>(parent_node, 0.1);
}

template <>
NodePtr UnaryFactory<Clamp>::create(NodePtr parent_node) const {
    return std::make_shared<Clamp>(parent_node, 0.1, 0.2);
}

template <>
NodePtr UnaryFactory<Convert>::create(NodePtr parent_node) const {
    return std::make_shared<Convert>(parent_node, element::f64);
}

template <typename UnaryT>
UnaryFactoryPtr CreateUnaryFactory(const std::string& type_name) {
    return std::make_shared<UnaryFactory<UnaryT>>(type_name);
}

// ----------------------------------------------------------------------------

class IPassFactory {
public:
    IPassFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

using PassFactoryPtr = std::shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    PassFactory(const std::string& type_name) : IPassFactory(type_name) {}
    void registerPass(ov::pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};

#define CREATE_PASS_FACTORY(pass_name) std::make_shared<PassFactory<ov::pass::pass_name>>(#pass_name)

#undef CREATE_UNARY_FACTORY
#define CREATE_UNARY_FACTORY(type_name) CreateUnaryFactory<type_name>(#type_name)

// ----------------------------------------------------------------------------

using FloatPtr = std::unique_ptr<float[]>;

using CreateGraphF = std::function<std::shared_ptr<ov::Model>(UnaryFactoryPtr unary_factory,
                                                              size_t num_unary_ops,
                                                              const Shape& input_shape,
                                                              element::Type input_type)>;

using TestParams = std::tuple<UnaryFactoryPtr,
                              PassFactoryPtr,
                              size_t,         /* num_unary_ops */
                              CreateGraphF,   /* model_factory */
                              CreateGraphF,   /* reference_model_factory */
                              Shape,          /* input shape */
                              element::Type>; /* input type */

class TransposeSinkingUnaryTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestParams>& obj) {
        UnaryFactoryPtr unary_factory;
        PassFactoryPtr pass_factory;
        size_t num_unary_ops;
        CreateGraphF model_factory;
        CreateGraphF reference_model_factory;
        Shape input_shape;
        element::Type input_type;
        std::tie(unary_factory,
                 pass_factory,
                 num_unary_ops,
                 model_factory,
                 reference_model_factory,
                 input_shape,
                 input_type) = obj.param;

        std::ostringstream test_name;
        test_name << "unaryFactory=" << unary_factory->getTypeName() << "/";
        test_name << "numUnaryOps=" << num_unary_ops << "/";
        test_name << "inputShape=" << to_string(input_shape) << "/";
        test_name << "unaryFactory=" << unary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "inputType=" << input_type;

        return test_name.str();
    }
};

namespace {

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    return std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(UnaryFactoryPtr unary_factory,
                                                        size_t num_unary_ops,
                                                        const Shape& input_shape,
                                                        element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
}

static NodePtr CreateReshape(NodePtr parent_node, const Shape& input_shape) {
    const size_t mul = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
    auto reshape_const = std::make_shared<Constant>(element::u64, Shape{1}, Shape{mul});
    return std::make_shared<Reshape>(parent_node, reshape_const, false);
}

namespace mult_consumers_last_node {
namespace with_reshape {

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(UnaryFactoryPtr unary_factory,
                                                        size_t num_unary_ops,
                                                        const Shape& input_shape,
                                                        element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    auto reshape1 = CreateReshape(transpose0, input_shape);
    auto reshape2 = CreateReshape(transpose0, input_shape);

    return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto reshape1 = CreateReshape(in_op, input_shape);
    auto reshape2 = CreateReshape(in_op, input_shape);

    return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2}, ov::ParameterVector{X});
}
}  // namespace with_reshape

namespace with_eltwise {

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(UnaryFactoryPtr unary_factory,
                                                        size_t num_unary_ops,
                                                        const Shape& input_shape,
                                                        element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto sinh = std::make_shared<Sinh>(in_op);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(sinh, ng_order0);

    auto cosh = std::make_shared<Cosh>(in_op);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(cosh, ng_order1);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0, transpose1}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto sinh = std::make_shared<Sinh>(in_op);
    auto cosh = std::make_shared<Cosh>(in_op);

    return std::make_shared<ov::Model>(ov::OutputVector{sinh, cosh}, ov::ParameterVector{X});
}

}  // namespace with_eltwise
}  // namespace mult_consumers_last_node

namespace mult_consumers_first_node {
namespace backward {

std::shared_ptr<ov::Model> CreateFunction(UnaryFactoryPtr unary_factory,
                                          size_t num_unary_ops,
                                          const Shape& input_shape,
                                          element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);
    ov::OutputVector outputs;

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
        auto cosh = std::make_shared<Cosh>(in_op);
        outputs.push_back(cosh);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);
    outputs.push_back(transpose0);

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

}  // namespace backward

namespace backward_mult_transposes {

std::shared_ptr<ov::Model> CreateFunction(UnaryFactoryPtr unary_factory,
                                          size_t num_unary_ops,
                                          const Shape& input_shape,
                                          element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    auto tanh0 = std::make_shared<Tanh>(transpose0);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(in_op, ng_order1);

    auto tanh1 = std::make_shared<Tanh>(transpose1);

    return std::make_shared<ov::Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(UnaryFactoryPtr unary_factory,
                                                   size_t num_unary_ops,
                                                   const Shape& input_shape,
                                                   element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto tanh0 = std::make_shared<Tanh>(in_op);
    auto tanh1 = std::make_shared<Tanh>(in_op);

    return std::make_shared<ov::Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace backward_mult_transposes

namespace forward {

std::shared_ptr<ov::Model> CreateFunction(UnaryFactoryPtr unary_factory,
                                          size_t num_unary_ops,
                                          const Shape& input_shape,
                                          element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto sinh = std::make_shared<Sinh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(sinh, ng_order0);

    auto reshape = CreateReshape(transpose0, input_shape);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op, reshape}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(UnaryFactoryPtr unary_factory,
                                                   size_t num_unary_ops,
                                                   const Shape& input_shape,
                                                   element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto sinh = std::make_shared<Sinh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(sinh, ng_order0);
    auto reshape = CreateReshape(transpose0, input_shape);

    NodePtr in_op = sinh;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(in_op, ng_order1);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose1, reshape}, ov::ParameterVector{X});
}

}  // namespace forward
}  // namespace mult_consumers_first_node

std::vector<UnaryFactoryPtr> unary_factories = {
    CREATE_UNARY_FACTORY(Clamp),      CREATE_UNARY_FACTORY(Elu),      CREATE_UNARY_FACTORY(SoftPlus),
    CREATE_UNARY_FACTORY(LogicalNot), CREATE_UNARY_FACTORY(Convert),  CREATE_UNARY_FACTORY(Abs),
    CREATE_UNARY_FACTORY(Acos),       CREATE_UNARY_FACTORY(Asin),     CREATE_UNARY_FACTORY(Asinh),
    CREATE_UNARY_FACTORY(Atan),       CREATE_UNARY_FACTORY(Ceiling),  CREATE_UNARY_FACTORY(Cos),
    CREATE_UNARY_FACTORY(Cosh),       CREATE_UNARY_FACTORY(Erf),      CREATE_UNARY_FACTORY(Exp),
    CREATE_UNARY_FACTORY(Gelu),       CREATE_UNARY_FACTORY(HSigmoid), CREATE_UNARY_FACTORY(HSwish),
    CREATE_UNARY_FACTORY(Log),        CREATE_UNARY_FACTORY(Negative), CREATE_UNARY_FACTORY(Relu),
    CREATE_UNARY_FACTORY(Sigmoid),    CREATE_UNARY_FACTORY(Sign),     CREATE_UNARY_FACTORY(Sin),
    CREATE_UNARY_FACTORY(Sinh),       CREATE_UNARY_FACTORY(SoftSign), CREATE_UNARY_FACTORY(Sqrt),
    CREATE_UNARY_FACTORY(Tan),        CREATE_UNARY_FACTORY(Tanh)};

std::vector<size_t> unary_operations_numbers = {1, 10};

}  // namespace

TEST_P(TransposeSinkingUnaryTestFixture, CompareFunctions) {
    UnaryFactoryPtr unary_factory;
    PassFactoryPtr pass_factory;
    size_t num_unary_ops;
    CreateGraphF model_factory;
    CreateGraphF reference_model_factory;
    Shape input_shape;
    element::Type input_type;
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

struct TestCase {
    std::vector<UnaryFactoryPtr> main_node;
    PassFactoryPtr transformation;
    std::vector<size_t> num_main_ops;
    CreateGraphF test_model;
    CreateGraphF ref_model;
    Shape input_shape;
    element::Type type;
};

auto wrapper = [](const TestCase& test_case) {
    return ::testing::Combine(::testing::ValuesIn(test_case.main_node),
                       ::testing::Values(test_case.transformation),
                       ::testing::ValuesIn(test_case.num_main_ops),
                       ::testing::Values(test_case.test_model),
                       ::testing::Values(test_case.ref_model),
                       ::testing::Values(test_case.input_shape),
                       ::testing::Values(test_case.type));
};

auto test_forward = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = CreateFunctionTransposeBefore;
    test_case.ref_model = CreateFunctionTransposeAfter;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_backward = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = CreateFunctionTransposeAfter;
    test_case.ref_model = CreateFunctionTransposeBefore;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_forward_multiple_consumers_reshape = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_last_node::with_reshape::CreateFunctionTransposeBefore;
    test_case.ref_model = mult_consumers_last_node::with_reshape::CreateFunctionTransposeAfter;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_backward_multiple_consumers_reshape = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_last_node::with_reshape::CreateFunctionTransposeAfter;
    test_case.ref_model = mult_consumers_last_node::with_reshape::CreateFunctionTransposeBefore;;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_forward_multiple_consumers_eltwise = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_last_node::with_eltwise::CreateFunctionTransposeBefore;
    test_case.ref_model = mult_consumers_last_node::with_eltwise::CreateFunctionTransposeAfter;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_backward_multiple_consumers_eltwise = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_last_node::with_eltwise::CreateFunctionTransposeAfter;
    test_case.ref_model = mult_consumers_last_node::with_eltwise::CreateFunctionTransposeBefore;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_backward_multiple_consumers_first_node = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_first_node::backward::CreateFunction;
    test_case.ref_model = mult_consumers_first_node::backward::CreateFunction;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_backward_multiple_transposes_first_node = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_first_node::backward_mult_transposes::CreateFunction;
    test_case.ref_model = mult_consumers_first_node::backward_mult_transposes::CreateReferenceFunction;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_forward_multiple_consumers_first_node = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_first_node::forward::CreateFunction;
    test_case.ref_model = mult_consumers_first_node::forward::CreateReferenceFunction;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryForwardTestSuite,
                         TransposeSinkingUnaryTestFixture,
                         test_forward(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryBackwardTestSuite,
                         TransposeSinkingUnaryTestFixture,
                         test_backward(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingUnaryForwardMultConsumersTestSuiteLastNodeReshape,
    TransposeSinkingUnaryTestFixture,
    test_forward_multiple_consumers_reshape(),
    TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingUnaryBackwardMultConsumersTestSuiteLastNodeReshape,
    TransposeSinkingUnaryTestFixture,
    test_backward_multiple_consumers_reshape(),
    TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingUnaryForwardMultConsumersTestSuiteLastNodeEltwise,
    TransposeSinkingUnaryTestFixture,
    test_forward_multiple_consumers_eltwise(),
    TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingUnaryForwardMultConsumersTestSuiteFirstNode,
    TransposeSinkingUnaryTestFixture,
    test_backward_multiple_consumers_eltwise(),
    TransposeSinkingUnaryTestFixture::get_test_name);


INSTANTIATE_TEST_SUITE_P(TransposeSinkingUnaryBackwardMultConsumersTestSuiteFirstNode,
                         TransposeSinkingUnaryTestFixture,
                         test_backward_multiple_consumers_first_node(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingUnaryBackwardMultTransposeConsumersTestSuiteFirstNode,
    TransposeSinkingUnaryTestFixture,
    test_backward_multiple_transposes_first_node(),
    TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
        TransposeSinkingUnaryForwardMultTransposeConsumersTestSuiteFirstNode,
        TransposeSinkingUnaryTestFixture,
        test_forward_multiple_consumers_first_node(),
        TransposeSinkingUnaryTestFixture::get_test_name);