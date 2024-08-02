// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_unary.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing::utils;

namespace transpose_sinking {
namespace testing {
namespace unary {

using CreateGraphF = std::function<std::shared_ptr<
    ov::Model>(FactoryPtr unary_factory, size_t num_unary_ops, const Shape& input_shape, element::Type input_type)>;

using TestParams = std::tuple<FactoryPtr,
                              PassFactoryPtr,
                              size_t,         /* num_unary_ops */
                              CreateGraphF,   /* model_factory */
                              CreateGraphF,   /* reference_model_factory */
                              Shape,          /* input shape */
                              element::Type>; /* input type */

class TransposeSinkingUnaryTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestParams>& obj) {
        FactoryPtr unary_factory;
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

template <typename UnaryT>
class UnaryFactory : public IFactory {
public:
    explicit UnaryFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& inputs) const override {
        return std::make_shared<UnaryT>(inputs[0]);
    }
};

template <>
NodePtr UnaryFactory<Elu>::create(const OutputVector& inputs) const {
    return std::make_shared<Elu>(inputs[0], 0.1);
}

template <>
NodePtr UnaryFactory<Clamp>::create(const OutputVector& inputs) const {
    return std::make_shared<Clamp>(inputs[0], 0.1, 0.2);
}

template <>
NodePtr UnaryFactory<Convert>::create(const OutputVector& inputs) const {
    return std::make_shared<Convert>(inputs[0], element::f64);
}

template <>
NodePtr UnaryFactory<Selu>::create(const OutputVector& inputs) const {
    auto alpha = std::make_shared<Constant>(element::f32, Shape{}, 2.0);
    auto lambda = std::make_shared<Constant>(element::f32, Shape{}, 3.0);
    return std::make_shared<Selu>(inputs[0], alpha, lambda);
}

template <>
NodePtr UnaryFactory<Swish>::create(const OutputVector& inputs) const {
    auto beta = std::make_shared<Constant>(element::f32, Shape{}, 0.9);
    return std::make_shared<Swish>(inputs[0], beta);
}

template <>
NodePtr UnaryFactory<HardSigmoid>::create(const OutputVector& inputs) const {
    auto alpha = std::make_shared<Constant>(element::f32, Shape{}, 2.0);
    auto beta = std::make_shared<Constant>(element::f32, Shape{}, 3.0);
    return std::make_shared<HardSigmoid>(inputs[0], alpha, beta);
}

template <>
NodePtr UnaryFactory<LogSoftmax>::create(const OutputVector& inputs) const {
    return std::make_shared<LogSoftmax>(inputs[0], 2);
}

template <>
NodePtr UnaryFactory<ConvertLike>::create(const OutputVector& inputs) const {
    auto like = std::make_shared<Constant>(element::f64, Shape{1, 2, 3, 2, 1, 1}, 1);
    return std::make_shared<ConvertLike>(inputs[0], like);
}

template <typename UnaryT>
FactoryPtr CreateUnaryFactory(const std::string& type_name) {
    return std::make_shared<UnaryFactory<UnaryT>>(type_name);
}

#undef CREATE_UNARY_FACTORY
#define CREATE_UNARY_FACTORY(type_name) CreateUnaryFactory<type_name>(#type_name)

// ----------------------------------------------------------------------------

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(const FactoryPtr& unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    return std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(const FactoryPtr& unary_factory,
                                                        size_t num_unary_ops,
                                                        const Shape& input_shape,
                                                        element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
}

// We consider HardSigmoid, Swish, Selu, ConvertLike as unary ops
// and handle only 0th input of these ops.
// Transpose on 2nd input should be ignored.
namespace ignore_transpose_on_second_input {
std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(const FactoryPtr& unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});

        // Connect Transpose to 2nd input of the main node
        std::vector<int> order(in_op->input(1).get_shape().size());
        std::iota(order.rbegin(), order.rend(), 0);
        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{order.size()}, order);
        auto transpose0 = std::make_shared<Transpose>(in_op->input_value(1), ng_order0);
        in_op->input(1).replace_source_output(transpose0);
    }

    return std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
}
}  // namespace ignore_transpose_on_second_input

NodePtr CreateReshape(const NodePtr& parent_node, const Shape& input_shape) {
    const size_t mul = std::accumulate(input_shape.begin(), input_shape.end(), (size_t)1, std::multiplies<size_t>());
    auto reshape_const = std::make_shared<Constant>(element::u64, Shape{1}, Shape{mul});
    return std::make_shared<Reshape>(parent_node, reshape_const, false);
}

namespace mult_consumers_last_node {
namespace with_reshape {

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(const FactoryPtr& unary_factory,
                                                        size_t num_unary_ops,
                                                        const Shape& input_shape,
                                                        element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    auto reshape1 = CreateReshape(transpose0, input_shape);
    auto reshape2 = CreateReshape(transpose0, input_shape);

    return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(const FactoryPtr& unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto reshape1 = CreateReshape(in_op, input_shape);
    auto reshape2 = CreateReshape(in_op, input_shape);

    return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2}, ov::ParameterVector{X});
}
}  // namespace with_reshape

namespace with_eltwise {

std::shared_ptr<ov::Model> CreateFunctionTransposeAfter(FactoryPtr unary_factory,
                                                        size_t num_unary_ops,
                                                        const Shape& input_shape,
                                                        element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto sinh = std::make_shared<Sinh>(in_op);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(sinh, ng_order0);

    auto cosh = std::make_shared<Cosh>(in_op);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(cosh, ng_order1);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0, transpose1}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(const FactoryPtr& unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto sinh = std::make_shared<Sinh>(in_op);
    auto cosh = std::make_shared<Cosh>(in_op);

    return std::make_shared<ov::Model>(ov::OutputVector{sinh, cosh}, ov::ParameterVector{X});
}

}  // namespace with_eltwise
}  // namespace mult_consumers_last_node

namespace mult_consumers_first_node {
namespace backward {

std::shared_ptr<ov::Model> CreateFunction(const FactoryPtr& unary_factory,
                                          size_t num_unary_ops,
                                          const Shape& input_shape,
                                          element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);
    ov::OutputVector outputs;

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
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

std::shared_ptr<ov::Model> CreateFunction(const FactoryPtr& unary_factory,
                                          size_t num_unary_ops,
                                          const Shape& input_shape,
                                          element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    auto tanh0 = std::make_shared<Tanh>(transpose0);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(in_op, ng_order1);

    auto tanh1 = std::make_shared<Tanh>(transpose1);

    return std::make_shared<ov::Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(const FactoryPtr& unary_factory,
                                                   size_t num_unary_ops,
                                                   const Shape& input_shape,
                                                   element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create({in_op});
    }

    auto tanh0 = std::make_shared<Tanh>(in_op);
    auto tanh1 = std::make_shared<Tanh>(in_op);

    return std::make_shared<ov::Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace backward_mult_transposes

namespace forward {

std::shared_ptr<ov::Model> CreateFunction(const FactoryPtr& unary_factory,
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
        in_op = unary_factory->create({in_op});
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op, reshape}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(const FactoryPtr& unary_factory,
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
        in_op = unary_factory->create({in_op});
    }

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(in_op, ng_order1);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose1, reshape}, ov::ParameterVector{X});
}

}  // namespace forward
}  // namespace mult_consumers_first_node

std::vector<FactoryPtr> unary_factories = {
    CREATE_UNARY_FACTORY(Clamp),      CREATE_UNARY_FACTORY(Elu),         CREATE_UNARY_FACTORY(SoftPlus),
    CREATE_UNARY_FACTORY(LogicalNot), CREATE_UNARY_FACTORY(Convert),     CREATE_UNARY_FACTORY(Abs),
    CREATE_UNARY_FACTORY(Acos),       CREATE_UNARY_FACTORY(Asin),        CREATE_UNARY_FACTORY(Asinh),
    CREATE_UNARY_FACTORY(Atan),       CREATE_UNARY_FACTORY(Ceiling),     CREATE_UNARY_FACTORY(Cos),
    CREATE_UNARY_FACTORY(Cosh),       CREATE_UNARY_FACTORY(Erf),         CREATE_UNARY_FACTORY(Exp),
    CREATE_UNARY_FACTORY(Gelu),       CREATE_UNARY_FACTORY(HSigmoid),    CREATE_UNARY_FACTORY(HSwish),
    CREATE_UNARY_FACTORY(Log),        CREATE_UNARY_FACTORY(Negative),    CREATE_UNARY_FACTORY(Relu),
    CREATE_UNARY_FACTORY(Sigmoid),    CREATE_UNARY_FACTORY(Sign),        CREATE_UNARY_FACTORY(Sin),
    CREATE_UNARY_FACTORY(Sinh),       CREATE_UNARY_FACTORY(SoftSign),    CREATE_UNARY_FACTORY(Sqrt),
    CREATE_UNARY_FACTORY(Tan),        CREATE_UNARY_FACTORY(Tanh),        CREATE_UNARY_FACTORY(Selu),
    CREATE_UNARY_FACTORY(Swish),      CREATE_UNARY_FACTORY(HardSigmoid), CREATE_UNARY_FACTORY(LogSoftmax),
    CREATE_UNARY_FACTORY(ConvertLike)};

TEST_P(TransposeSinkingUnaryTestFixture, CompareFunctions) {
    FactoryPtr unary_factory;
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
    std::vector<FactoryPtr> main_node;
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = CreateFunctionTransposeBefore;
    test_case.ref_model = CreateFunctionTransposeAfter;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_forward_unary_with_multiple_inputs = []() {
    TestCase test_case;
    test_case.main_node = std::vector<FactoryPtr>{CREATE_UNARY_FACTORY(HardSigmoid),
                                                  CREATE_UNARY_FACTORY(Selu),
                                                  CREATE_UNARY_FACTORY(ConvertLike),
                                                  CREATE_UNARY_FACTORY(Swish)};
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = ignore_transpose_on_second_input::CreateFunctionTransposeBefore;
    test_case.ref_model = ignore_transpose_on_second_input::CreateFunctionTransposeBefore;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_backward = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryBackward);
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryForward);
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_last_node::with_reshape::CreateFunctionTransposeAfter;
    test_case.ref_model = mult_consumers_last_node::with_reshape::CreateFunctionTransposeBefore;
    ;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

auto test_forward_multiple_consumers_eltwise = []() {
    TestCase test_case;
    test_case.main_node = unary_factories;
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryForward);
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryBackward);
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryBackward);
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryBackward);
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
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.test_model = mult_consumers_first_node::forward::CreateFunction;
    test_case.ref_model = mult_consumers_first_node::forward::CreateReferenceFunction;
    test_case.input_shape = {1, 96, 55, 55};
    test_case.type = element::f32;
    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TSUnaryForwardTestSuite,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_forward(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryForwardMultipleInputsTestSuite,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_forward_unary_with_multiple_inputs(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryBackwardTestSuite,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_backward(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryForwardMultConsumersTestSuiteLastNodeReshape,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_forward_multiple_consumers_reshape(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryBackwardMultConsumersTestSuiteLastNodeReshape,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_backward_multiple_consumers_reshape(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryForwardMultConsumersTestSuiteLastNodeEltwise,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_forward_multiple_consumers_eltwise(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryBackwardMultConsumersTestSuiteEltwise,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_backward_multiple_consumers_eltwise(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryBackwardMultConsumersTestSuiteFirstNode,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_backward_multiple_consumers_first_node(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryBackwardMultTransposeConsumersTestSuiteFirstNode,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_backward_multiple_transposes_first_node(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSUnaryForwardMultTransposeConsumersTestSuiteFirstNode,
                         TransposeSinkingUnaryTestFixture,
                         transpose_sinking::testing::unary::test_forward_multiple_consumers_first_node(),
                         TransposeSinkingUnaryTestFixture::get_test_name);

TEST_F(TransformationTestsF, TSUnaryForwardDynamic) {
    {
        auto X = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
        auto ts_order = std::make_shared<Constant>(element::u64, Shape{0}, Shape{});
        auto transpose = std::make_shared<Transpose>(X, ts_order);

        auto tanh = std::make_shared<Tanh>(transpose);

        model = std::make_shared<Model>(ov::OutputVector{tanh}, ov::ParameterVector{X});

        manager.register_pass<TSUnaryForward>();
    }
    {
        auto X = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

        auto tanh = std::make_shared<Tanh>(X);

        auto ts_order = std::make_shared<Constant>(element::u64, Shape{0}, Shape{});
        auto transpose = std::make_shared<Transpose>(tanh, ts_order);

        model_ref = std::make_shared<Model>(ov::OutputVector{transpose}, ov::ParameterVector{X});
    }
}

}  // namespace unary
}  // namespace testing
}  // namespace transpose_sinking
