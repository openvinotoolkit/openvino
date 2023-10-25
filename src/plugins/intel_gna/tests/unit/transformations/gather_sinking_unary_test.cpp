// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_unary.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

using NodePtr = std::shared_ptr<ov::Node>;

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

std::vector<size_t> GenerateVector(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);
    return vec;
}

std::vector<size_t> MakeGatherIndexes(size_t size) {
    std::vector<size_t> indexes = GenerateVector(size, 0);
    std::next_permutation(indexes.begin(), indexes.end());
    return indexes;
}

std::shared_ptr<Gather> MakeGather(NodePtr input_node) {
    const ov::Shape& input_shape = input_node->get_output_shape(0);
    const size_t input_shape_product =
        std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    const std::vector<size_t> indexes = MakeGatherIndexes(input_shape_product);
    auto gather_indexes_node = Constant::create(element::i64, ov::Shape{indexes.size()}, indexes);

    const size_t axis = 1;
    auto gather_axis_node = Constant::create(element::i64, Shape{}, {axis});

    return std::make_shared<Gather>(input_node, gather_indexes_node, gather_axis_node);
}

}  // namespace

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

#define CREATE_PASS_FACTORY(pass_name) std::make_shared<PassFactory<ov::intel_gna::pass::pass_name>>(#pass_name)

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

class GatherSinkingUnaryTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
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

std::string GetFinalNodeName(std::shared_ptr<ov::Model> model, int index = 0) {
    NodePtr result_node = model->get_results()[index];
    return result_node->get_input_node_ptr(0)->get_friendly_name();
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather = MakeGather(X);

    NodePtr in_op = gather;
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

    auto gather = MakeGather(in_op);

    return std::make_shared<ov::Model>(gather, ov::ParameterVector{X});
}

static NodePtr CreateReshape(NodePtr parent_node) {
    const Shape& input_shape = parent_node->get_output_shape(0);
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

    auto gather = MakeGather(in_op);

    auto reshape1 = CreateReshape(gather);
    auto reshape2 = CreateReshape(gather);

    return std::make_shared<ov::Model>(ov::OutputVector{reshape1, reshape2}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather = MakeGather(X);

    NodePtr in_op = gather;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto reshape1 = CreateReshape(in_op);
    auto reshape2 = CreateReshape(in_op);

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

    auto gather0 = MakeGather(sinh);

    auto cosh = std::make_shared<Cosh>(in_op);

    auto gather1 = MakeGather(cosh);

    return std::make_shared<ov::Model>(ov::OutputVector{gather0, gather1}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateFunctionTransposeBefore(UnaryFactoryPtr unary_factory,
                                                         size_t num_unary_ops,
                                                         const Shape& input_shape,
                                                         element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather0 = MakeGather(X);

    NodePtr in_op = gather0;
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

    auto gather0 = MakeGather(in_op);

    outputs.push_back(gather0);

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

    auto gather0 = MakeGather(in_op);

    auto tanh0 = std::make_shared<Tanh>(gather0);

    auto gather1 = MakeGather(in_op);

    auto tanh1 = std::make_shared<Tanh>(gather1);

    return std::make_shared<ov::Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(UnaryFactoryPtr unary_factory,
                                                   size_t num_unary_ops,
                                                   const Shape& input_shape,
                                                   element::Type input_type) {
    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather0 = MakeGather(X);

    NodePtr in_op = gather0;
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

    auto gather0 = MakeGather(sinh);

    auto reshape = CreateReshape(gather0);

    NodePtr in_op = gather0;
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

    auto gather0 = MakeGather(sinh);

    auto reshape = CreateReshape(gather0);

    NodePtr in_op = sinh;
    for (size_t i = 0; i < num_unary_ops; ++i) {
        in_op = unary_factory->create(in_op);
    }

    auto gather1 = MakeGather(in_op);

    return std::make_shared<ov::Model>(ov::OutputVector{gather1, reshape}, ov::ParameterVector{X});
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

TEST_P(GatherSinkingUnaryTestFixture, CompareFunctions) {
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

INSTANTIATE_TEST_SUITE_P(GatherSinkingUnaryForwardTestSuite,
                         GatherSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryForward)),
                                            ::testing::ValuesIn(unary_operations_numbers),
                                            ::testing::Values(CreateFunctionTransposeBefore),
                                            ::testing::Values(CreateFunctionTransposeAfter),
                                            ::testing::Values(Shape{1, 96, 55, 55}),
                                            ::testing::Values(element::f32)),
                         GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(GatherSinkingUnaryBackwardTestSuite,
                         GatherSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryBackward)),
                                            ::testing::ValuesIn(unary_operations_numbers),
                                            ::testing::Values(CreateFunctionTransposeAfter),
                                            ::testing::Values(CreateFunctionTransposeBefore),
                                            ::testing::Values(Shape{1, 96, 55, 55}),
                                            ::testing::Values(element::f32)),
                         GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingUnaryForwardMultConsumersTestSuiteLastNodeReshape,
    GatherSinkingUnaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(unary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryForward)),
                       ::testing::ValuesIn(unary_operations_numbers),
                       ::testing::Values(mult_consumers_last_node::with_reshape::CreateFunctionTransposeBefore),
                       ::testing::Values(mult_consumers_last_node::with_reshape::CreateFunctionTransposeAfter),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::Values(element::f32)),
    GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingUnaryBackwardMultConsumersTestSuiteLastNodeReshape,
    GatherSinkingUnaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(unary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryBackward)),
                       ::testing::ValuesIn(unary_operations_numbers),
                       ::testing::Values(mult_consumers_last_node::with_reshape::CreateFunctionTransposeAfter),
                       ::testing::Values(mult_consumers_last_node::with_reshape::CreateFunctionTransposeBefore),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::Values(element::f32)),
    GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingUnaryForwardMultConsumersTestSuiteLastNodeEltwise,
    GatherSinkingUnaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(unary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryForward)),
                       ::testing::ValuesIn(unary_operations_numbers),
                       ::testing::Values(mult_consumers_last_node::with_eltwise::CreateFunctionTransposeBefore),
                       ::testing::Values(mult_consumers_last_node::with_eltwise::CreateFunctionTransposeAfter),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::Values(element::f32)),
    GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingUnaryForwardMultConsumersTestSuiteFirstNode,
    GatherSinkingUnaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(unary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryForward)),
                       ::testing::ValuesIn(unary_operations_numbers),
                       ::testing::Values(mult_consumers_first_node::forward::CreateFunction),
                       ::testing::Values(mult_consumers_first_node::forward::CreateReferenceFunction),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::Values(element::f32)),
    GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(GatherSinkingUnaryBackwardMultConsumersTestSuiteFirstNode,
                         GatherSinkingUnaryTestFixture,
                         ::testing::Combine(::testing::ValuesIn(unary_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryBackward)),
                                            ::testing::ValuesIn(unary_operations_numbers),
                                            ::testing::Values(mult_consumers_first_node::backward::CreateFunction),
                                            ::testing::Values(mult_consumers_first_node::backward::CreateFunction),
                                            ::testing::Values(Shape{1, 96, 55, 55}),
                                            ::testing::Values(element::f32)),
                         GatherSinkingUnaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingUnaryBackwardMultTransposeConsumersTestSuiteFirstNode,
    GatherSinkingUnaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(unary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingUnaryBackward)),
                       ::testing::ValuesIn(unary_operations_numbers),
                       ::testing::Values(mult_consumers_first_node::backward_mult_transposes::CreateFunction),
                       ::testing::Values(mult_consumers_first_node::backward_mult_transposes::CreateReferenceFunction),
                       ::testing::Values(Shape{1, 96, 55, 55}),
                       ::testing::Values(element::f32)),
    GatherSinkingUnaryTestFixture::get_test_name);
