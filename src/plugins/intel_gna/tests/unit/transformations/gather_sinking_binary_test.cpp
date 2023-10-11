// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_binary.hpp"

#include <functional>

#include "common_test_utils/ov_test_utils.hpp"
#include "gather_sinking_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace ov::opset12;

namespace gather_sinking_binary_eltwise {

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<Model>;
using Output = ov::Output<ov::Node>;

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

// ----------------------------------------------------------------------------

class IBinaryFactory {
public:
    IBinaryFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IBinaryFactory() = default;
    virtual NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const = 0;
    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

using BinaryFactoryPtr = std::shared_ptr<IBinaryFactory>;

template <typename BinaryT>
class BinaryFactory : public IBinaryFactory {
public:
    BinaryFactory(const std::string& type_name) : IBinaryFactory(type_name) {}
    NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const override {
        return std::make_shared<BinaryT>(parent_left_node, parent_right_node);
    }
};

template <typename BinaryT>
BinaryFactoryPtr create_binary_factory(const std::string& type_name) {
    return std::make_shared<BinaryFactory<BinaryT>>(type_name);
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

#undef CREATE_BINARY_FACTORY
#define CREATE_BINARY_FACTORY(type_name) create_binary_factory<type_name>(#type_name)

std::vector<BinaryFactoryPtr> binary_elementwise_factories = {CREATE_BINARY_FACTORY(Add),
                                                              CREATE_BINARY_FACTORY(Divide),
                                                              CREATE_BINARY_FACTORY(Maximum),
                                                              CREATE_BINARY_FACTORY(Minimum),
                                                              CREATE_BINARY_FACTORY(Mod),
                                                              CREATE_BINARY_FACTORY(Multiply),
                                                              CREATE_BINARY_FACTORY(Power),
                                                              CREATE_BINARY_FACTORY(SquaredDifference),
                                                              CREATE_BINARY_FACTORY(Subtract)};

std::vector<size_t> binary_operations_numbers = {1, 10};

std::vector<size_t> binary_transpose_input_indexes = {0, 1};

}  // namespace

namespace single_consumer {
namespace forward {
namespace one_input_transpose {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      size_t num_binary_ops,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};
    const Shape const_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);
    auto gather = make_gather(X, gather_forward, /* axis */ 1);

    NodePtr in_op = gather;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});
        if (!binary_gather_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    return std::make_shared<Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               size_t num_binary_ops,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        auto gather_reversed = make_gather(in_constant, gather_backward, /* axis */ 1);

        if (!binary_gather_input_idx)
            in_op = binary_factory->create(in_op, gather_reversed);
        else
            in_op = binary_factory->create(gather_reversed, in_op);
    }

    auto gather = make_gather(in_op, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather}, ov::ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      size_t num_binary_ops,
                                      element::Type input_type) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);
    auto gather = make_gather(X, gather_forward, /* axis */ 1);

    NodePtr in_op = gather;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        auto gather = make_gather(in_constant, gather_forward, /* axis */ 1);

        in_op = binary_factory->create(in_op, gather);
    }

    return std::make_shared<Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               size_t num_binary_ops,
                                               element::Type input_type) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        auto gather = make_gather(in_constant, gather_forward, /* axis */ 1);
        auto gather_reversed = make_gather(gather, gather_backward, /* axis */ 1);

        in_op = binary_factory->create(in_op, gather_reversed);
    }

    auto gather = make_gather(in_op, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather}, ov::ParameterVector{X});
}

using CreateGraphBinaryTwoTransposeInputsF = std::function<
    std::shared_ptr<Model>(BinaryFactoryPtr binary_factory, size_t num_binary_ops, element::Type input_type)>;

using TestBinaryTwoTransposeInputsParams =
    std::tuple<BinaryFactoryPtr,
               PassFactoryPtr,
               size_t,                               /* num_binary_ops */
               CreateGraphBinaryTwoTransposeInputsF, /* model_factory */
               CreateGraphBinaryTwoTransposeInputsF, /* reference_model_factory */
               element::Type>;                       /* input type */

class GatherSinkingBinaryTwoTransposeInputsTestFixture
    : public ::testing::WithParamInterface<TestBinaryTwoTransposeInputsParams>,
      public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryTwoTransposeInputsParams>& obj) {
        BinaryFactoryPtr binary_factory;
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

TEST_P(GatherSinkingBinaryTwoTransposeInputsTestFixture, CompareFunctions) {
    BinaryFactoryPtr binary_factory;
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

INSTANTIATE_TEST_SUITE_P(GatherSinkingBinaryTwoTransposeInputsForwardTestSuite,
                         GatherSinkingBinaryTwoTransposeInputsTestFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryForward)),
                                            ::testing::ValuesIn(binary_operations_numbers),
                                            ::testing::Values(CreateFunction),
                                            ::testing::Values(CreateReferenceFunction),
                                            ::testing::Values(element::f32)),
                         GatherSinkingBinaryTwoTransposeInputsTestFixture::get_test_name);

}  // namespace double_transpose
}  // namespace forward

namespace backward {
namespace one_input_transpose {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      size_t num_binary_ops,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        if (!binary_gather_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }
    auto gather = make_gather(in_op, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               size_t num_binary_ops,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh = std::make_shared<Tanh>(X);
    auto gather = make_gather(X, gather_forward, /* axis */ 1);

    NodePtr in_op = gather;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        auto gather = make_gather(in_constant, gather_forward, /* axis */ 1);
        if (!binary_gather_input_idx)
            in_op = binary_factory->create(in_op, gather);
        else
            in_op = binary_factory->create(gather, in_op);
    }

    return std::make_shared<Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

using CreateGraphBinaryF = std::function<std::shared_ptr<Model>(BinaryFactoryPtr binary_factory,
                                                                size_t num_binary_ops,
                                                                element::Type input_type,
                                                                size_t binary_gather_input_idx)>;

using TestBinaryParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    size_t,             /* num_binary_ops */
                                    CreateGraphBinaryF, /* model_factory */
                                    CreateGraphBinaryF, /* reference_model_factory */
                                    element::Type,      /* input type */
                                    size_t>;            /* binary_gather_input_idx */

class GatherSinkingBinaryTestFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                       public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryParams>& obj) {
        BinaryFactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        size_t num_binary_ops;
        CreateGraphBinaryF model_factory;
        CreateGraphBinaryF reference_model_factory;
        element::Type input_type;
        size_t binary_gather_input_idx;

        std::tie(binary_factory,
                 pass_factory,
                 num_binary_ops,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 binary_gather_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "numBinaryOps=" << num_binary_ops << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_gather_input_idx;

        return test_name.str();
    }
};

TEST_P(GatherSinkingBinaryTestFixture, CompareFunctions) {
    BinaryFactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryF model_factory;
    CreateGraphBinaryF reference_model_factory;
    element::Type input_type;
    size_t binary_gather_input_idx;
    std::tie(binary_factory,
             pass_factory,
             num_binary_ops,
             model_factory,
             reference_model_factory,
             input_type,
             binary_gather_input_idx) = this->GetParam();

    model = model_factory(binary_factory, num_binary_ops, input_type, binary_gather_input_idx);
    model_ref = reference_model_factory(binary_factory, num_binary_ops, input_type, binary_gather_input_idx);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryForwardTestSuite,
    GatherSinkingBinaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryForward)),
                       ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryBackwardTestSuite,
    GatherSinkingBinaryTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryBackward)),
                       ::testing::ValuesIn(binary_operations_numbers),
                       ::testing::Values(single_consumer::backward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::backward::one_input_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryTestFixture::get_test_name);

// --------------------------------------------------------------------------------------

using CreateGraphBinaryIncompatShapesF = std::function<
    std::shared_ptr<Model>(BinaryFactoryPtr unary_factory, element::Type input_type, size_t binary_gather_input_idx)>;

using TestBinaryIncompatShapesParams = std::tuple<BinaryFactoryPtr,
                                                  PassFactoryPtr,
                                                  CreateGraphBinaryIncompatShapesF, /* model_factory */
                                                  CreateGraphBinaryIncompatShapesF, /* reference_model_factory */
                                                  element::Type,                    /* input type */
                                                  size_t>;                          /* binary_gather_input_idx */

class GatherSinkingBinaryIncompatShapesTestFixture
    : public ::testing::WithParamInterface<TestBinaryIncompatShapesParams>,
      public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryIncompatShapesParams>& obj) {
        BinaryFactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        CreateGraphBinaryIncompatShapesF model_factory;
        CreateGraphBinaryIncompatShapesF reference_model_factory;
        element::Type input_type;
        size_t binary_gather_input_idx;
        std::tie(binary_factory,
                 pass_factory,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 binary_gather_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_gather_input_idx;

        return test_name.str();
    }
};

namespace binary {
namespace single_consumer {
namespace backward {
namespace incompat_shapes {
namespace insert_gather {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(X, in_constant);
    else
        binary_op = binary_factory->create(in_constant, X);

    auto gather = make_gather(binary_op, gather_forward, /* axis */ 2);
    return std::make_shared<Model>(ov::OutputVector{gather}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});
    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});
    auto gather1 = make_gather(in_constant, gather_forward, /* axis */ 2);

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(gather0, gather1);
    else
        binary_op = binary_factory->create(gather1, gather0);

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}
}  // namespace insert_gather

namespace no_insert_gather {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(X, in_constant);
    else
        binary_op = binary_factory->create(in_constant, X);

    auto gather = make_gather(binary_op, gather_forward, /* axis */ 0);
    return std::make_shared<Model>(ov::OutputVector{gather}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});
    auto gather1 = make_gather(in_constant, gather_forward, /* axis */ 0);

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(X, gather1);
    else
        binary_op = binary_factory->create(gather1, X);

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}
}  // namespace no_insert_gather

}  // namespace incompat_shapes
}  // namespace backward

namespace forward {
namespace incompat_shapes {
namespace gather_small_input {
namespace insert_gather {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});
    auto gather0 = make_gather(in_constant, gather_forward, /* axis */ 2);

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(gather0, X);
    else
        binary_op = binary_factory->create(X, gather0);

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});
    auto gather0 = make_gather(X, gather_backward, /* axis */ 1);

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(in_constant, gather0);
    else
        binary_op = binary_factory->create(gather0, in_constant);

    auto gather1 = make_gather(binary_op, gather_forward, /* axis */ 2);

    return std::make_shared<Model>(ov::OutputVector{gather1}, ov::ParameterVector{X});
}
}  // namespace insert_gather

namespace no_insert_gather {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});
    auto gather0 = make_gather(in_constant, gather_forward, /* axis */ 0);

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(gather0, X);
    else
        binary_op = binary_factory->create(X, gather0);

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{1, 20, 1});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{5, 3, 20, 7}, Shape{1});

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(in_constant, X);
    else
        binary_op = binary_factory->create(X, in_constant);

    auto gather1 = make_gather(binary_op, gather_forward, /* axis */ 0);

    return std::make_shared<Model>(ov::OutputVector{gather1}, ov::ParameterVector{X});
}
}  // namespace no_insert_gather

}  // namespace gather_small_input

namespace gather_large_input {
namespace insert_gather {
std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{5, 3, 20, 7});

    auto in_constant = std::make_shared<Constant>(input_type, Shape{1, 20, 1}, Shape{1});
    auto gather0 = make_gather(in_constant, gather_forward, /* axis */ 1);

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(gather0, X);
    else
        binary_op = binary_factory->create(X, gather0);

    return std::make_shared<Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    auto X = std::make_shared<Parameter>(input_type, Shape{5, 3, 20, 7});
    auto gather0 = make_gather(X, gather_backward, /* axis */ 2);

    auto in_constant = std::make_shared<Constant>(input_type, Shape{1, 20, 1}, Shape{1});

    NodePtr binary_op;
    if (!binary_gather_input_idx)
        binary_op = binary_factory->create(in_constant, gather0);
    else
        binary_op = binary_factory->create(gather0, in_constant);

    auto gather1 = make_gather(binary_op, gather_forward, /* axis */ 2);

    return std::make_shared<Model>(ov::OutputVector{gather1}, ov::ParameterVector{X});
}
}  // namespace insert_gather

}  // namespace gather_large_input

}  // namespace incompat_shapes
}  // namespace forward

}  // namespace single_consumer
}  // namespace binary

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryIncompatShapesBackwardInsertGatherTestSuite,
    GatherSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_elementwise_factories),
        ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryBackward)),
        ::testing::Values(binary::single_consumer::backward::incompat_shapes::insert_gather::CreateFunction),
        ::testing::Values(binary::single_consumer::backward::incompat_shapes::insert_gather::CreateReferenceFunction),
        ::testing::Values(element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryIncompatShapesBackwardNoGatherInsertTestSuite,
    GatherSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_elementwise_factories),
        ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryBackward)),
        ::testing::Values(binary::single_consumer::backward::incompat_shapes::no_insert_gather::CreateFunction),
        ::testing::Values(
            binary::single_consumer::backward::incompat_shapes::no_insert_gather::CreateReferenceFunction),
        ::testing::Values(element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryIncompatShapesGatherSmallInputForwardInsertGatherTestSuite,
    GatherSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_elementwise_factories),
        ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryForward)),
        ::testing::Values(
            binary::single_consumer::forward::incompat_shapes::gather_small_input::insert_gather::CreateFunction),
        ::testing::Values(binary::single_consumer::forward::incompat_shapes::gather_small_input::insert_gather::
                              CreateReferenceFunction),
        ::testing::Values(element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryIncompatShapesGatherSmallInputForwardNoGatherInsertTestSuite,
    GatherSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_elementwise_factories),
        ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryForward)),
        ::testing::Values(
            binary::single_consumer::forward::incompat_shapes::gather_small_input::no_insert_gather::CreateFunction),
        ::testing::Values(binary::single_consumer::forward::incompat_shapes::gather_small_input::no_insert_gather::
                              CreateReferenceFunction),
        ::testing::Values(element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    GatherSinkingBinaryIncompatShapesGatherLargeInputInsertgather_forwardTestSuite,
    GatherSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_elementwise_factories),
        ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryForward)),
        ::testing::Values(
            binary::single_consumer::forward::incompat_shapes::gather_large_input::insert_gather::CreateFunction),
        ::testing::Values(binary::single_consumer::forward::incompat_shapes::gather_large_input::insert_gather::
                              CreateReferenceFunction),
        ::testing::Values(element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    GatherSinkingBinaryIncompatShapesTestFixture::get_test_name);

}  // namespace one_input_transpose
}  // namespace backward
}  // namespace single_consumer

namespace mult_consumers {
namespace forward {
namespace input_transpose_consumers {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);
    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    auto tanh = std::make_shared<Tanh>(gather0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(gather0, in_constant);
    else
        binary = binary_factory->create(in_constant, gather0);

    return std::make_shared<Model>(ov::OutputVector{binary, tanh}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);
    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    auto tanh = std::make_shared<Tanh>(gather0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    auto gather_reversed = make_gather(in_constant, gather_backward, /* axis */ 1);

    if (!binary_gather_input_idx)
        binary = binary_factory->create(X, gather_reversed);
    else
        binary = binary_factory->create(gather_reversed, X);

    auto gather1 = make_gather(binary, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather1, tanh}, ov::ParameterVector{X});
}

}  // namespace input_transpose_consumers

namespace output_consumers {

namespace one_binary {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(gather0, in_constant);
    else
        binary = binary_factory->create(in_constant, gather0);

    auto tanh1 = std::make_shared<Tanh>(binary);
    auto tanh2 = std::make_shared<Tanh>(binary);

    return std::make_shared<Model>(ov::OutputVector{tanh1, tanh2}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto gather_reversed = make_gather(in_constant, gather_backward, /* axis */ 1);

    if (!binary_gather_input_idx)
        binary = binary_factory->create(X, gather_reversed);
    else
        binary = binary_factory->create(gather_reversed, X);

    auto gather0 = make_gather(binary, gather_forward, /* axis */ 1);

    auto tanh1 = std::make_shared<Tanh>(gather0);
    auto tanh2 = std::make_shared<Tanh>(gather0);

    return std::make_shared<Model>(ov::OutputVector{tanh1, tanh2}, ov::ParameterVector{X});
}

}  // namespace one_binary

}  // namespace output_consumers

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(gather0, in_constant);
    else
        binary = binary_factory->create(in_constant, gather0);

    auto tanh = std::make_shared<Tanh>(X);

    return std::make_shared<Model>(ov::OutputVector{binary, tanh}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh = std::make_shared<Tanh>(X);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto gather_reversed = make_gather(in_constant, gather_backward, /* axis */ 1);

    if (!binary_gather_input_idx)
        binary = binary_factory->create(X, gather_reversed);
    else
        binary = binary_factory->create(gather_reversed, X);

    auto gather1 = make_gather(binary, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather1, tanh}, ov::ParameterVector{X});
}

}  // namespace input_node_consumers

}  // namespace forward

namespace backward {

namespace output_consumers {

namespace one_binary {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(tanh0, in_constant);
    else
        binary = binary_factory->create(in_constant, tanh0);

    auto tanh = std::make_shared<Tanh>(binary);

    auto gather0 = make_gather(binary, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather0, tanh}, ov::ParameterVector{X});
}

}  // namespace one_binary

namespace multiple_binaries {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};
    const size_t n_binaries = 10;

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    NodePtr in_op = tanh0;
    for (size_t i = 0; i < n_binaries; ++i) {
        auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
        if (!binary_gather_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    auto tanh = std::make_shared<Tanh>(in_op);

    auto gather0 = make_gather(in_op, gather_forward, /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{gather0, tanh}, ov::ParameterVector{X});
}

}  // namespace multiple_binaries

}  // namespace output_consumers

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(tanh0, in_constant);
    else
        binary = binary_factory->create(in_constant, tanh0);

    auto gather0 = make_gather(binary, gather_forward, /* axis */ 1);

    auto tanh1 = std::make_shared<Tanh>(tanh0);

    return std::make_shared<Model>(ov::OutputVector{gather0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    auto gather0 = make_gather(tanh0, gather_forward, /* axis */ 1);

    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto gather = make_gather(in_constant, gather_forward, /* axis */ 1);

    NodePtr binary;
    if (!binary_gather_input_idx)
        binary = binary_factory->create(gather0, gather);
    else
        binary = binary_factory->create(gather, gather0);

    auto tanh1 = std::make_shared<Tanh>(tanh0);

    return std::make_shared<Model>(ov::OutputVector{binary, tanh1}, ov::ParameterVector{X});
}

}  // namespace input_node_consumers

namespace output_transpose_mult_consumers {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(X, in_constant);
    else
        binary = binary_factory->create(in_constant, X);

    auto gather0 = make_gather(binary, gather_forward, /* axis */ 1);

    auto tanh0 = std::make_shared<Tanh>(gather0);
    auto tanh1 = std::make_shared<Tanh>(gather0);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto gather = make_gather(in_constant, gather_forward, /* axis */ 1);

    NodePtr binary;
    if (!binary_gather_input_idx)
        binary = binary_factory->create(gather0, gather);
    else
        binary = binary_factory->create(gather, gather0);

    auto tanh0 = std::make_shared<Tanh>(binary);
    auto tanh1 = std::make_shared<Tanh>(binary);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace output_transpose_mult_consumers

namespace output_transpose_mult_transposes {

std::shared_ptr<Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                      element::Type input_type,
                                      size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
    if (!binary_gather_input_idx)
        binary = binary_factory->create(X, in_constant);
    else
        binary = binary_factory->create(in_constant, X);

    auto gather0 = make_gather(binary, gather_forward, /* axis */ 1);

    auto tanh0 = std::make_shared<Tanh>(gather0);

    auto gather1 = make_gather(binary, gather_forward, /* axis */ 1);

    auto tanh1 = std::make_shared<Tanh>(gather1);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                               element::Type input_type,
                                               size_t binary_gather_input_idx) {
    const Shape input_shape{1, 20};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto gather0 = make_gather(X, gather_forward, /* axis */ 1);

    auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

    auto gather = make_gather(in_constant, gather_forward, /* axis */ 1);

    NodePtr binary;
    if (!binary_gather_input_idx)
        binary = binary_factory->create(gather0, gather);
    else
        binary = binary_factory->create(gather, gather0);

    auto tanh0 = std::make_shared<Tanh>(binary);
    auto tanh1 = std::make_shared<Tanh>(binary);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace output_transpose_mult_transposes

}  // namespace backward

using CreateGraphF = std::function<
    std::shared_ptr<Model>(BinaryFactoryPtr binary_factory, element::Type input_type, size_t binary_gather_input_idx)>;

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

using TestBinaryParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    CreateGraphFunctionDesc,
                                    element::Type, /* input type */
                                    size_t>;       /*binary_gather_input_idx*/

class TransposeBinaryMultiSinkingFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                           public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryParams>& obj) {
        BinaryFactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        CreateGraphFunctionDesc function_desc;
        element::Type input_type;
        size_t binary_gather_input_idx;

        std::tie(binary_factory, pass_factory, function_desc, input_type, binary_gather_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << function_desc.subtest_name << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_gather_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeBinaryMultiSinkingFixture, CompareFunctions) {
    BinaryFactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    CreateGraphFunctionDesc function_desc;
    element::Type input_type;
    size_t binary_gather_input_idx;

    std::tie(binary_factory, pass_factory, function_desc, input_type, binary_gather_input_idx) = this->GetParam();

    model = function_desc.model_factory(binary_factory, input_type, binary_gather_input_idx);
    model_ref = function_desc.reference_model_factory(binary_factory, input_type, binary_gather_input_idx);
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

INSTANTIATE_TEST_SUITE_P(GatherSinkingBinaryForwardMultiConsumersTestSuite,
                         TransposeBinaryMultiSinkingFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryForward)),
                                            ::testing::ValuesIn(forward_subtests),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)),
                         TransposeBinaryMultiSinkingFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(GatherSinkingBinaryBackwardMultiConsumersTestSuite,
                         TransposeBinaryMultiSinkingFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryBackward)),
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

using TestBinaryParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    CreateGraphFunctionDesc,
                                    element::Type, /* input type */
                                    size_t>;       /*binary_gather_input_idx*/

class TransposeBinaryMultiSinkingBinaryMultiConsumersFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                                               public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryParams>& obj) {
        BinaryFactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        CreateGraphFunctionDesc function_desc;
        element::Type input_type;
        size_t binary_gather_input_idx;

        std::tie(binary_factory, pass_factory, function_desc, input_type, binary_gather_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binaryFactory=" << binary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << function_desc.subtest_name << "/";
        test_name << "inputType=" << input_type << "/";
        test_name << "binaryTransposeInputIdx=" << binary_gather_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeBinaryMultiSinkingBinaryMultiConsumersFixture, CompareFunctions) {
    BinaryFactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    CreateGraphFunctionDesc function_desc;
    element::Type input_type;
    size_t binary_gather_input_idx;

    std::tie(binary_factory, pass_factory, function_desc, input_type, binary_gather_input_idx) = this->GetParam();

    model = function_desc.model_factory(binary_factory, input_type, binary_gather_input_idx);
    model_ref = model->clone();
    pass_factory->registerPass(manager);
}

#define SUBTEST(nmspace, subtest_name) CreateGraphFunctionDesc(nmspace::CreateFunction, subtest_name)

std::vector<CreateGraphFunctionDesc> backward_subtests_binary_consumers = {
    SUBTEST(backward::output_consumers::one_binary, "backwardOutputConsumersOneBinary"),
    SUBTEST(backward::output_consumers::multiple_binaries, "backwardOutputConsumersMultipleBinaries"),
};
#undef SUBTEST

INSTANTIATE_TEST_SUITE_P(GatherSinkingBinaryBackwardBinaryMultiConsumersTestSuite,
                         TransposeBinaryMultiSinkingBinaryMultiConsumersFixture,
                         ::testing::Combine(::testing::ValuesIn(binary_elementwise_factories),
                                            ::testing::Values(CREATE_PASS_FACTORY(GatherSinkingBinaryBackward)),
                                            ::testing::ValuesIn(backward_subtests_binary_consumers),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(binary_transpose_input_indexes)),
                         TransposeBinaryMultiSinkingBinaryMultiConsumersFixture::get_test_name);

}  // namespace no_sinking

}  // namespace mult_consumers

}  // namespace gather_sinking_binary_eltwise
