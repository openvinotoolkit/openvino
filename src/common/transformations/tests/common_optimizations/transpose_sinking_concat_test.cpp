// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_concat.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

using namespace ov;
using namespace ov::opset9;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<Model>;

class IPassFactory {
public:
    IPassFactory(const std::string & type_name) : type_name_(type_name) {}
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
    const std::string & getTypeName() const { return type_name_; }
private:
    const std::string type_name_;
};

using PassFactoryPtr = std::shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    PassFactory(const std::string & type_name) : IPassFactory(type_name) {}
    void registerPass(ov::pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};

#define CREATE_PASS_FACTORY(pass_name) std::make_shared<PassFactory<ov::pass::pass_name>>(#pass_name)

}  // namespace

using CreateGraphConcatF = std::function<std::shared_ptr<Model>(size_t num_concat_ops,
                                                                    element::Type input_type,
                                                                    size_t concat_transpose_input_idx,
                                                                    size_t num_concat_inputs)>;

using TestConcatParams = std::tuple<PassFactoryPtr,
                                    size_t,             /* num_concat_ops */
                                    CreateGraphConcatF, /* model_factory */
                                    CreateGraphConcatF, /* reference_model_factory */
                                    element::Type,  /* input type */
                                    size_t,             /* concat_transpose_input_idx */
                                    size_t>;            /* num_concat_inputs */

class TransposeSinkingConcatTestFixture : public ::testing::WithParamInterface<TestConcatParams>,
                                          public TransformationTestsF {};

namespace {

std::vector<size_t> concat_operations_numbers = {1, 10};

std::vector<size_t> concat_transpose_input_indexes = {0, 2};

}  // namespace

namespace single_consumer {
namespace forward {
namespace one_input_transpose {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                          element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx)
                concat_inputs.push_back(in_op);
            else
                concat_inputs.push_back(std::make_shared<Constant>(input_type, const_shape, Shape{1}));
        }
        in_op = std::make_shared<Concat>(concat_inputs, 1);
    }

    return std::make_shared<Model>(OutputVector{in_op}, ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   element::Type input_type,
                                                   size_t concat_transpose_input_idx,
                                                   size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx) {
                concat_inputs.push_back(in_op);
            } else {
                auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

                auto transpose_reversed_const =
                    std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
                auto transpose_reversed =
                    std::make_shared<Transpose>(in_constant, transpose_reversed_const);

                concat_inputs.push_back(transpose_reversed);
            }
        }
        in_op = std::make_shared<Concat>(concat_inputs, 2);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(OutputVector{transpose0}, ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                          element::Type input_type,
                                          size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        concat_inputs.push_back(in_op);
        for (size_t j = 1; j < num_concat_inputs; ++j) {
            auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});
            auto ng_order1 =
                std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);
            concat_inputs.push_back(transpose1);
        }
        in_op = std::make_shared<Concat>(concat_inputs, 1);
    }

    return std::make_shared<Model>(OutputVector{in_op}, ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   element::Type input_type,
                                                   size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;

        concat_inputs.push_back(in_op);

        for (size_t j = 1; j < num_concat_inputs; ++j) {
            auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

            auto ng_order1 =
                std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);

            auto transpose_reversed_const =
                std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
            auto transpose_reversed = std::make_shared<Transpose>(transpose1, transpose_reversed_const);

            concat_inputs.push_back(transpose_reversed);
        }
        in_op = std::make_shared<Concat>(concat_inputs, 2);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(OutputVector{transpose0}, ParameterVector{X});
}

}  // namespace double_transpose

}  // namespace forward

namespace backward {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                          element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx)
                concat_inputs.push_back(in_op);
            else
                concat_inputs.push_back(std::make_shared<Constant>(input_type, input_shape, Shape{1}));
        }
        in_op = std::make_shared<Concat>(concat_inputs, 1);
    }

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

    return std::make_shared<Model>(OutputVector{transpose0}, ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   element::Type input_type,
                                                   size_t concat_transpose_input_idx,
                                                   size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx) {
                concat_inputs.push_back(in_op);
            } else {
                auto in_constant = std::make_shared<Constant>(input_type, input_shape, Shape{1});

                auto transpose_reversed_const =
                    std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
                auto transpose_reversed =
                    std::make_shared<Transpose>(in_constant, transpose_reversed_const);

                concat_inputs.push_back(transpose_reversed);
            }
        }
        in_op = std::make_shared<Concat>(concat_inputs, 3);
    }

    return std::make_shared<Model>(OutputVector{in_op}, ParameterVector{X});
}

}  // namespace backward
}  // namespace single_consumer

TEST_P(TransposeSinkingConcatTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_concat_ops;
    CreateGraphConcatF model_factory;
    CreateGraphConcatF reference_model_factory;
    element::Type input_type;
    size_t concat_transpose_input_idx;
    size_t num_concat_inputs;
    std::tie(pass_factory,
             num_concat_ops,
             model_factory,
             reference_model_factory,
             input_type,
             concat_transpose_input_idx,
             num_concat_inputs) = this->GetParam();

    model = model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    model_ref = reference_model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingConcatForwardTestSuite,
    TransposeSinkingConcatTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingConcatForward)),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(concat_transpose_input_indexes),
                       ::testing::Values(5)));

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingConcatBackwardTestSuite,
    TransposeSinkingConcatTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingConcatBackward)),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::backward::CreateFunction),
                       ::testing::Values(single_consumer::backward::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(concat_transpose_input_indexes),
                       ::testing::Values(5)));

// --------------------------------------------------------------------------------------

using CreateGraphConcatAllTransposesInputF = std::function<
    std::shared_ptr<Model>(size_t num_concat_ops, element::Type input_type, size_t num_concat_inputs)>;

using TestConcatAllTransposesInputParams =
    std::tuple<PassFactoryPtr,
               size_t,                               /* num_concat_ops */
               CreateGraphConcatAllTransposesInputF, /* model_factory */
               CreateGraphConcatAllTransposesInputF, /* reference_model_factory */
               element::Type,                    /* input type */
               size_t>;                              /* num_concat_inputs */

class TransposeSinkingConcatAllTransposesInputTestFixture
    : public ::testing::WithParamInterface<TestConcatAllTransposesInputParams>,
      public TransformationTestsF {};

TEST_P(TransposeSinkingConcatAllTransposesInputTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_concat_ops;
    CreateGraphConcatAllTransposesInputF model_factory;
    CreateGraphConcatAllTransposesInputF reference_model_factory;
    element::Type input_type;
    size_t num_concat_inputs;
    std::tie(pass_factory, num_concat_ops, model_factory, reference_model_factory, input_type, num_concat_inputs) =
        this->GetParam();

    model = model_factory(num_concat_ops, input_type, num_concat_inputs);
    model_ref = reference_model_factory(num_concat_ops, input_type, num_concat_inputs);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingConcatForwardAllTransposesTestSuite,
    TransposeSinkingConcatAllTransposesInputTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingConcatForward)),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::forward::double_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::double_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::Values(5)));
