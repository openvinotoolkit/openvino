// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_concat.hpp"

#include <functional>

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing;
using namespace transpose_sinking::testing::utils;

namespace {

std::vector<size_t> concat_operations_numbers = {1, 10};
std::vector<size_t> concat_transpose_input_indexes = {0, 2};

NodePtr CreateConcatChain(NodePtr input_node,
                          size_t num_concat_ops,
                          element::Type input_type,
                          size_t concat_transpose_input_idx,
                          size_t num_concat_inputs,
                          const Shape& const_shape,
                          int64_t axis) {
    NodePtr in_op = input_node;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx)
                concat_inputs.push_back(in_op);
            else
                concat_inputs.push_back(std::make_shared<Constant>(input_type, const_shape, Shape{1}));
        }
        in_op = std::make_shared<Concat>(concat_inputs, axis);
    }

    return in_op;
}

NodePtr CreateConcatTransposedChain(NodePtr input_node,
                                    size_t num_concat_ops,
                                    element::Type input_type,
                                    size_t concat_transpose_input_idx,
                                    size_t num_concat_inputs,
                                    const Shape& const_shape,
                                    int64_t axis,
                                    const Shape& transpose_order) {
    NodePtr in_op = input_node;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx) {
                concat_inputs.push_back(in_op);
            } else {
                auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

                auto transpose_const =
                    std::make_shared<Constant>(element::u64, Shape{transpose_order.size()}, transpose_order);
                auto transpose = std::make_shared<Transpose>(in_constant, transpose_const);

                concat_inputs.push_back(transpose);
            }
        }
        in_op = std::make_shared<Concat>(concat_inputs, axis);
    }

    return in_op;
}

NodePtr CreateConcatDoubleTransposedChain(NodePtr input_node,
                                          size_t num_concat_ops,
                                          element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs,
                                          const Shape& const_shape,
                                          int64_t axis,
                                          const Shape& transpose1_order,
                                          const Shape& transpose2_order) {
    NodePtr in_op = input_node;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx) {
                concat_inputs.push_back(in_op);
            } else {
                auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

                auto ng_order1 =
                    std::make_shared<Constant>(element::u64, Shape{transpose1_order.size()}, transpose1_order);
                auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);

                auto ng_order2 =
                    std::make_shared<Constant>(element::u64, Shape{transpose2_order.size()}, transpose2_order);
                auto transpose2 = std::make_shared<Transpose>(transpose1, ng_order2);

                concat_inputs.push_back(transpose2);
            }
        }
        in_op = std::make_shared<Concat>(concat_inputs, axis);
    }

    return in_op;
}

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

    auto concat = CreateConcatChain(transpose0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    const_shape,
                                    /* axis */ 1);

    return std::make_shared<Model>(OutputVector{concat}, ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto concat = CreateConcatTransposedChain(X,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              const_shape,
                                              /* axis */ 2,
                                              /* transpose order */ Shape{0, 3, 1, 2});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    return std::make_shared<Model>(OutputVector{transpose0}, ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops, element::Type input_type, size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto concat = CreateConcatTransposedChain(transpose0,
                                              num_concat_ops,
                                              input_type,
                                              /* concat_transpose_input_idx */ 0,
                                              num_concat_inputs,
                                              input_shape,
                                              /* axis */ 1,
                                              /* transpose order */ Shape{0, 2, 3, 1});

    return std::make_shared<Model>(OutputVector{concat}, ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto concat = CreateConcatDoubleTransposedChain(X,
                                                    num_concat_ops,
                                                    input_type,
                                                    /* concat_transpose_input_idx */ 0,
                                                    num_concat_inputs,
                                                    input_shape,
                                                    /* axis */ 2,
                                                    /* transpose1 order */ Shape{0, 2, 3, 1},
                                                    /* transpose2 order */ Shape{0, 3, 1, 2});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

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

    auto concat = CreateConcatChain(X,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    input_shape,
                                    /* axis */ 1);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

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

    auto concat = CreateConcatTransposedChain(transpose0,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              input_shape,
                                              /* axis */ 3,
                                              /* transpose order */ Shape{0, 2, 3, 1});

    return std::make_shared<Model>(OutputVector{concat}, ParameterVector{X});
}

}  // namespace backward
}  // namespace single_consumer

using CreateGraphConcatF = std::function<std::shared_ptr<Model>(size_t num_concat_ops,
                                                                element::Type input_type,
                                                                size_t concat_transpose_input_idx,
                                                                size_t num_concat_inputs)>;

using TestConcatParams = std::tuple<PassFactoryPtr,
                                    size_t,             /* num_concat_ops */
                                    CreateGraphConcatF, /* model_factory */
                                    CreateGraphConcatF, /* reference_model_factory */
                                    element::Type,      /* input type */
                                    size_t,             /* concat_transpose_input_idx */
                                    size_t>;            /* num_concat_inputs */

class TransposeSinkingConcatTestFixture : public ::testing::WithParamInterface<TestConcatParams>,
                                          public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestConcatParams>& obj) {
        const auto& [pass_factory,
                     num_concat_ops,
                     model_factory,
                     reference_model_factory,
                     input_type,
                     concat_transpose_input_idx,
                     num_concat_inputs] = obj.param;

        std::ostringstream test_name;
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "numConcatOps=" << num_concat_ops << "/";
        test_name << "concatTransposeInputIdx=" << concat_transpose_input_idx << "/";
        test_name << "numConcatInputs=" << num_concat_inputs << "/";
        test_name << "inputType=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingConcatTestFixture, CompareFunctions) {
    const auto& [pass_factory,
                 num_concat_ops,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 concat_transpose_input_idx,
                 num_concat_inputs] = this->GetParam();

    model = model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    model_ref = reference_model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TSConcatForwardTestSuite,
    TransposeSinkingConcatTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TSConcatForward)),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(concat_transpose_input_indexes),
                       ::testing::Values(5)),
    TransposeSinkingConcatTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSConcatBackwardTestSuite,
                         TransposeSinkingConcatTestFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TSConcatBackward)),
                                            ::testing::ValuesIn(concat_operations_numbers),
                                            ::testing::Values(single_consumer::backward::CreateFunction),
                                            ::testing::Values(single_consumer::backward::CreateReferenceFunction),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)),
                         TransposeSinkingConcatTestFixture::get_test_name);

// --------------------------------------------------------------------------------------

using CreateGraphConcatAllTransposesInputF =
    std::function<std::shared_ptr<Model>(size_t num_concat_ops, element::Type input_type, size_t num_concat_inputs)>;

using TestConcatAllTransposesInputParams =
    std::tuple<PassFactoryPtr,
               size_t,                               /* num_concat_ops */
               CreateGraphConcatAllTransposesInputF, /* model_factory */
               CreateGraphConcatAllTransposesInputF, /* reference_model_factory */
               element::Type,                        /* input type */
               size_t>;                              /* num_concat_inputs */

class TransposeSinkingConcatAllTransposesInputTestFixture
    : public ::testing::WithParamInterface<TestConcatAllTransposesInputParams>,
      public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestConcatAllTransposesInputParams>& obj) {
        const auto& [pass_factory,
                     num_concat_ops,
                     model_factory,
                     reference_model_factory,
                     input_type,
                     num_concat_inputs] = obj.param;

        std::ostringstream test_name;
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "numConcatOps=" << num_concat_ops << "/";
        test_name << "numConcatInputs=" << num_concat_inputs << "/";
        test_name << "inputType=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingConcatAllTransposesInputTestFixture, CompareFunctions) {
    const auto& [pass_factory, num_concat_ops, model_factory, reference_model_factory, input_type, num_concat_inputs] =
        this->GetParam();

    model = model_factory(num_concat_ops, input_type, num_concat_inputs);
    model_ref = reference_model_factory(num_concat_ops, input_type, num_concat_inputs);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TSConcatForwardAllTransposesTestSuite,
    TransposeSinkingConcatAllTransposesInputTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TSConcatForward)),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::forward::double_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::double_transpose::CreateReferenceFunction),
                       ::testing::Values(element::f32),
                       ::testing::Values(5)),
    TransposeSinkingConcatAllTransposesInputTestFixture::get_test_name);

// --------------------------------------------------------------------------------------

namespace mult_consumers {
namespace forward {
namespace input_transpose_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto tanh = std::make_shared<Tanh>(transpose0);

    auto concat = CreateConcatChain(transpose0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    const_shape,
                                    /* axis */ 1);

    return std::make_shared<Model>(OutputVector{concat, tanh}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
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

    auto concat = CreateConcatTransposedChain(X,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              const_shape,
                                              /* axis */ 2,
                                              /* transpose order */ Shape{0, 3, 1, 2});

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(concat, ng_order1);

    return std::make_shared<Model>(ov::OutputVector{transpose1, tanh}, ov::ParameterVector{X});
}

}  // namespace input_transpose_consumers

namespace output_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto concat = CreateConcatChain(transpose0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    const_shape,
                                    /* axis */ 1);

    auto tanh1 = std::make_shared<Tanh>(concat);
    auto tanh2 = std::make_shared<Tanh>(concat);

    return std::make_shared<Model>(ov::OutputVector{tanh1, tanh2}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

    auto transpose_reversed_const = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
    auto transpose_reversed = std::make_shared<Transpose>(in_constant, transpose_reversed_const);

    auto concat = CreateConcatTransposedChain(X,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              const_shape,
                                              /* axis */ 2,
                                              /* transpose order */ Shape{0, 3, 1, 2});

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    auto tanh1 = std::make_shared<Tanh>(transpose0);
    auto tanh2 = std::make_shared<Tanh>(transpose0);

    return std::make_shared<Model>(ov::OutputVector{tanh1, tanh2}, ov::ParameterVector{X});
}

}  // namespace output_consumers

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh = std::make_shared<Tanh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto concat = CreateConcatChain(transpose0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    const_shape,
                                    /* axis */ 1);

    return std::make_shared<Model>(ov::OutputVector{concat, tanh}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};
    const Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh = std::make_shared<Tanh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    NodePtr binary;
    auto in_constant = std::make_shared<Constant>(input_type, const_shape, Shape{1});

    auto concat = CreateConcatTransposedChain(X,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              const_shape,
                                              /* axis */ 2,
                                              /* transpose order */ Shape{0, 3, 1, 2});

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(concat, ng_order1);

    return std::make_shared<Model>(ov::OutputVector{transpose1, tanh}, ov::ParameterVector{X});
}

}  // namespace input_node_consumers

}  // namespace forward

namespace backward {

namespace output_consumers {

namespace one_binary {
std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    auto concat = CreateConcatChain(tanh0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    input_shape,
                                    /* axis */ 1);

    auto tanh = std::make_shared<Tanh>(concat);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0, tanh}, ov::ParameterVector{X});
}

}  // namespace one_binary

namespace multiple_binaries {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    auto concat = CreateConcatChain(tanh0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    input_shape,
                                    /* axis */ 1);

    auto tanh = std::make_shared<Tanh>(concat);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    return std::make_shared<Model>(ov::OutputVector{transpose0, tanh}, ov::ParameterVector{X});
}

}  // namespace multiple_binaries

}  // namespace output_consumers

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    auto concat = CreateConcatChain(tanh0,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    input_shape,
                                    /* axis */ 1);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    auto tanh1 = std::make_shared<Tanh>(tanh0);

    return std::make_shared<Model>(ov::OutputVector{transpose0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto tanh0 = std::make_shared<Tanh>(X);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(tanh0, ng_order0);

    auto concat = CreateConcatTransposedChain(transpose0,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              input_shape,
                                              /* axis */ 3,
                                              /* transpose order */ Shape{0, 2, 3, 1});

    auto tanh1 = std::make_shared<Tanh>(tanh0);

    return std::make_shared<Model>(ov::OutputVector{concat, tanh1}, ov::ParameterVector{X});
}

}  // namespace input_node_consumers

namespace output_transpose_mult_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto concat = CreateConcatChain(X,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    input_shape,
                                    /* axis */ 1);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    auto tanh0 = std::make_shared<Tanh>(transpose0);
    auto tanh1 = std::make_shared<Tanh>(transpose0);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto concat = CreateConcatTransposedChain(transpose0,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              input_shape,
                                              /* axis */ 3,
                                              /* transpose order */ Shape{0, 2, 3, 1});

    auto tanh0 = std::make_shared<Tanh>(concat);
    auto tanh1 = std::make_shared<Tanh>(concat);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace output_transpose_mult_consumers

namespace output_transpose_mult_transposes {

std::shared_ptr<Model> CreateFunction(size_t num_concat_ops,
                                      element::Type input_type,
                                      size_t concat_transpose_input_idx,
                                      size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto concat = CreateConcatChain(X,
                                    num_concat_ops,
                                    input_type,
                                    concat_transpose_input_idx,
                                    num_concat_inputs,
                                    input_shape,
                                    /* axis */ 1);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(concat, ng_order0);

    auto tanh0 = std::make_shared<Tanh>(transpose0);

    auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<Transpose>(concat, ng_order1);

    auto tanh1 = std::make_shared<Tanh>(transpose1);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_concat_ops,
                                               element::Type input_type,
                                               size_t concat_transpose_input_idx,
                                               size_t num_concat_inputs) {
    const Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

    auto concat = CreateConcatTransposedChain(transpose0,
                                              num_concat_ops,
                                              input_type,
                                              concat_transpose_input_idx,
                                              num_concat_inputs,
                                              input_shape,
                                              /* axis */ 3,
                                              /* transpose order */ Shape{0, 2, 3, 1});

    auto tanh0 = std::make_shared<Tanh>(concat);
    auto tanh1 = std::make_shared<Tanh>(concat);

    return std::make_shared<Model>(ov::OutputVector{tanh0, tanh1}, ov::ParameterVector{X});
}

}  // namespace output_transpose_mult_transposes

}  // namespace backward

using CreateGraphF = std::function<std::shared_ptr<Model>(size_t num_concat_ops,
                                                          element::Type input_type,
                                                          size_t concat_transpose_input_idx,
                                                          size_t num_concat_inputs)>;

struct CreateGraphFunctionDesc {
    CreateGraphFunctionDesc() = default;
    CreateGraphFunctionDesc(CreateGraphF a_model_factory, CreateGraphF a_ref_model_factory, std::string a_subtest_name)
        : model_factory(a_model_factory),
          reference_model_factory(a_ref_model_factory),
          subtest_name(a_subtest_name) {}
    CreateGraphF model_factory;
    CreateGraphF reference_model_factory;
    std::string subtest_name;
};

using TestConcatParams = std::tuple<PassFactoryPtr,
                                    size_t, /* num_concat_ops */
                                    CreateGraphFunctionDesc,
                                    element::Type, /* input type */
                                    size_t,        /* concat_transpose_input_idx */
                                    size_t>;       /* num_concat_inputs */

class TransposeConcatMultiSinkingFixture : public ::testing::WithParamInterface<TestConcatParams>,
                                           public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestConcatParams>& obj) {
        const auto& [pass_factory,
                     num_concat_ops,
                     function_desc,
                     input_type,
                     concat_transpose_input_idx,
                     num_concat_inputs] = obj.param;

        std::ostringstream test_name;
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "functionDesc=" << function_desc.subtest_name << "/";
        test_name << "numConcatOps=" << num_concat_ops << "/";
        test_name << "concatTransposeInputIdx=" << concat_transpose_input_idx << "/";
        test_name << "numConcatInputs=" << num_concat_inputs << "/";
        test_name << "inputType=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeConcatMultiSinkingFixture, CompareFunctions) {
    const auto& [pass_factory,
                 num_concat_ops,
                 function_desc,
                 input_type,
                 concat_transpose_input_idx,
                 num_concat_inputs] = this->GetParam();

    model = function_desc.model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    model_ref = function_desc.reference_model_factory(num_concat_ops,
                                                      input_type,
                                                      concat_transpose_input_idx,
                                                      num_concat_inputs);
    pass_factory->registerPass(manager);
}

#define SUBTEST(nmspace, subtest_name) \
    CreateGraphFunctionDesc(nmspace::CreateFunction, nmspace::CreateReferenceFunction, subtest_name)

std::vector<CreateGraphFunctionDesc> forward_subtests = {
    SUBTEST(forward::input_transpose_consumers, "forwardInputTransposeConsumers"),
    SUBTEST(forward::output_consumers, "forwardOutputConsumers"),
    SUBTEST(forward::input_node_consumers, "forwardInputNodeConsumers")};

std::vector<CreateGraphFunctionDesc> backward_subtests = {
    SUBTEST(backward::input_node_consumers, "backwardInputNodeConsumers"),
    SUBTEST(backward::output_transpose_mult_consumers, "backwardOutputTransposeMultConsumers"),
    SUBTEST(backward::output_transpose_mult_transposes, "outputTransposeMultTransposes")};

#undef SUBTEST

INSTANTIATE_TEST_SUITE_P(TSConcatForwardMultiConsumersTestSuite,
                         TransposeConcatMultiSinkingFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TSConcatForward)),
                                            ::testing::ValuesIn(concat_operations_numbers),
                                            ::testing::ValuesIn(forward_subtests),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)),
                         TransposeConcatMultiSinkingFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TSConcatBackwardMultiConsumersTestSuite,
                         TransposeConcatMultiSinkingFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TSConcatBackward)),
                                            ::testing::ValuesIn(concat_operations_numbers),
                                            ::testing::ValuesIn(backward_subtests),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)),
                         TransposeConcatMultiSinkingFixture::get_test_name);

namespace no_sinking {

struct CreateGraphFunctionNoSinkingDesc {
    CreateGraphFunctionNoSinkingDesc() = default;
    CreateGraphFunctionNoSinkingDesc(CreateGraphF a_model_factory, std::string a_subtest_name)
        : model_factory(a_model_factory),
          subtest_name(a_subtest_name) {}
    CreateGraphF model_factory;
    std::string subtest_name;
};

using TestConcatParams = std::tuple<PassFactoryPtr,
                                    size_t, /* num_concat_ops */
                                    CreateGraphFunctionNoSinkingDesc,
                                    element::Type, /* input type */
                                    size_t,        /* concat_transpose_input_idx */
                                    size_t>;       /* num_concat_inputs */

class TransposeConcatMultiSinkingConcatConsumersFixture : public ::testing::WithParamInterface<TestConcatParams>,
                                                          public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestConcatParams>& obj) {
        const auto& [pass_factory,
                     num_concat_ops,
                     function_desc,
                     input_type,
                     concat_transpose_input_idx,
                     num_concat_inputs] = obj.param;

        std::ostringstream test_name;
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        test_name << "functionDesc=" << function_desc.subtest_name << "/";
        test_name << "numConcatOps=" << num_concat_ops << "/";
        test_name << "concatTransposeInputIdx=" << concat_transpose_input_idx << "/";
        test_name << "numConcatInputs=" << num_concat_inputs << "/";
        test_name << "inputType=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeConcatMultiSinkingConcatConsumersFixture, CompareFunctions) {
    const auto& [pass_factory,
                 num_concat_ops,
                 function_desc,
                 input_type,
                 concat_transpose_input_idx,
                 num_concat_inputs] = this->GetParam();

    model = function_desc.model_factory(num_concat_ops, input_type, concat_transpose_input_idx, num_concat_inputs);
    model_ref = model->clone();
    pass_factory->registerPass(manager);
}

#define SUBTEST(nmspace, subtest_name) CreateGraphFunctionNoSinkingDesc(nmspace::CreateFunction, subtest_name)

std::vector<CreateGraphFunctionNoSinkingDesc> backward_subtests_no_sinking = {
    SUBTEST(backward::output_consumers::one_binary, "backwardOutputConsumersOneBinary"),
    SUBTEST(backward::output_consumers::multiple_binaries, "backwardOutputConsumersMultipleBinaries")};

#undef SUBTEST

INSTANTIATE_TEST_SUITE_P(TSConcatBackwardMultiConsumersTestSuite,
                         TransposeConcatMultiSinkingConcatConsumersFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TSConcatBackward)),
                                            ::testing::ValuesIn(concat_operations_numbers),
                                            ::testing::ValuesIn(backward_subtests_no_sinking),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)),
                         TransposeConcatMultiSinkingConcatConsumersFixture::get_test_name);

}  // namespace no_sinking

}  // namespace mult_consumers
