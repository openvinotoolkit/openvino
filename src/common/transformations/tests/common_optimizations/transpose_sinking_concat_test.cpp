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

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<ov::Model>;
using Output = ov::Output<ov::Node>;

namespace {
    std::string to_string(const ov::Shape & shape) {
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
}

// ----------------------------------------------------------------------------

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

using CreateGraphConcatF = std::function<std::shared_ptr<ov::Model>(size_t num_concat_ops,
                                                                    ov::element::Type input_type,
                                                                    size_t concat_transpose_input_idx,
                                                                    size_t num_concat_inputs)>;

using TestConcatParams = std::tuple<PassFactoryPtr,
                                    size_t,             /* num_concat_ops */
                                    CreateGraphConcatF, /* model_factory */
                                    CreateGraphConcatF, /* reference_model_factory */
                                    ov::element::Type,  /* input type */
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

std::shared_ptr<ov::Model> CreateFunction(size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx)
                concat_inputs.push_back(in_op);
            else
                concat_inputs.push_back(std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1}));
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   ov::element::Type input_type,
                                                   size_t concat_transpose_input_idx,
                                                   size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx) {
                concat_inputs.push_back(in_op);
            } else {
                auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});

                auto transpose_reversed_const =
                    std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
                auto transpose_reversed =
                    std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

                concat_inputs.push_back(transpose_reversed);
            }
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 2);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {

std::shared_ptr<ov::Model> CreateFunction(size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        concat_inputs.push_back(in_op);
        for (size_t j = 1; j < num_concat_inputs; ++j) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);
            concat_inputs.push_back(transpose1);
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   ov::element::Type input_type,
                                                   size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;

        concat_inputs.push_back(in_op);

        for (size_t j = 1; j < num_concat_inputs; ++j) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

            auto transpose_reversed_const =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(transpose1, transpose_reversed_const);

            concat_inputs.push_back(transpose_reversed);
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 2);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace double_transpose

}  // namespace forward

namespace backward {

std::shared_ptr<ov::Model> CreateFunction(size_t num_concat_ops,
                                          ov::element::Type input_type,
                                          size_t concat_transpose_input_idx,
                                          size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_concat_ops; ++i) {
        ov::OutputVector concat_inputs;
        for (size_t j = 0; j < num_concat_inputs; ++j) {
            if (j == concat_transpose_input_idx)
                concat_inputs.push_back(in_op);
            else
                concat_inputs.push_back(std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1}));
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_concat_ops,
                                                   ov::element::Type input_type,
                                                   size_t concat_transpose_input_idx,
                                                   size_t num_concat_inputs) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;
            for (size_t j = 0; j < num_concat_inputs; ++j) {
                if (j == concat_transpose_input_idx) {
                    concat_inputs.push_back(in_op);
                } else {
                    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
                    // FIXME: name that is not reversed
                    auto transpose_reversed_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                    auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

                concat_inputs.push_back(transpose_reversed);
            }
        }
        in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 3);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

}  // namespace backward
}  // namespace single_consumer

TEST_P(TransposeSinkingConcatTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_concat_ops;
    CreateGraphConcatF model_factory;
    CreateGraphConcatF reference_model_factory;
    ov::element::Type input_type;
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

INSTANTIATE_TEST_SUITE_P(TransposeSinkingConcatForwardTestSuite, TransposeSinkingConcatTestFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingConcatForward)),
                                            ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateFunction),
                       ::testing::Values(single_consumer::forward::one_input_transpose::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)));

INSTANTIATE_TEST_SUITE_P(TransposeSinkingConcatBackwardTestSuite, TransposeSinkingConcatTestFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingConcatBackward)),
                                            ::testing::ValuesIn(concat_operations_numbers),
                                            ::testing::Values(single_consumer::backward::CreateFunction),
                                            ::testing::Values(single_consumer::backward::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(concat_transpose_input_indexes),
                                            ::testing::Values(5)));

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingConcatBackwardTestSuite,
    TransposeSinkingConcatTestFixture,
    ::testing::Combine(::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingConcatBackward>()),
                       ::testing::ValuesIn(concat_operations_numbers),
                       ::testing::Values(single_consumer::backward::CreateFunction),
                       ::testing::Values(single_consumer::backward::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(concat_transpose_input_indexes),
                       ::testing::Values(5)));

// --------------------------------------------------------------------------------------

using CreateGraphConcatAllTransposesInputF = std::function<
    std::shared_ptr<ov::Model>(size_t num_concat_ops, ov::element::Type input_type, size_t num_concat_inputs)>;

using TestConcatAllTransposesInputParams =
    std::tuple<PassFactoryPtr,
               size_t,                               /* num_concat_ops */
               CreateGraphConcatAllTransposesInputF, /* model_factory */
               CreateGraphConcatAllTransposesInputF, /* reference_model_factory */
               ov::element::Type,                    /* input type */
               size_t>;                              /* num_concat_inputs */

class TransposeSinkingConcatAllTransposesInputTestFixture
    : public ::testing::WithParamInterface<TestConcatAllTransposesInputParams>,
      public TransformationTestsF {};

TEST_P(TransposeSinkingConcatAllTransposesInputTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_concat_ops;
    CreateGraphConcatAllTransposesInputF model_factory;
    CreateGraphConcatAllTransposesInputF reference_model_factory;
    ov::element::Type input_type;
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
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(5)));

// --------------------------------------------------------------------------------------


using CreateGraphConcatIncompatShapesF = std::function<std::shared_ptr<ov::Model>(ov::element::Type input_type,
                                                                    ov::Shape input_shape,
                                                                    ov::Shape constant_shape,
                                                                    size_t num_concat_inputs,
                                                                    size_t concat_transpose_input_idx)>;

using TesConcatIncompatShapesParams = std::tuple<PassFactoryPtr,
                                    ov::Shape,          /* input shape */
                                    ov::Shape,          /* constant_shape */
                                    CreateGraphConcatIncompatShapesF, /* model_factory */
                                    CreateGraphConcatIncompatShapesF, /* reference_model_factory */
                                    ov::element::Type,  /* input type */
                                    size_t,             /* num_concat_inputs */
                                    size_t>;            /* concat_transpose_input_idx */

class TransposeSinkingConcatIncompatShapesTestFixture : public ::testing::WithParamInterface<TesConcatIncompatShapesParams>,
                                          public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TesConcatIncompatShapesParams>& obj) {
        PassFactoryPtr pass_factory;
        ov::Shape input_shape;
        ov::Shape constant_shape;
        CreateGraphConcatIncompatShapesF model_factory;
        CreateGraphConcatIncompatShapesF reference_model_factory;
        ov::element::Type input_type;
        size_t num_concat_inputs;
        size_t concat_transpose_input_idx;
        std::tie(pass_factory,
             input_shape,
             constant_shape,
             model_factory,
             reference_model_factory,
             input_type,
             num_concat_inputs,
             concat_transpose_input_idx) = obj.param;
        
        std::ostringstream test_name;
        test_name << "pass_factory=" << pass_factory->getTypeName() << "_";
        test_name << "input_shape=" << to_string(input_shape) << "_";
        test_name << "constant_shape=" << to_string(constant_shape) << "_";
        test_name << "input_type=" << input_type << "_";
        test_name << "num_concat_inputs=" << num_concat_inputs << "_";
        test_name << "concat_transpose_input_idx=" << concat_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingConcatIncompatShapesTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    ov::Shape input_shape;
    ov::Shape constant_shape;
    CreateGraphConcatIncompatShapesF model_factory;
    CreateGraphConcatIncompatShapesF reference_model_factory;
    ov::element::Type input_type;
    size_t num_concat_inputs;
    size_t concat_transpose_input_idx;
    std::tie(pass_factory,
             input_shape,
             constant_shape,
             model_factory,
             reference_model_factory,
             input_type,
             num_concat_inputs,
             concat_transpose_input_idx) = this->GetParam();

    model = model_factory(input_type, input_shape, constant_shape, num_concat_inputs, concat_transpose_input_idx);
    model_ref = reference_model_factory(input_type, input_shape, constant_shape, num_concat_inputs, concat_transpose_input_idx);
    pass_factory->registerPass(manager);
}

namespace concat {
namespace single_consumer {
namespace backward {
namespace incompat_shapes {

std::shared_ptr<ov::Model> CreateFunction(ov::element::Type input_type,
                                          ov::Shape input_shape,
                                          ov::Shape constant_shape,
                                          size_t num_concat_inputs,
                                          size_t concat_transpose_input_idx) {
    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, constant_shape, ov::Shape{1});
    
    ov::OutputVector concat_inputs;
    for (size_t j = 0; j < num_concat_inputs; ++j) {
        if (j == concat_transpose_input_idx)
            concat_inputs.push_back(X);
        else
            concat_inputs.push_back(std::make_shared<ov::opset9::Constant>(input_type, constant_shape, ov::Shape{1}));
    }
    auto concat = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(concat, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(ov::element::Type input_type,
                                          ov::Shape input_shape,
                                          ov::Shape constant_shape,
                                          size_t num_concat_inputs,
                                          size_t concat_transpose_input_idx) {
    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    ov::OutputVector concat_inputs;
    for (size_t j = 0; j < num_concat_inputs; ++j) {
        if (j == concat_transpose_input_idx) {
            concat_inputs.push_back(transpose0);
        } else {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

            std::vector<size_t> dims(input_shape.size() - constant_shape.size());
            std::iota(dims.begin(), dims.end(), 0);
            auto unsqueeze_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{dims.size()}, dims);
            auto unsqeeze = std::make_shared<ov::opset9::Unsqueeze>(in_constant, unsqueeze_const);
            // FIXME: name that is not reversed
            auto transpose_reversed_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, constant_shape, ov::Shape{0, 2, 3, 1});
            auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(unsqeeze, transpose_reversed_const);

            concat_inputs.push_back(transpose_reversed);
        }
    }
    auto concat = std::make_shared<ov::opset9::Concat>(concat_inputs, 1);

    return std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{X});
}

std::vector<ov::Shape> constant_shapes = {ov::Shape{96, 55, 55}, ov::Shape{1}};
std::vector<size_t> concat_transpose_input_indexes = {0, 2};
std::vector<size_t> num_concat_inputs = {3, 5};

} // namespace incompat_shapes
} // namespace backward
} // namespace single_consumer
} // namespace concat

INSTANTIATE_TEST_SUITE_P(TransposeSinkingConcatIncompatShapesTestSuite, TransposeSinkingConcatIncompatShapesTestFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingConcatBackward)),
                                            ::testing::Values(ov::Shape{1, 96, 55, 55}),
                                            ::testing::ValuesIn(concat::single_consumer::backward::incompat_shapes::constant_shapes),
                       ::testing::Values(concat::single_consumer::backward::incompat_shapes::CreateFunction),
                       ::testing::Values(concat::single_consumer::backward::incompat_shapes::CreateReferenceFunction),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(concat::single_consumer::backward::incompat_shapes::num_concat_inputs),
                                            ::testing::ValuesIn(concat::single_consumer::backward::incompat_shapes::concat_transpose_input_indexes)),
                                            TransposeSinkingConcatIncompatShapesTestFixture::get_test_name);
