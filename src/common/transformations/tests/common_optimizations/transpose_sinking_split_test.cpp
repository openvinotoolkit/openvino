// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_split.hpp>
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

// --------------------------------------------------------------------------------------

using CreateGraphSplitForwardF = std::function<
    std::shared_ptr<ov::Model>(size_t num_split_ops, size_t num_split_outputs, ov::element::Type input_type)>;

using TestSplitForwardParams = std::tuple<PassFactoryPtr,
                                          size_t,                   /* num_split_ops */
                                          size_t,                   /* num_split_outputs */
                                          CreateGraphSplitForwardF, /* model_factory */
                                          CreateGraphSplitForwardF, /* reference_model_factory */
                                          ov::element::Type> /* input type */;

class TransposeSinkingSplitForwardTestFixture : public ::testing::WithParamInterface<TestSplitForwardParams>,
                                                public TransformationTestsF {};

namespace {

std::vector<size_t> split_operations_numbers = {1, 10};

std::vector<size_t> split_outputs_numbers = {2, 3};

}  // namespace

namespace split {
namespace forward {
std::shared_ptr<ov::Model> CreateFunction(size_t num_split_ops,
                                          size_t num_split_outputs,
                                          ov::element::Type input_type) {
    const ov::Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    ov::OutputVector outputs;
    Output in_op = transpose0->output(0);
    for (size_t i = 0; i < num_split_ops; ++i) {
        auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{}, 2);
        auto split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);
        for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
            outputs.push_back(split->output(num_output));
        }
        in_op = split->output(num_split_outputs - 1);
    }
    outputs.push_back(in_op);

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t num_split_ops,
                                                   size_t num_split_outputs,
                                                   ov::element::Type input_type) {
    const ov::Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    ov::OutputVector outputs;
    Output in_op = X->output(0);
    for (size_t i = 0; i < num_split_ops; ++i) {
        auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{}, 1);
        auto split = std::make_shared<ov::opset9::Split>(in_op, split_axis_const, num_split_outputs);
        for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
            auto ng_order0 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(split->output(num_output), ng_order0);
            outputs.push_back(transpose0);
        }
        in_op = split->output(num_split_outputs - 1);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);
    outputs.push_back(transpose0);

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

}  // namespace forward
}  // namespace split

TEST_P(TransposeSinkingSplitForwardTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_split_ops;
    size_t num_split_outputs;
    CreateGraphSplitForwardF model_factory;
    CreateGraphSplitForwardF reference_model_factory;
    ov::element::Type input_type;
    std::tie(pass_factory, num_split_ops, num_split_outputs, model_factory, reference_model_factory, input_type) =
        this->GetParam();

    model = model_factory(num_split_ops, num_split_outputs, input_type);
    model_ref = reference_model_factory(num_split_ops, num_split_outputs, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitForwardTestSuite,
    TransposeSinkingSplitForwardTestFixture,
    ::testing::Combine(::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingSplitForward>()),
                       ::testing::ValuesIn(split_operations_numbers),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(split::forward::CreateFunction),
                       ::testing::Values(split::forward::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32)));

// --------------------------------------------------------------------------------------

using CreateGraphSplitBackwardF = std::function<
    std::shared_ptr<ov::Model>(size_t split_tree_depth, size_t num_split_outputs, ov::element::Type input_type)>;

using TestSplitBackwardParams = std::tuple<PassFactoryPtr,
                                           size_t,                    /* split_tree_depth */
                                           size_t,                    /* num_split_outputs */
                                           CreateGraphSplitBackwardF, /* model_factory */
                                           CreateGraphSplitBackwardF, /* reference_model_factory */
                                           ov::element::Type> /* input type */;

class TransposeSinkingSplitBackwardTestFixture : public ::testing::WithParamInterface<TestSplitBackwardParams>,
                                                 public TransformationTestsF {};

namespace {
std::vector<size_t> split_tree_depth_nums = {1, 3};
}  // namespace

// --------------------------------------------------------------------------------------

namespace split {
namespace backward {

class SplitFactory {
public:
    SplitFactory(size_t axis, size_t n_outputs, ov::element::Type elem_type)
        : _axis(axis),
          _n_outputs(n_outputs),
          _elem_type(elem_type) {}
    NodePtr create(Output parent) const {
        auto split_axis_const = std::make_shared<ov::opset9::Constant>(_elem_type, ov::Shape{}, _axis);
        return std::make_shared<ov::opset9::Split>(parent, split_axis_const, _n_outputs);
    }

private:
    const size_t _axis;
    const size_t _n_outputs;
    const ov::element::Type _elem_type;
};

void CreateSplitTree(size_t max_depth,
                     size_t depth,
                     Output parent,
                     const SplitFactory& split_factory,
                     ov::OutputVector& leaves) {
    if (depth == max_depth) {
        leaves.push_back(parent);
        return;
    }

    auto split = split_factory.create(parent);

    for (size_t output_idx = 0; output_idx < split->get_output_size(); ++output_idx) {
        CreateSplitTree(max_depth, depth + 1, split->output(output_idx), split_factory, leaves);
    }
}

std::shared_ptr<ov::Model> CreateFunction(size_t split_tree_depth,
                                          size_t num_split_outputs,
                                          ov::element::Type input_type) {
    const size_t split_input_dim_value = static_cast<size_t>(std::pow(num_split_outputs, split_tree_depth + 1));
    const ov::Shape input_shape{96, split_input_dim_value, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    ov::OutputVector split_tree_leaves;
    {
        SplitFactory split_factory(/* axis */ 1, num_split_outputs, /* elem_type */ ov::element::u64);
        CreateSplitTree(split_tree_depth, /* depth */ 0, X->output(0), split_factory, split_tree_leaves);
    }

    ov::OutputVector outputs;
    for (auto& split_tree_leaf : split_tree_leaves) {
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);

        const size_t split_dim_current_value =
            static_cast<size_t>(split_input_dim_value / std::pow(num_split_outputs, split_tree_depth));
        auto reshape_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                    ov::Shape{3},
                                                                    ov::Shape{96, 55, split_dim_current_value * 55});
        auto reshape = std::make_shared<ov::opset9::Reshape>(transpose, reshape_const, false);
        outputs.push_back(reshape);
    }

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(size_t split_tree_depth,
                                                   size_t num_split_outputs,
                                                   ov::element::Type input_type) {
    const size_t split_input_dim_value = static_cast<size_t>(std::pow(num_split_outputs, split_tree_depth + 1));
    const ov::Shape input_shape{96, split_input_dim_value, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
    auto transpose = std::make_shared<ov::opset9::Transpose>(X, ng_order);

    ov::OutputVector split_tree_leaves;
    {
        SplitFactory split_factory(/* axis */ 2, num_split_outputs, /* elem_type */ ov::element::u64);
        CreateSplitTree(split_tree_depth, /* depth */ 0, transpose->output(0), split_factory, split_tree_leaves);
    }

    ov::OutputVector outputs;
    for (auto& split_tree_leaf : split_tree_leaves) {
        const size_t split_dim_current_value =
            static_cast<size_t>(split_input_dim_value / std::pow(num_split_outputs, split_tree_depth));
        auto reshape_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                    ov::Shape{3},
                                                                    ov::Shape{96, 55, split_dim_current_value * 55});
        auto reshape = std::make_shared<ov::opset9::Reshape>(split_tree_leaf, reshape_const, false);
        outputs.push_back(reshape);
    }

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

}  // namespace backward
}  // namespace split

TEST_P(TransposeSinkingSplitBackwardTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t split_tree_depth;
    size_t num_split_outputs;
    CreateGraphSplitBackwardF model_factory;
    CreateGraphSplitBackwardF reference_model_factory;
    ov::element::Type input_type;
    std::tie(pass_factory, split_tree_depth, num_split_outputs, model_factory, reference_model_factory, input_type) =
        this->GetParam();

    model = model_factory(split_tree_depth, num_split_outputs, input_type);
    model_ref = reference_model_factory(split_tree_depth, num_split_outputs, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitBackwardTestSuite,
    TransposeSinkingSplitBackwardTestFixture,
    ::testing::Combine(::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingSplitBackward>()),
                       ::testing::ValuesIn(split_tree_depth_nums),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(split::backward::CreateFunction),
                       ::testing::Values(split::backward::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32)));

using TransposeInsertF = std::function<ov::OutputVector(const ov::OutputVector& split_tree_leaves)>;

using CreateGraphSplitBackwardRestrictF =
    std::function<std::shared_ptr<ov::Model>(size_t split_tree_depth,
                                             size_t num_split_outputs,
                                             ov::element::Type input_type,
                                             TransposeInsertF tranpose_insert_function)>;

using TestSplitBackwardRestrictParams = std::tuple<PassFactoryPtr,
                                                   size_t,                            /* split_tree_depth */
                                                   size_t,                            /* num_split_outputs */
                                                   CreateGraphSplitBackwardRestrictF, /* model_factory */
                                                   ov::element::Type,                 /* input type */
                                                   TransposeInsertF>;                 /* insert transpose function */

class TransposeSinkingSplitBackwardRestrictTestFixture
    : public ::testing::WithParamInterface<TestSplitBackwardRestrictParams>,
      public TransformationTestsF {};

TEST_P(TransposeSinkingSplitBackwardRestrictTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t split_tree_depth;
    size_t num_split_outputs;
    CreateGraphSplitBackwardRestrictF model_factory;
    ov::element::Type input_type;
    TransposeInsertF tranpose_insert_function;
    std::tie(pass_factory, split_tree_depth, num_split_outputs, model_factory, input_type, tranpose_insert_function) =
        this->GetParam();

    model = model_factory(split_tree_depth, num_split_outputs, input_type, tranpose_insert_function);
    model_ref = model->clone();
    pass_factory->registerPass(manager);
}
namespace split {
namespace backward {
namespace restrictions {

std::shared_ptr<ov::Model> CreateFunction(size_t split_tree_depth,
                                          size_t num_split_outputs,
                                          ov::element::Type input_type,
                                          TransposeInsertF transpose_insert_func) {
    const size_t split_input_dim_value = static_cast<size_t>(std::pow(num_split_outputs, split_tree_depth + 1));
    const ov::Shape input_shape{96, split_input_dim_value, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    ov::OutputVector split_tree_leaves;
    {
        SplitFactory split_factory(/* axis */ 1, num_split_outputs, /* elem_type */ ov::element::u64);
        CreateSplitTree(split_tree_depth, /* depth */ 0, X->output(0), split_factory, split_tree_leaves);
    }

    ov::OutputVector outputs;
    for (auto& split_tree_leaf : transpose_insert_func(split_tree_leaves)) {
        const size_t split_dim_current_value =
            static_cast<size_t>(split_input_dim_value / std::pow(num_split_outputs, split_tree_depth));
        auto reshape_const = std::make_shared<ov::opset9::Constant>(ov::element::u64,
                                                                    ov::Shape{3},
                                                                    ov::Shape{96, 55, split_dim_current_value * 55});
        auto reshape = std::make_shared<ov::opset9::Reshape>(split_tree_leaf, reshape_const, false);
        outputs.push_back(reshape);
    }

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

ov::OutputVector OnlyFirstTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    {
        auto& split_tree_leaf = split_tree_leaves.front();
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    for (size_t leaf_idx = 1; leaf_idx < split_tree_leaves.size(); ++leaf_idx) {
        outputs.push_back(split_tree_leaves[leaf_idx]);
    }

    return outputs;
}

ov::OutputVector OnlyLastTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    {
        auto& split_tree_leaf = split_tree_leaves.back();
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    for (size_t leaf_idx = 0; leaf_idx < split_tree_leaves.size() - 1; ++leaf_idx) {
        outputs.push_back(split_tree_leaves[leaf_idx]);
    }

    return outputs;
}

ov::OutputVector OnlyMiddleTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    size_t middle_idx = split_tree_leaves.size() / 2;
    if (split_tree_leaves.size() % 2)
        ++middle_idx;
    for (size_t leaf_idx = 0; leaf_idx < split_tree_leaves.size() - 1; ++leaf_idx) {
        if (leaf_idx == middle_idx) {
            auto& split_tree_leaf = split_tree_leaves[leaf_idx];
            auto ng_order =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
            outputs.push_back(transpose);
        } else {
            outputs.push_back(split_tree_leaves[leaf_idx]);
        }
    }

    return outputs;
}

ov::OutputVector FirstAnotherTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    {
        auto& split_tree_leaf = split_tree_leaves.front();
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    for (size_t leaf_idx = 1; leaf_idx < split_tree_leaves.size(); ++leaf_idx) {
        auto& split_tree_leaf = split_tree_leaves[leaf_idx];
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    return outputs;
}

ov::OutputVector LastAnotherTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    {
        auto& split_tree_leaf = split_tree_leaves.back();
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    for (size_t leaf_idx = 0; leaf_idx < split_tree_leaves.size() - 1; ++leaf_idx) {
        auto& split_tree_leaf = split_tree_leaves[leaf_idx];
        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    return outputs;
}

ov::OutputVector MiddleAnotherTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    size_t middle_idx = split_tree_leaves.size() / 2;
    if (split_tree_leaves.size() % 2)
        ++middle_idx;
    for (size_t leaf_idx = 0; leaf_idx < split_tree_leaves.size(); ++leaf_idx) {
        auto& split_tree_leaf = split_tree_leaves[leaf_idx];
        if (leaf_idx == middle_idx) {
            auto ng_order =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
            outputs.push_back(transpose);
        } else {
            auto ng_order =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<ov::opset9::Transpose>(split_tree_leaf, ng_order);
            outputs.push_back(transpose);
        }
    }

    return outputs;
}

}  // namespace restrictions
}  // namespace backward
}  // namespace split

namespace {

std::vector<TransposeInsertF> insertTransposeFactories = {split::backward::restrictions::OnlyFirstTranspose,
                                                          split::backward::restrictions::OnlyLastTranspose,
                                                          split::backward::restrictions::OnlyMiddleTranspose,
                                                          split::backward::restrictions::FirstAnotherTranspose,
                                                          split::backward::restrictions::LastAnotherTranspose,
                                                          split::backward::restrictions::MiddleAnotherTranspose};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitBackwardRestrictTestSuite,
    TransposeSinkingSplitBackwardRestrictTestFixture,
    ::testing::Combine(::testing::Values(CreatePassFactory<ov::pass::TransposeSinkingSplitBackward>()),
                       ::testing::Values(1),
                       ::testing::Values(5),
                       ::testing::Values(split::backward::restrictions::CreateFunction),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(insertTransposeFactories)));
