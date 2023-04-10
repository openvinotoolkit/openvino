// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_general.hpp"

#include <functional>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using NodePtr = std::shared_ptr<ov::Node>;

namespace transpose_sinking {
namespace testing {
namespace general {

TEST_F(TransformationTestsF, TSGeneralTestUnariesTransposesForward) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

            auto unary = std::make_shared<Tanh>(transpose0);

            auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<Transpose>(unary, ng_order1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<Tanh>(in_op);
        }

        function_ref = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    manager.register_pass<TSGeneralForward>();
}

TEST_F(TransformationTestsF, TSGeneralTestUnariesTransposesBackward) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

            auto unary = std::make_shared<Tanh>(transpose0);

            auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<Transpose>(unary, ng_order1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<Tanh>(in_op);
        }

        function_ref = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }
    manager.register_pass<TSGeneralBackward>();
}

TEST_F(TransformationTestsF, TSGeneralTestUnariesTransposesGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

            auto unary = std::make_shared<Tanh>(transpose0);

            auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<Transpose>(unary, ng_order1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<Tanh>(in_op);
        }

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

        function_ref = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    manager.register_pass<TSGeneral>();
}

TEST_F(TransformationTestsF, TSGeneralTestBinaryGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_binary_ops = 10;

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_binary_ops; ++i) {
            auto in_constant = std::make_shared<Constant>(input_type, input_shape, ov::Shape{1});
            auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);

            in_op = std::make_shared<Add>(in_op, transpose1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_binary_ops; ++i) {
            auto in_constant = std::make_shared<Constant>(input_type, input_shape, ov::Shape{1});
            in_op = std::make_shared<Add>(in_op, in_constant);
        }

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

        function_ref = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    manager.register_pass<TSGeneral>();
}

TEST_F(TransformationTestsF, TSGeneralTestConcatGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    const size_t num_concat_ops = 3;
    const size_t num_concat_inputs = 2;

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;
            concat_inputs.push_back(in_op);
            for (size_t j = 1; j < num_concat_inputs; ++j) {
                auto in_constant = std::make_shared<Constant>(input_type, input_shape, ov::Shape{1});
                auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
                auto transpose1 = std::make_shared<Transpose>(in_constant, ng_order1);
                concat_inputs.push_back(transpose1);
            }
            in_op = std::make_shared<Concat>(concat_inputs, 1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;

            concat_inputs.push_back(in_op);

            for (size_t j = 1; j < num_concat_inputs; ++j) {
                auto in_constant = std::make_shared<Constant>(input_type, input_shape, ov::Shape{1});
                concat_inputs.push_back(in_constant);
            }
            in_op = std::make_shared<Concat>(concat_inputs, 2);
        }

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);

        function_ref = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    manager.register_pass<TSGeneral>();
}

// ----------------------------------------------------------------------------------------------------------------------

class IFactory {
public:
    virtual ~IFactory() = default;
    virtual NodePtr create(const ov::OutputVector& parent) = 0;

    virtual size_t getNumInputs() const = 0;
    virtual size_t getNumOutputs() const = 0;
};

using FactoryPtr = std::shared_ptr<IFactory>;

class UnaryFactory : public IFactory {
public:
    NodePtr create(const ov::OutputVector& parent) override {
        return std::make_shared<Sinh>(parent.front());
    }

    static FactoryPtr createFactory() {
        return std::make_shared<UnaryFactory>();
    }

    size_t getNumInputs() const override {
        return 1;
    }
    size_t getNumOutputs() const override {
        return 1;
    }
};

class BinaryFactory : public IFactory {
public:
    NodePtr create(const ov::OutputVector& parent) override {
        return std::make_shared<Add>(parent[0], parent[1]);
    }

    static FactoryPtr createFactory() {
        return std::make_shared<BinaryFactory>();
    }

    size_t getNumInputs() const override {
        return 2;
    }
    size_t getNumOutputs() const override {
        return 1;
    }
};

class SplitFactory : public IFactory {
public:
    explicit SplitFactory(size_t axis) : axis_(axis) {}
    NodePtr create(const ov::OutputVector& parent) override {
        auto split_axis_const = std::make_shared<Constant>(ov::element::u64, ov::Shape{}, axis_);
        return std::make_shared<Split>(parent.front(), split_axis_const, 2);
    }

    static FactoryPtr createFactory(size_t axis) {
        return std::make_shared<SplitFactory>(axis);
    }

    size_t getNumInputs() const override {
        return 1;
    }
    size_t getNumOutputs() const override {
        return 2;
    }

private:
    const size_t axis_;
};

class ConcatFactory : public IFactory {
public:
    explicit ConcatFactory(size_t axis) : axis_(axis) {}
    NodePtr create(const ov::OutputVector& parent) override {
        return std::make_shared<Concat>(parent, axis_);
    }

    static FactoryPtr createFactory(size_t axis) {
        return std::make_shared<ConcatFactory>(axis);
    }

    size_t getNumInputs() const override {
        return 2;
    }
    size_t getNumOutputs() const override {
        return 1;
    }

private:
    const size_t axis_;
};

/*
    Each node pair should be started with input size = 1 node and finished with node output size = 1
    Insert Split/Concat to fullfill that.
*/
NodePtr CreateNodePair(const FactoryPtr& factory_first,
                       const FactoryPtr& factory_second,
                       NodePtr parent,
                       size_t split_axis,
                       size_t concat_axis) {
    NodePtr input = parent;
    if (factory_first->getNumInputs() != 1) {
        input = SplitFactory(split_axis).create(input->outputs());
    }

    input = factory_first->create(input->outputs());
    if (factory_first->getNumOutputs() < factory_second->getNumInputs()) {
        input = SplitFactory(split_axis).create(input->outputs());
    } else if (factory_first->getNumOutputs() > factory_second->getNumInputs()) {
        input = ConcatFactory(concat_axis).create(input->outputs());
    }

    auto output = factory_second->create(input->outputs());
    if (output->get_output_size() > 1) {
        output = ConcatFactory(concat_axis).create(output->outputs());
    }

    return output;
}

NodePtr MakeAllNodesSubgraph(NodePtr parent, size_t split_axis, size_t concat_axis) {
    std::vector<FactoryPtr> factories = {UnaryFactory::createFactory(),
                                         SplitFactory::createFactory(split_axis),
                                         ConcatFactory::createFactory(concat_axis)};
    NodePtr in_op = parent;
    for (size_t i = 0; i < factories.size(); ++i) {
        for (size_t j = 0; j < factories.size(); ++j) {
            in_op = CreateNodePair(factories[i], factories[j], in_op, split_axis, concat_axis);
        }
    }

    return in_op;
}

TEST_F(TransformationTestsF, TSGeneralTestMultipleTypes) {
    using namespace transpose_sinking::testing::general;
    ov::Shape input_shape = {1, 96, 40, 55};
    ov::element::Type input_type = ov::element::f32;

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto node0 = MakeAllNodesSubgraph(X, 1, 1);

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(node0, ng_order0);

        auto reshape_const = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{1, 40, 55, 96});
        auto reshape = std::make_shared<Reshape>(transpose0, reshape_const, false);

        auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<Transpose>(reshape, ng_order1);

        auto node1 = MakeAllNodesSubgraph(transpose1, 1, 1);

        function = std::make_shared<ov::Model>(node1, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

        auto node0 = MakeAllNodesSubgraph(transpose0, 3, 3);

        auto reshape_const = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{1, 40, 55, 96});
        auto reshape = std::make_shared<Reshape>(node0, reshape_const, false);

        auto node1 = MakeAllNodesSubgraph(reshape, 3, 3);

        auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<Transpose>(node1, ng_order1);

        function_ref = std::make_shared<ov::Model>(transpose1, ov::ParameterVector{X});
    }

    manager.register_pass<TSGeneral>();
}

TEST_F(TransformationTestsF, TSGeneralCheckShapeOfConstFoldingDisabled) {
    using namespace transpose_sinking::testing::general;
    ov::Shape input_shape = {96, 40, 55};
    ov::Shape reshape_shape = {1, 96, 40, 55};
    ov::element::Type input_type = ov::element::f32;
    {
        auto X = std::make_shared<Parameter>(input_type, input_shape);
        auto Shape = std::make_shared<Parameter>(input_type, reshape_shape);

        auto order = std::make_shared<Constant>(ov::element::u64, ov::Shape{3}, ov::Shape{0, 2, 1});
        auto transpose = std::make_shared<Transpose>(X, order);

        auto shape_of = std::make_shared<ShapeOf>(Shape);
        auto reshape = std::make_shared<Reshape>(transpose, shape_of, false);

        auto ng_order1 = std::make_shared<Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<Transpose>(reshape, ng_order1);

        function = std::make_shared<ov::Model>(transpose1, ov::ParameterVector{X, Shape});
    }
    manager.register_pass<TSGeneral>();
}

}  // namespace general
}  // namespace testing
}  // namespace transpose_sinking