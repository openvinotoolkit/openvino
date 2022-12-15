// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_general.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

using namespace testing;

using NodePtr = std::shared_ptr<ov::Node>;

TEST_F(TransformationTestsF, TransposeSinkingGeneralTestUnariesTransposesForward) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

            auto unary = std::make_shared<ov::opset9::Tanh>(transpose0);

            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<ov::opset9::Transpose>(unary, ng_order1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<ov::opset9::Tanh>(in_op);
        }

        function_ref = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    manager.register_pass<ov::pass::TransposeSinkingGeneralForward>();
}

TEST_F(TransformationTestsF, TransposeSinkingGeneralTestUnariesTransposesBackward) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

            auto unary = std::make_shared<ov::opset9::Tanh>(transpose0);

            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<ov::opset9::Transpose>(unary, ng_order1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<ov::opset9::Tanh>(in_op);
        }

        function_ref = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    manager.register_pass<ov::pass::TransposeSinkingGeneralBackward>();
}

TEST_F(TransformationTestsF, TransposeSinkingGeneralTestUnariesTransposesGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_unary_ops = 10;

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            auto ng_order0 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

            auto unary = std::make_shared<ov::opset9::Tanh>(transpose0);

            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
            in_op = std::make_shared<ov::opset9::Transpose>(unary, ng_order1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_unary_ops; ++i) {
            in_op = std::make_shared<ov::opset9::Tanh>(in_op);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        function_ref = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    manager.register_pass<ov::pass::TransposeSinkingGeneral>();
}

TEST_F(TransformationTestsF, TransposeSinkingGeneralTestBinaryGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    size_t num_binary_ops = 10;

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        NodePtr in_op = transpose0;
        for (size_t i = 0; i < num_binary_ops; ++i) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
            auto ng_order1 =
                std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
            auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

            in_op = std::make_shared<ov::opset9::Add>(in_op, transpose1);
        }

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_binary_ops; ++i) {
            auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
            in_op = std::make_shared<ov::opset9::Add>(in_op, in_constant);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        function_ref = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    manager.register_pass<ov::pass::TransposeSinkingGeneral>();
}

TEST_F(TransformationTestsF, TransposeSinkingGeneralTestConcatGeneral) {
    ov::Shape input_shape = {1, 96, 55, 55};
    ov::element::Type input_type = ov::element::f32;
    const size_t num_concat_ops = 3;
    const size_t num_concat_inputs = 2;

    {
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

        function = std::make_shared<ov::Model>(in_op, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        NodePtr in_op = X;
        for (size_t i = 0; i < num_concat_ops; ++i) {
            ov::OutputVector concat_inputs;

            concat_inputs.push_back(in_op);

            for (size_t j = 1; j < num_concat_inputs; ++j) {
                auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
                concat_inputs.push_back(in_constant);
            }
            in_op = std::make_shared<ov::opset9::Concat>(concat_inputs, 2);
        }

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

        function_ref = std::make_shared<ov::Model>(transpose0, ov::ParameterVector{X});
    }

    manager.register_pass<ov::pass::TransposeSinkingGeneral>();
}

// ----------------------------------------------------------------------------------------------------------------------

class IFactory {
public:
    virtual ~IFactory() = default;
    virtual NodePtr create(const ov::OutputVector& parent) = 0;

    virtual size_t getNumInputs() const = 0;
    virtual size_t getNumOuputs() const = 0;
};

using FactoryPtr = std::shared_ptr<IFactory>;

class UnaryFactory : public IFactory {
public:
    NodePtr create(const ov::OutputVector& parent) override {
        return std::make_shared<ov::opset9::Sinh>(parent.front());
    }

    static FactoryPtr createFactory() {
        return std::make_shared<UnaryFactory>();
    }

    size_t getNumInputs() const override {
        return 1;
    }
    size_t getNumOuputs() const override {
        return 1;
    }
};

class BinaryFactory : public IFactory {
public:
    NodePtr create(const ov::OutputVector& parent) override {
        return std::make_shared<ov::opset9::Add>(parent[0], parent[1]);
    }

    static FactoryPtr createFactory() {
        return std::make_shared<BinaryFactory>();
    }

    size_t getNumInputs() const override {
        return 2;
    }
    size_t getNumOuputs() const override {
        return 1;
    }
};

class SplitFactory : public IFactory {
public:
    SplitFactory(size_t axis) : axis_(axis) {}
    NodePtr create(const ov::OutputVector& parent) override {
        auto split_axis_const = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{}, axis_);
        return std::make_shared<ov::opset9::Split>(parent.front(), split_axis_const, 2);
    }

    static FactoryPtr createFactory(size_t axis) {
        return std::make_shared<SplitFactory>(axis);
    }

    size_t getNumInputs() const override {
        return 1;
    }
    size_t getNumOuputs() const override {
        return 2;
    }

private:
    const size_t axis_;
};

class ConcatFactory : public IFactory {
public:
    ConcatFactory(size_t axis) : axis_(axis) {}
    NodePtr create(const ov::OutputVector& parent) override {
        return std::make_shared<ov::opset9::Concat>(parent, axis_);
    }

    static FactoryPtr createFactory(size_t axis) {
        return std::make_shared<ConcatFactory>(axis);
    }

    size_t getNumInputs() const override {
        return 2;
    }
    size_t getNumOuputs() const override {
        return 1;
    }

private:
    const size_t axis_;
};

/*
    Each node pair should be started with input size = 1 node and finished with node output size = 1
    Insert Split/Concat to fullfill that.
*/
NodePtr CreateNodePair(FactoryPtr factory_first,
                       FactoryPtr factory_second,
                       NodePtr parent,
                       size_t split_axis,
                       size_t concat_axis) {
    NodePtr input = parent;
    if (factory_first->getNumInputs() != 1) {
        input = SplitFactory(split_axis).create(input->outputs());
    }

    input = factory_first->create(input->outputs());
    if (factory_first->getNumOuputs() < factory_second->getNumInputs()) {
        input = SplitFactory(split_axis).create(input->outputs());
    } else if (factory_first->getNumOuputs() > factory_second->getNumInputs()) {
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
    for (int i = 0; i < factories.size(); ++i) {
        for (int j = 0; j < factories.size(); ++j) {
            in_op = CreateNodePair(factories[i], factories[j], in_op, split_axis, concat_axis);
        }
    }

    return in_op;
}

TEST_F(TransformationTestsF, TransposeSinkingGeneralTestMultipleTypes) {
    ov::Shape input_shape = {1, 96, 40, 55};
    ov::element::Type input_type = ov::element::f32;

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto node0 = MakeAllNodesSubgraph(X, 1, 1);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(node0, ng_order0);

        auto reshape_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{1, 40, 55, 96});
        auto reshape = std::make_shared<ov::opset9::Reshape>(transpose0, reshape_const, false);

        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(reshape, ng_order1);

        auto node1 = MakeAllNodesSubgraph(transpose1, 1, 1);

        function = std::make_shared<ov::Model>(node1, ov::ParameterVector{X});
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        auto node0 = MakeAllNodesSubgraph(transpose0, 3, 3);

        auto reshape_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{1, 40, 55, 96});
        auto reshape = std::make_shared<ov::opset9::Reshape>(node0, reshape_const, false);

        auto node1 = MakeAllNodesSubgraph(reshape, 3, 3);

        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(node1, ng_order1);

        function_ref = std::make_shared<ov::Model>(transpose1, ov::ParameterVector{X});
    }

    manager.register_pass<ov::pass::TransposeSinkingGeneral>();
}
