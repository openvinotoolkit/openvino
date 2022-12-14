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

using namespace ov;
using namespace ov::opset9;

namespace transpose_sinking_split {

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

std::vector<size_t> split_tree_depth_nums = {1, 3};
std::vector<size_t> split_operations_numbers = {1, 10};
std::vector<size_t> split_outputs_numbers = {2, 3};

namespace forward {

namespace single_consumer {

std::shared_ptr<Model> CreateFunction(size_t num_split_ops,
                                      size_t num_split_outputs,
                                      element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

        ov::OutputVector outputs;
        ov::Output<ov::Node> in_op = transpose0->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 2);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                outputs.push_back(split->output(num_output));
            }
            in_op = split->output(num_split_outputs - 1);
        }
        outputs.push_back(in_op);

        return std::make_shared<ov::Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_split_ops,
                                               size_t num_split_outputs,
                                               element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

        ov::OutputVector outputs;
        ov::Output<ov::Node> in_op = X->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 1);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
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

} // namespace single_consumer

namespace mult_consumers {

namespace input_node_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_split_ops,
                                      size_t num_split_outputs,
                                      element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto tanh = std::make_shared<Tanh>(X);

        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<Transpose>(tanh, ng_order0);

        ov::OutputVector outputs;
        auto in_op = transpose0->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 2);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                outputs.push_back(split->output(num_output));
            }
            in_op = split->output(num_split_outputs - 1);
        }
        outputs.push_back(in_op);

        auto tanh1 = std::make_shared<Tanh>(tanh);
        outputs.push_back(tanh1);

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_split_ops,
                                               size_t num_split_outputs,
                                               element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto tanh = std::make_shared<Tanh>(X);

        ov::OutputVector outputs;
        auto in_op = tanh->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 1);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
                auto transpose0 = std::make_shared<Transpose>(split->output(num_output), ng_order0);
                outputs.push_back(transpose0);
            }
            in_op = split->output(num_split_outputs - 1);
        }

        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);
        outputs.push_back(transpose0);

        auto tanh1 = std::make_shared<Tanh>(tanh);
        outputs.push_back(tanh1);

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

} // namespace input_node_consumers

namespace input_transpose_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_split_ops,
                                      size_t num_split_outputs,
                                      element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);
        
        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

        ov::OutputVector outputs;
        auto in_op = transpose0->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 2);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                outputs.push_back(split->output(num_output));
            }
            in_op = split->output(num_split_outputs - 1);
        }
        outputs.push_back(in_op);

        auto tanh = std::make_shared<Tanh>(transpose0);
        outputs.push_back(tanh);

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_split_ops,
                                               size_t num_split_outputs,
                                               element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        ov::OutputVector outputs;
        auto in_op = X->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 1);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
                auto transpose0 = std::make_shared<Transpose>(split->output(num_output), ng_order0);
                outputs.push_back(transpose0);
            }
            in_op = split->output(num_split_outputs - 1);
        }

        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);
        outputs.push_back(transpose0);

        auto ng_order1 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose1 = std::make_shared<Transpose>(X, ng_order1);

        auto tanh = std::make_shared<Tanh>(transpose1);
        outputs.push_back(tanh);

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

} // namespace input_transpose_consumers

namespace output_consumers {

std::shared_ptr<Model> CreateFunction(size_t num_split_ops,
                                      size_t num_split_outputs,
                                      element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);
        
        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<Transpose>(X, ng_order0);

        ov::OutputVector outputs;
        auto in_op = transpose0->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 2);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                outputs.push_back(split->output(num_output));
            }
            in_op = split->output(num_split_outputs - 1);
        }
        outputs.push_back(in_op);

        auto tanh = std::make_shared<Tanh>(in_op);
        auto tanh1 = std::make_shared<Tanh>(in_op);
        outputs.push_back(tanh);
        outputs.push_back(tanh1);

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t num_split_ops,
                                               size_t num_split_outputs,
                                               element::Type input_type) {
        const Shape input_shape{96, static_cast<size_t>(std::pow(num_split_outputs, num_split_ops + 1)), 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        ov::OutputVector outputs;
        auto in_op = X->output(0);
        for (size_t i = 0; i < num_split_ops; ++i) {
            auto split_axis_const = std::make_shared<Constant>(element::u64, Shape{}, 1);
            auto split = std::make_shared<Split>(in_op, split_axis_const, num_split_outputs);
            for (size_t num_output = 0; num_output < num_split_outputs - 1; ++num_output) {
                auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
                auto transpose0 = std::make_shared<Transpose>(split->output(num_output), ng_order0);
                outputs.push_back(transpose0);
            }
            in_op = split->output(num_split_outputs - 1);
        }

        auto ng_order0 = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose0 = std::make_shared<Transpose>(in_op, ng_order0);
        outputs.push_back(transpose0);

        auto tanh = std::make_shared<Tanh>(transpose0);
        auto tanh1 = std::make_shared<Tanh>(transpose0);
        outputs.push_back(tanh);
        outputs.push_back(tanh1);

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

} // namespace output_consumers

} // namespace mult_consumers

} // namespace forward

namespace backward {

class SplitFactory {
public:
    SplitFactory(size_t axis, size_t n_outputs, element::Type elem_type) :
        _axis(axis), _n_outputs(n_outputs), _elem_type(elem_type) {}
    NodePtr create(ov::Output<ov::Node> parent) const {
        auto split_axis_const = std::make_shared<Constant>(_elem_type, Shape{}, _axis);
        return std::make_shared<Split>(parent, split_axis_const, _n_outputs);
    }

private:
    const size_t _axis;
    const size_t _n_outputs;
    const element::Type _elem_type;
};

void CreateSplitTree(size_t max_depth,
                     size_t depth,
                     ov::Output<ov::Node> parent,
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

std::shared_ptr<Model> CreateFunction(size_t split_tree_depth,
                                      size_t num_split_outputs,
                                      element::Type input_type) {
        const size_t split_input_dim_value = static_cast<size_t>(std::pow(num_split_outputs, split_tree_depth + 1));
        const Shape input_shape{96, split_input_dim_value, 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        ov::OutputVector split_tree_leaves;
        {
            SplitFactory split_factory(/* axis */ 1, num_split_outputs, /* elem_type */ element::u64);
            CreateSplitTree(split_tree_depth, /* depth */ 0, X->output(0), split_factory, split_tree_leaves);
        }

        ov::OutputVector outputs;
        for (auto& split_tree_leaf : split_tree_leaves) {
            auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);

            const size_t split_dim_current_value = static_cast<size_t>(split_input_dim_value / std::pow(num_split_outputs, split_tree_depth));
            auto reshape_const = std::make_shared<Constant>(element::u64, Shape{3}, Shape{96, 55, split_dim_current_value * 55});
            auto reshape = std::make_shared<Reshape>(transpose, reshape_const, false);
            outputs.push_back(reshape);
        }

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

std::shared_ptr<Model> CreateReferenceFunction(size_t split_tree_depth,
                                               size_t num_split_outputs,
                                               element::Type input_type) {
        const size_t split_input_dim_value = static_cast<size_t>(std::pow(num_split_outputs, split_tree_depth + 1));
        const Shape input_shape{96, split_input_dim_value, 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<Transpose>(X, ng_order);

        ov::OutputVector split_tree_leaves;
        {
            SplitFactory split_factory(/* axis */ 2, num_split_outputs, /* elem_type */ element::u64);
            CreateSplitTree(split_tree_depth, /* depth */ 0, transpose->output(0), split_factory, split_tree_leaves);
        }

        ov::OutputVector outputs;
        for (auto& split_tree_leaf : split_tree_leaves) {
            const size_t split_dim_current_value = static_cast<size_t>(split_input_dim_value / std::pow(num_split_outputs, split_tree_depth));
            auto reshape_const = std::make_shared<Constant>(element::u64, Shape{3}, Shape{96, 55, split_dim_current_value * 55});
            auto reshape = std::make_shared<Reshape>(split_tree_leaf, reshape_const, false);
            outputs.push_back(reshape);
        }

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

} // namespace backward

using CreateGraphSplitF = std::function< std::shared_ptr<Model> (size_t num_split_ops,
                                                                 size_t num_split_outputs,
                                                                 element::Type input_type)>;

using TestSplitParams = std::tuple<PassFactoryPtr,
                              size_t, /* num_split_ops */
                              size_t, /* num_split_outputs */
                              CreateGraphSplitF, /* model_factory */
                              CreateGraphSplitF, /* reference_model_factory */
                              element::Type> /* input type */;

class TransposeSinkingSplitTestFixture: public ::testing::WithParamInterface<TestSplitParams>,
                                        public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestSplitParams>& obj) {
        PassFactoryPtr pass_factory;
        size_t num_split_ops;
        size_t num_split_outputs;
        CreateGraphSplitF model_factory;
        CreateGraphSplitF reference_model_factory;
        element::Type input_type;

        std::tie(pass_factory,
             num_split_ops,
             num_split_outputs,
             model_factory,
             reference_model_factory,
             input_type) = obj.param;

        std::ostringstream test_name;
        test_name << "pass_factory=" << pass_factory->getTypeName() << "_";
        test_name << "num_split_ops=" << num_split_ops << "_";
        test_name << "num_split_outputs=" << num_split_outputs << "_";
        test_name << "input_type=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingSplitTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_split_ops;
    size_t num_split_outputs;
    CreateGraphSplitF model_factory;
    CreateGraphSplitF reference_model_factory;
    element::Type input_type;
    std::tie(pass_factory,
             num_split_ops,
             num_split_outputs,
             model_factory,
             reference_model_factory,
             input_type) = this->GetParam();

    model = model_factory(num_split_ops, num_split_outputs, input_type);
    model_ref = reference_model_factory(num_split_ops, num_split_outputs, input_type);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitForwardSingleConsumerTestSuite,
    TransposeSinkingSplitTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingSplitForward)),
                       ::testing::ValuesIn(split_operations_numbers),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(forward::single_consumer::CreateFunction),
                       ::testing::Values(forward::single_consumer::CreateReferenceFunction),
                       ::testing::Values(element::f32)),
                       TransposeSinkingSplitTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitForwardMultInputNodeConsumersTestSuite,
    TransposeSinkingSplitTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingSplitForward)),
                       ::testing::ValuesIn(split_operations_numbers),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(forward::mult_consumers::input_node_consumers::CreateFunction),
                       ::testing::Values(forward::mult_consumers::input_node_consumers::CreateReferenceFunction),
                       ::testing::Values(element::f32)),
                       TransposeSinkingSplitTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitForwardMultInputTransposeConsumersTestSuite,
    TransposeSinkingSplitTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingSplitForward)),
                       ::testing::ValuesIn(split_operations_numbers),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(forward::mult_consumers::input_transpose_consumers::CreateFunction),
                       ::testing::Values(forward::mult_consumers::input_transpose_consumers::CreateReferenceFunction),
                       ::testing::Values(element::f32)),
                       TransposeSinkingSplitTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitForwardMultOutputConsumersTestSuite,
    TransposeSinkingSplitTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingSplitForward)),
                       ::testing::ValuesIn(split_operations_numbers),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(forward::mult_consumers::output_consumers::CreateFunction),
                       ::testing::Values(forward::mult_consumers::output_consumers::CreateReferenceFunction),
                       ::testing::Values(element::f32)),
                       TransposeSinkingSplitTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitBackwardTestSuite,
    TransposeSinkingSplitTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingSplitBackward)),
                       ::testing::ValuesIn(split_tree_depth_nums),
                       ::testing::ValuesIn(split_outputs_numbers),
                       ::testing::Values(backward::CreateFunction),
                       ::testing::Values(backward::CreateReferenceFunction),
                       ::testing::Values(element::f32)),
                       TransposeSinkingSplitTestFixture::get_test_name);

namespace backward {
namespace restrictions {

using TransposeInsertF = std::function< ov::OutputVector (const ov::OutputVector& split_tree_leaves)>;

std::shared_ptr<Model> CreateFunction(size_t split_tree_depth,
                                          size_t num_split_outputs,
                                          element::Type input_type,
                                          TransposeInsertF transpose_insert_func) {
        const size_t split_input_dim_value = static_cast<size_t>(std::pow(num_split_outputs, split_tree_depth + 1));
        const Shape input_shape{96, split_input_dim_value, 55, 55};

        auto X = std::make_shared<Parameter>(input_type, input_shape);

        ov::OutputVector split_tree_leaves;
        {
            SplitFactory split_factory(/* axis */ 1, num_split_outputs, /* elem_type */ element::u64);
            CreateSplitTree(split_tree_depth, /* depth */ 0, X->output(0), split_factory, split_tree_leaves);
        }

        ov::OutputVector outputs;
        for (auto& split_tree_leaf : transpose_insert_func(split_tree_leaves)) {
            const size_t split_dim_current_value = static_cast<size_t>(split_input_dim_value / std::pow(num_split_outputs, split_tree_depth));
            auto reshape_const = std::make_shared<Constant>(element::u64, Shape{3}, Shape{96, 55, split_dim_current_value * 55});
            auto reshape = std::make_shared<Reshape>(split_tree_leaf, reshape_const, false);
            outputs.push_back(reshape);
        }

        return std::make_shared<Model>(outputs, ov::ParameterVector{X});
}

ov::OutputVector OnlyFirstTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    {
        auto& split_tree_leaf = split_tree_leaves.front();
        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
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
        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
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
            auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
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
        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    for (size_t leaf_idx = 1; leaf_idx < split_tree_leaves.size(); ++leaf_idx) {
        auto& split_tree_leaf = split_tree_leaves[leaf_idx];
        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    return outputs;
}

ov::OutputVector LastAnotherTranspose(const ov::OutputVector& split_tree_leaves) {
    ov::OutputVector outputs;
    {
        auto& split_tree_leaf = split_tree_leaves.back();
        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
        outputs.push_back(transpose);
    }

    for (size_t leaf_idx = 0; leaf_idx < split_tree_leaves.size() - 1; ++leaf_idx) {
        auto& split_tree_leaf = split_tree_leaves[leaf_idx];
        auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
        auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
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
            auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 2, 3, 1});
            auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
            outputs.push_back(transpose);
        } else {
            auto ng_order = std::make_shared<Constant>(element::u64, Shape{4}, Shape{0, 3, 1, 2});
            auto transpose = std::make_shared<Transpose>(split_tree_leaf, ng_order);
            outputs.push_back(transpose);
        }
    }

    return outputs;
}

struct TransposeInsertFuncDesc {
    TransposeInsertFuncDesc() = default;
    TransposeInsertFuncDesc(TransposeInsertF a_function,
                            std::string a_name) :
                            function(a_function),
                            name(a_name) {}

    TransposeInsertF function;
    std::string name;
};

using CreateGraphSplitBackwardRestrictF = std::function< std::shared_ptr<Model> (size_t split_tree_depth,
                                                                                     size_t num_split_outputs,
                                                                                     element::Type input_type,
                                                                                     TransposeInsertF tranpose_insert_function)>;

using TestSplitBackwardRestrictParams = std::tuple<PassFactoryPtr,
                              size_t, /* split_tree_depth */
                              size_t, /* num_split_outputs */
                              CreateGraphSplitBackwardRestrictF, /* model_factory */
                              element::Type, /* input type */
                              TransposeInsertFuncDesc>; /* insert transpose function */

class TransposeSinkingSplitBackwardRestrictTestFixture: public ::testing::WithParamInterface<TestSplitBackwardRestrictParams>,
                                        public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestSplitBackwardRestrictParams>& obj) {
        PassFactoryPtr pass_factory;
        size_t split_tree_depth;
        size_t num_split_outputs;
        CreateGraphSplitBackwardRestrictF model_factory;
        element::Type input_type;
        TransposeInsertFuncDesc tranpose_insert_function;

        std::tie(pass_factory,
             split_tree_depth,
             num_split_outputs,
             model_factory,
             input_type,
             tranpose_insert_function) = obj.param;

        std::ostringstream test_name;
        test_name << "pass_factory=" << pass_factory->getTypeName() << "_";
        test_name << "split_tree_depth=" << split_tree_depth << "_";
        test_name << "num_split_outputs=" << num_split_outputs << "_";
        test_name << "tranpose_insert_function=" << tranpose_insert_function.name << "_";
        test_name << "input_type=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingSplitBackwardRestrictTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t split_tree_depth;
    size_t num_split_outputs;
    CreateGraphSplitBackwardRestrictF model_factory;
    element::Type input_type;
    TransposeInsertFuncDesc tranpose_insert_function;

    std::tie(pass_factory,
             split_tree_depth,
             num_split_outputs,
             model_factory,
             input_type,
             tranpose_insert_function) = this->GetParam();

    model = model_factory(split_tree_depth, num_split_outputs, input_type, tranpose_insert_function.function);
    model_ref = model->clone();
    pass_factory->registerPass(manager);
}

#define FUNC(name) TransposeInsertFuncDesc(backward::restrictions::name, #name)

std::vector<TransposeInsertFuncDesc> insertTransposeFactories = {
    FUNC(OnlyFirstTranspose),
    FUNC(OnlyLastTranspose),
    FUNC(OnlyMiddleTranspose),
    FUNC(FirstAnotherTranspose),
    FUNC(LastAnotherTranspose),
    FUNC(MiddleAnotherTranspose)
};

#undef FUNC

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSplitBackwardRestrictTestSuite,
    TransposeSinkingSplitBackwardRestrictTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingSplitBackward)),
                       ::testing::Values(1),
                       ::testing::Values(5),
                       ::testing::Values(backward::restrictions::CreateFunction),
                       ::testing::Values(element::f32),
                       ::testing::ValuesIn(insertTransposeFactories)),
                       TransposeSinkingSplitBackwardRestrictTestFixture::get_test_name);

} // namespace restrictions
} // namespace backward

} // namespace transpose_sinking_split
