// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_pad.hpp>
#include <transformations/common_optimizations/transpose_sinking_utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace ov;
using namespace opset10;

namespace transpose_sinking_pad {
namespace {

using NodePtr = shared_ptr<Node>;
using ModelPtr = shared_ptr<Model>;

class IPassFactory {
public:
    IPassFactory(const string& type_name) : type_name_(type_name) {}

    virtual ~IPassFactory() = default;

    virtual void registerPass(pass::Manager& pass_manager) const = 0;

    const string& getTypeName() const {
        return type_name_;
    }

private:
    const string type_name_;
};

using PassFactoryPtr = shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    PassFactory(const string& type_name) : IPassFactory(type_name) {}

    void registerPass(pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
        pass_manager.register_pass<ov::pass::ConstantFolding>();
    }
};

#define CREATE_PASS_FACTORY(pass_name) make_shared<PassFactory<pass::pass_name>>(#pass_name)

vector<int64_t> TransposePadValues(const vector<int64_t>& pads, const vector<size_t>& order) {
    vector<int64_t> new_pads(pads.size());
    for (size_t i = 0; i < pads.size(); ++i) {
        new_pads[i] = pads[order[i]];
    }
    return new_pads;
};
}  // namespace

namespace forward {
namespace single_consumer {

shared_ptr<Model> CreateFunction(size_t num_pad_ops, element::Type input_type) {
    const Shape input_shape{96, 32, 55, 55};

    auto X = make_shared<Parameter>(input_type, input_shape);

    auto order = make_shared<Constant>(element::i64, Shape{4}, Shape{0, 3, 1, 2});
    auto transpose = make_shared<Transpose>(X, order); // 96 55 32 55

    OutputVector outputs;
    Output<Node> in_op = transpose->output(0);
    auto pad_value = make_shared<Constant>(input_type, Shape{}, 0);
    for (size_t i = 0; i < num_pad_ops; ++i) {
        auto pad_begin_const = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{95, 54, 31, 53});
        auto pad_end_const = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{95, 54, 31, 53});
        auto pad = make_shared<Pad>(in_op, pad_begin_const, pad_end_const, pad_value, ov::op::PadMode::REFLECT);
        outputs.push_back((pad->output(0)));
        in_op = pad;
    }
    outputs.push_back(in_op);

    return make_shared<Model>(outputs, ParameterVector{X});
}

shared_ptr<Model> CreateReferenceFunction(size_t num_pad_ops, element::Type input_type) {
    const Shape input_shape{96, 32, 55, 55};

    auto X = make_shared<Parameter>(input_type, input_shape);

    OutputVector outputs;
    Output<Node> in_op = X->output(0);
    vector<int64_t> pads{95, 54, 31, 53};
    auto transpose_pad_values = [&](const vector<size_t>& order) {
        vector<int64_t> new_pads(pads.size());
        for (size_t i = 0; i < pads.size(); ++i) {
            new_pads[i] = pads[order[i]];
        }
        return new_pads;
    };
    auto axis = make_shared<Constant>(element::i64, Shape{}, 0);
    auto pad_value = make_shared<Constant>(input_type, Shape{}, 0);
    vector<size_t> order_val = {0, 3, 1, 2};
    for (size_t i = 0; i < num_pad_ops; ++i) {
        auto pad_begin_const = make_shared<Constant>(element::i64, Shape{4}, transpose_pad_values({0, 2, 3, 1}));
        auto pad_end_const = make_shared<Constant>(element::i64, Shape{4}, transpose_pad_values({0, 2, 3, 1}));
        auto pad = make_shared<Pad>(in_op, pad_begin_const, pad_end_const, pad_value, ov::op::PadMode::CONSTANT);

        auto order = make_shared<Constant>(element::i64, Shape{4}, Shape{order_val});
        auto transpose = make_shared<Transpose>(pad->output(0), order);
        outputs.push_back(transpose);
        in_op = pad;
    }

    auto order = make_shared<Constant>(element::i64, Shape{4}, order_val);
    auto transpose = make_shared<Transpose>(in_op, order);
    outputs.push_back(transpose);

    auto ref = make_shared<Model>(outputs, ParameterVector{X});
    ov::pass::Manager ps_manager;
    ps_manager.run_passes(ref);
    return ref;
}

}  // namespace single_consumer
}  // namespace forward

namespace backward {
namespace single_consumer {

shared_ptr<Model> CreateFunction(size_t num_pad_ops, element::Type input_type) {
    const Shape input_shape{96, 32, 55, 55};

    auto X = make_shared<Parameter>(input_type, input_shape);

    OutputVector outputs;
    Output<Node> in_op = X->output(0);
    auto pad_value = make_shared<Constant>(input_type, Shape{}, 0);
    for (size_t i = 0; i < num_pad_ops; ++i) {
        auto pad_begin_const = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 1, 2, 3});
        auto pad_end_const = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 1, 2, 3});
        auto pad = make_shared<Pad>(in_op, pad_begin_const, pad_end_const, pad_value, ov::op::PadMode::CONSTANT);
        in_op = pad;
    }
    auto order = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 2});
    auto transpose = make_shared<Transpose>(in_op, order);
    auto relu = make_shared<Relu>(transpose);
    outputs.push_back(relu);
    return make_shared<Model>(outputs, ParameterVector{X});
}

shared_ptr<Model> CreateReferenceFunction(size_t num_pad_ops, element::Type input_type) {
    const Shape input_shape{96, 32, 55, 55};

    auto X = make_shared<Parameter>(input_type, input_shape);
    vector<size_t> order_val = {0, 3, 1, 2};
    auto order = make_shared<Constant>(element::i64, Shape{4}, order_val);
    auto transpose = make_shared<Transpose>(X, order);

    OutputVector outputs;
    Output<Node> in_op = transpose->output(0);
    vector<int64_t> pads{0, 1, 2, 3};
    auto axis = make_shared<Constant>(element::i64, Shape{}, 0);
    auto pad_value = make_shared<Constant>(input_type, Shape{}, 0);
    for (size_t i = 0; i < num_pad_ops; ++i) {
        auto pad_begin_const = make_shared<Constant>(element::i64, Shape{4}, TransposePadValues(pads, order_val));
        auto pad_end_const = make_shared<Constant>(element::i64, Shape{4}, TransposePadValues(pads, order_val));
        auto pad = make_shared<Pad>(in_op, pad_begin_const, pad_end_const, pad_value, ov::op::PadMode::CONSTANT);
        in_op = pad;
    }
    auto relu = make_shared<Relu>(in_op);
    outputs.push_back(relu);
    auto ref = make_shared<Model>(outputs, ParameterVector{X});
    return ref;
}

}  // namespace single_consumer

namespace output_transpose_mult_transposes {
shared_ptr<Model> CreateFunction(size_t num_pad_ops, element::Type input_type) {
    const Shape input_shape{96, 32, 55, 55};

    auto X = make_shared<Parameter>(input_type, input_shape);

    OutputVector outputs;
    Output<Node> in_op = X->output(0);
    auto pad_value = make_shared<Constant>(input_type, Shape{}, 0);
    for (size_t i = 0; i < num_pad_ops; ++i) {
        auto pad_begin_const = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 1, 2, 3});
        auto pad_end_const = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 1, 2, 3});
        auto pad = make_shared<Pad>(in_op, pad_begin_const, pad_end_const, pad_value, ov::op::PadMode::CONSTANT);
        in_op = pad;
    }
    auto order = make_shared<Constant>(element::i64, Shape{4}, vector<int64_t>{0, 3, 1, 2});
    auto transpose_1 = make_shared<Transpose>(in_op, order);
    auto relu_1 = make_shared<Relu>(transpose_1);
    outputs.push_back(relu_1);

    auto transpose_2 = make_shared<Transpose>(in_op, order);
    auto relu_2 = make_shared<Relu>(transpose_2);
    outputs.push_back(relu_2);
    return make_shared<Model>(outputs, ParameterVector{X});
}

shared_ptr<Model> CreateReferenceFunction(size_t num_pad_ops, element::Type input_type) {
    const Shape input_shape{96, 32, 55, 55};

    auto X = make_shared<Parameter>(input_type, input_shape);
    vector<size_t> order_val = {0, 3, 1, 2};
    auto order = make_shared<Constant>(element::i64, Shape{4}, order_val);
    auto transpose = make_shared<Transpose>(X, order);

    OutputVector outputs;
    Output<Node> in_op = transpose->output(0);
    vector<int64_t> pads{0, 1, 2, 3};

    auto axis = make_shared<Constant>(element::i64, Shape{}, 0);
    auto pad_value = make_shared<Constant>(input_type, Shape{}, 0);
    for (size_t i = 0; i < num_pad_ops; ++i) {
        auto pad_begin_const = make_shared<Constant>(element::i64, Shape{4}, TransposePadValues(pads, order_val));
        auto pad_end_const = make_shared<Constant>(element::i64, Shape{4}, TransposePadValues(pads, order_val));
        auto pad = make_shared<Pad>(in_op, pad_begin_const, pad_end_const, pad_value, ov::op::PadMode::CONSTANT);
        in_op = pad;
    }
    auto relu_1 = make_shared<Relu>(in_op);
    auto relu_2 = make_shared<Relu>(in_op);
    outputs.push_back(relu_1);
    outputs.push_back(relu_2);
    auto ref = make_shared<Model>(outputs, ParameterVector{X});
    return ref;
}
}  // namespace output_transpose_mult_transposes
}  // namespace backward

using CreateGraphPadF = function<shared_ptr<Model>(size_t num_pad_ops, element::Type input_type)>;

using TestPadParams = tuple<PassFactoryPtr,
                            size_t,          /* num_pad_ops */
                            CreateGraphPadF, /* model_factory */
                            CreateGraphPadF, /* reference_model_factory */
                            element::Type> /* input type */;

class TransposeSinkingPadTestFixture : public ::testing::WithParamInterface<TestPadParams>,
                                       public TransformationTestsF {
public:
    static string get_test_name(const testing::TestParamInfo<TestPadParams>& obj) {
        PassFactoryPtr pass_factory;
        size_t num_pad_ops;
        CreateGraphPadF model_factory;
        CreateGraphPadF reference_model_factory;
        element::Type input_type;

        tie(pass_factory, num_pad_ops, model_factory, reference_model_factory, input_type) = obj.param;

        ostringstream test_name;
        test_name << "pass_factory=" << pass_factory->getTypeName() << "_";
        test_name << "num_pad_ops=" << num_pad_ops << "_";
        test_name << "input_type=" << input_type;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingPadTestFixture, CompareFunctions) {
    PassFactoryPtr pass_factory;
    size_t num_pad_ops;
    CreateGraphPadF model_factory;
    CreateGraphPadF reference_model_factory;
    element::Type input_type;
    tie(pass_factory, num_pad_ops, model_factory, reference_model_factory, input_type) = this->GetParam();

    model = model_factory(num_pad_ops, input_type);
    model_ref = reference_model_factory(num_pad_ops, input_type);
    pass_factory->registerPass(manager);
}

std::vector<size_t> pad_operations_numbers = {1, 10};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingPadForwardSingleConsumerTestSuite,
                         TransposeSinkingPadTestFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingPadForward)),
                                            ::testing::ValuesIn(pad_operations_numbers),
                                            ::testing::Values(forward::single_consumer::CreateFunction),
                                            ::testing::Values(forward::single_consumer::CreateReferenceFunction),
                                            ::testing::Values(element::f32)),
                         TransposeSinkingPadTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(TransposeSinkingPadBackwardSingleConsumerTestSuite,
                         TransposeSinkingPadTestFixture,
                         ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingPadBackward)),
                                            ::testing::ValuesIn(pad_operations_numbers),
                                            ::testing::Values(backward::single_consumer::CreateFunction),
                                            ::testing::Values(backward::single_consumer::CreateReferenceFunction),
                                            ::testing::Values(element::f32)),
                         TransposeSinkingPadTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingPadBackwardSingleConsumerMultiTransposesTestSuite,
    TransposeSinkingPadTestFixture,
    ::testing::Combine(::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingPadBackward)),
                       ::testing::ValuesIn(pad_operations_numbers),
                       ::testing::Values(backward::output_transpose_mult_transposes::CreateFunction),
                       ::testing::Values(backward::output_transpose_mult_transposes::CreateReferenceFunction),
                       ::testing::Values(element::f32)),
    TransposeSinkingPadTestFixture::get_test_name);
}  // namespace transpose_sinking_pad