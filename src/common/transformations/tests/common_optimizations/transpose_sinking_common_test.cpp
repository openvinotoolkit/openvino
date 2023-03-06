// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_unary.hpp"
#include "transformations/common_optimizations/transpose_sinking_binary.hpp"
#include "transformations/common_optimizations/transpose_sinking_concat.hpp"
#include "transformations/common_optimizations/transpose_sinking_split.hpp"
#include "transpose_sinking_test_utils.hpp"

#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace ov;
using namespace ov::opset10;

template <typename UnaryT>
class UnaryFactory : public IFactory {
public:
    explicit UnaryFactory(const string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        OPENVINO_ASSERT(parent_nodes.size() == 1, "Exactly one input node expected for Unary operation");
        return make_shared<UnaryT>(parent_nodes[0]);
    }
};

template <>
NodePtr UnaryFactory<Elu>::create(const OutputVector& parent_nodes) const {
    OPENVINO_ASSERT(parent_nodes.size() == 1, "Exactly one input node expected for Unary operation");
    return make_shared<Elu>(parent_nodes[0], 0.1);
}

template <>
NodePtr UnaryFactory<Clamp>::create(const OutputVector& parent_nodes) const {
    OPENVINO_ASSERT(parent_nodes.size() == 1, "Exactly one input node expected for Unary operation");
    return make_shared<Clamp>(parent_nodes[0], 0.1, 0.2);
}

template <>
NodePtr UnaryFactory<Convert>::create(const OutputVector& parent_nodes) const {
    OPENVINO_ASSERT(parent_nodes.size() == 1, "Exactly one input node expected for Unary operation");
    return make_shared<Convert>(parent_nodes[0], element::f64);
}

template <typename UnaryT>
FactoryPtr CreateUnaryFactory(const string& type_name) {
    return make_shared<UnaryFactory<UnaryT>>(type_name);
}

template <typename BinaryT>
class BinaryFactory : public IFactory {
public:
    explicit BinaryFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        OPENVINO_ASSERT(parent_nodes.size() == 2, "Exactly 2 inputs node expected for Unary operation");
        return std::make_shared<BinaryT>(parent_nodes[0], parent_nodes[1]);
    }
};

template <typename BinaryT>
FactoryPtr CreateBinaryFactory(const std::string& type_name) {
    return std::make_shared<BinaryFactory<BinaryT>>(type_name);
}

class ConcatFactory : public IFactory {
public:
    explicit ConcatFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Concat>(parent_nodes, 1);
    }
};
FactoryPtr CreateConcatFactory(const std::string& type_name) {
    return std::make_shared<ConcatFactory>(type_name);
}
class ConcatFactoryRef : public IFactory {
public:
    explicit ConcatFactoryRef(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Concat>(parent_nodes, 2);
    }
};
FactoryPtr CreateConcatRefFactory(const std::string& type_name) {
    return std::make_shared<ConcatFactoryRef>(type_name);
}


class SplitFactory : public IFactory {
public:
    explicit SplitFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Split>(parent_nodes[0], parent_nodes[1], 3);
    }
};
FactoryPtr CreateSplitFactory(const std::string& type_name) {
    return std::make_shared<SplitFactory>(type_name);
}
// ----------------------------------------------------------------------------

#undef CREATE_UNARY_FACTORY
#define CREATE_UNARY_FACTORY(type_name) CreateUnaryFactory<type_name>(#type_name)

#undef CREATE_BINARY_FACTORY
#define CREATE_BINARY_FACTORY(type_name) CreateBinaryFactory<type_name>(#type_name)

#undef CREATE_CONCAT_FACTORY
#define CREATE_CONCAT_FACTORY(type_name) CreateConcatFactory(#type_name)

#undef CREATE_CONCAT_REF_FACTORY
#define CREATE_CONCAT_REF_FACTORY(type_name) CreateConcatRefFactory(#type_name)

#undef CREATE_SPLIT_FACTORY
#define CREATE_SPLIT_FACTORY(type_name) CreateSplitFactory(#type_name)
// ----------------------------------------------------------------------------

struct Preprocessing {
    vector<function<OutputVector(vector<size_t>, OutputVector)>> preprocessing;
    vector<vector<size_t>> indices;

    OutputVector apply(const OutputVector& inputs) const {
        OutputVector new_inputs = inputs;
        for (size_t i = 0; i < preprocessing.size(); ++i) {
            new_inputs = preprocessing[i](indices[i], new_inputs);
        }
        return new_inputs;
    }
};
using CreateGraphF = function<shared_ptr<ov::Model>(Preprocessing, FactoryPtr, Preprocessing, size_t, OutputVector)>;

using TestParams = tuple<FactoryPtr,
                         FactoryPtr,
                         PassFactoryPtr,
                         size_t,         /* num_unary_ops */
                         Preprocessing,
                         CreateGraphF,   /* model_factory */
                         Preprocessing,
                         Preprocessing,
                         CreateGraphF,   /* reference_model_factory */
                         Preprocessing,
                         OutputVector>;

struct TestCases {
    vector<FactoryPtr> main_node;
    vector<FactoryPtr> main_node_ref;
    PassFactoryPtr transformation;
    vector<size_t> num_main_ops;
    Preprocessing preprocess_before;
    CreateGraphF test_model;
    Preprocessing preprocess_after;
    Preprocessing preprocess_before_ref;
    CreateGraphF ref_model;
    Preprocessing preprocess_after_ref;
    OutputVector inputs_to_main;
};

struct TestCase {
    FactoryPtr main_node;
    FactoryPtr main_node_ref;
    PassFactoryPtr transformation;
    size_t num_main_ops = 0;
    Preprocessing preprocess_before;
    CreateGraphF test_model;
    Preprocessing preprocess_after;
    Preprocessing preprocess_before_ref;
    CreateGraphF ref_model;
    Preprocessing preprocess_after_ref;
    OutputVector inputs_to_main;

    explicit TestCase(const TestParams& params) {
        tie(main_node,
            main_node_ref,
            transformation,
            num_main_ops,
            preprocess_before,
            test_model,
            preprocess_after,
            preprocess_before_ref,
            ref_model,
            preprocess_after_ref,
            inputs_to_main) = params;
    }
};

class TransposeSinkingTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
public:
/*    static string get_test_name(const testing::TestParamInfo<TestParams>& obj) {
        auto test_case = TestCase(obj.param);

        ostringstream test_name;
        test_name << "unaryFactory=" << unary_factory->getTypeName() << "/";
        test_name << "numUnaryOps=" << num_unary_ops << "/";
        //test_name << "inputShape=" << to_string(input_shape) << "/";
        test_name << "unaryFactory=" << unary_factory->getTypeName() << "/";
        test_name << "passFactory=" << pass_factory->getTypeName() << "/";
        //test_name << "inputType=" << input_type;

        return test_name.str();
    }*/
};



vector<FactoryPtr> unary_factories = {
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

std::vector<FactoryPtr> binary_factories = {CREATE_BINARY_FACTORY(Add),
                                            CREATE_BINARY_FACTORY(Divide),
                                            CREATE_BINARY_FACTORY(Maximum),
                                            CREATE_BINARY_FACTORY(Minimum),
                                            CREATE_BINARY_FACTORY(Mod),
                                            CREATE_BINARY_FACTORY(Multiply),
                                            CREATE_BINARY_FACTORY(Power),
                                            CREATE_BINARY_FACTORY(SquaredDifference),
                                            CREATE_BINARY_FACTORY(Subtract),
                                            CREATE_BINARY_FACTORY(PRelu)};


TEST_P(TransposeSinkingTestFixture, CompareFunctions) {
    auto test_case = TestCase(this->GetParam());
    model = test_case.test_model(test_case.preprocess_before,
                                 test_case.main_node,
                                 test_case.preprocess_after,
                                 test_case.num_main_ops,
                                 test_case.inputs_to_main);
    model_ref = test_case.ref_model(test_case.preprocess_before_ref,
                                    test_case.main_node_ref,
                                    test_case.preprocess_after_ref,
                                    test_case.num_main_ops,
                                    test_case.inputs_to_main);
    test_case.transformation->registerPass(manager);
}

namespace transpose_sinking {
namespace common {

shared_ptr<ov::Model> create_model(const Preprocessing& preprocess_before,
                                   const FactoryPtr& main_op,
                                        const Preprocessing& preprocess_after,
                                        size_t num_ops,
                                        const OutputVector& inputs_to_main) {
    auto new_inputs = preprocess_before.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, main_op);
    auto outputs = preprocess_after.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}
}
}


auto wrapper = [](const TestCases& test_cases) {
    return ::testing::Combine(::testing::ValuesIn(test_cases.main_node),
                              ::testing::ValuesIn(test_cases.main_node_ref),
                              ::testing::Values(test_cases.transformation),
                              ::testing::ValuesIn(test_cases.num_main_ops),
                              ::testing::Values(test_cases.preprocess_before),
                              ::testing::Values(test_cases.test_model),
                              ::testing::Values(test_cases.preprocess_after),
                              ::testing::Values(test_cases.preprocess_before_ref),
                              ::testing::Values(test_cases.ref_model),
                              ::testing::Values(test_cases.preprocess_after_ref),
                              ::testing::Values(test_cases.inputs_to_main));
};

shared_ptr<Node> parameter(element::Type el_type, const PartialShape& ps) {
    return make_shared<Parameter>(el_type, ps);
}

shared_ptr<Node> constant(element::Type el_type, const Shape& shape, const vector<int64_t>& value) {
    return make_shared<Constant>(el_type, shape, value);
}

auto test_forward_unary = []() {
    TestCases test_cases;
    test_cases.main_node = unary_factories;
    test_cases.main_node_ref = unary_factories;
    test_cases.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryForward);
    test_cases.num_main_ops = {1, 10};
    test_cases.preprocess_before = {{set_transpose_for}, {{0}}};
    test_cases.test_model = transpose_sinking::common::create_model;
    test_cases.ref_model = transpose_sinking::common::create_model;
    test_cases.preprocess_after_ref = {{set_transpose_for}, {{0}}};
    test_cases.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
    };
    return wrapper(test_cases);
};

auto test_forward_binary = []() {
    TestCases test_cases;
    test_cases.main_node = binary_factories;
    test_cases.main_node_ref = binary_factories;
    test_cases.transformation = CREATE_PASS_FACTORY(TransposeSinkingBinaryForward);
    test_cases.num_main_ops = {1, 10};
    test_cases.preprocess_before = {{set_transpose_for}, {{0}}};
    test_cases.test_model = transpose_sinking::common::create_model;
    test_cases.preprocess_before_ref = {{set_transpose_for}, {{1}}};
    test_cases.ref_model = transpose_sinking::common::create_model;
    test_cases.preprocess_after_ref = {{set_transpose_for}, {{0}}};
    test_cases.inputs_to_main = {
            parameter(element::f32, {1, 96, 55, 55}),
            parameter(element::f32, {55, 55, 96, 1}),
    };
    return wrapper(test_cases);
};

auto test_forward_concat = []() {
    TestCases test_cases;
    test_cases.main_node = {CREATE_CONCAT_FACTORY(Concat)};
    test_cases.main_node_ref = {CREATE_CONCAT_REF_FACTORY(Concat)};
    test_cases.transformation = CREATE_PASS_FACTORY(TransposeSinkingConcatForward);
    test_cases.num_main_ops = {1, 10};
    test_cases.preprocess_before = {{set_transpose_for}, {{0}}};
    test_cases.test_model = transpose_sinking::common::create_model;
    test_cases.preprocess_before_ref = {{set_transpose_for}, {{1, 2}}};
    test_cases.ref_model = transpose_sinking::common::create_model;
    test_cases.preprocess_after_ref = {{set_transpose_for}, {{0}}};
    test_cases.inputs_to_main = {
            parameter(element::f32, {1, 96, 55, 55}),
            parameter(element::f32, {55, 55, 96, 1}),
            parameter(element::f32, {55, 55, 96, 1}),
    };
    return wrapper(test_cases);
};

auto test_forward_split = []() {
    TestCases test_cases;
    test_cases.main_node = {CREATE_SPLIT_FACTORY(Concat)};
    test_cases.main_node_ref = {CREATE_SPLIT_FACTORY(Concat)};
    test_cases.transformation = CREATE_PASS_FACTORY(TransposeSinkingSplitForward);
    test_cases.num_main_ops = {1};
    test_cases.preprocess_before = {{set_transpose_for}, {{0}}};
    test_cases.test_model = transpose_sinking::common::create_model;

    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{1});
        return new_out_vec;
    };
    test_cases.preprocess_before_ref = {{new_constant}, {{1}}};
    test_cases.ref_model = transpose_sinking::common::create_model;
    test_cases.preprocess_after_ref = {{set_transpose_for}, {{0, 1, 2}}};
    test_cases.inputs_to_main = {
            parameter(element::f32, {1, 3, 55, 55}),
            constant(element::i32, {}, {2}),
    };
    return wrapper(test_cases);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnaryForward,
                         TransposeSinkingTestFixture,
                         test_forward_unary());

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBinaryForward,
                         TransposeSinkingTestFixture,
                         test_forward_binary());

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonConcatForward,
                         TransposeSinkingTestFixture,
                         test_forward_concat());

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSplitForward,
                         TransposeSinkingTestFixture,
                         test_forward_split());

