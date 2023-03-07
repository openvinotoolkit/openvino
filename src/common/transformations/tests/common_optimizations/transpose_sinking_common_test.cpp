// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_unary.hpp"
#include "transformations/common_optimizations/transpose_sinking_binary.hpp"
#include "transformations/common_optimizations/transpose_sinking_concat.hpp"
#include "transformations/common_optimizations/transpose_sinking_split.hpp"
#include "transformations/common_optimizations/transpose_sinking_data_movement.hpp"
#include "transformations/common_optimizations/transpose_sinking_reduction.hpp"
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

class PadFactory : public IFactory {
public:
    explicit PadFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Pad>(parent_nodes[0], parent_nodes[1], parent_nodes[2], ov::op::PadMode::CONSTANT);
    }
};
FactoryPtr CreatePadFactory(const std::string& type_name) {
    return std::make_shared<PadFactory>(type_name);
}

class BatchToSpaceFactory : public IFactory {
public:
    explicit BatchToSpaceFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<BatchToSpace>(parent_nodes[0], parent_nodes[1], parent_nodes[2], parent_nodes[3]);
    }
};

FactoryPtr CreateBatchToSpaceFactory(const std::string& type_name) {
    return std::make_shared<BatchToSpaceFactory>(type_name);
}

class SpaceToBatchFactory : public IFactory {
public:
    explicit SpaceToBatchFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<SpaceToBatch>(parent_nodes[0], parent_nodes[1], parent_nodes[2], parent_nodes[3]);
    }
};
FactoryPtr CreateSpaceToBatchFactory(const std::string& type_name) {
    return std::make_shared<SpaceToBatchFactory>(type_name);
}

template <typename ReductionT>
class ReductionFactory : public IFactory {
public:
    explicit ReductionFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<ReductionT>(parent_nodes[0], parent_nodes[1]);
    }
};

template <typename ReductionT>
FactoryPtr CreateReductionFactory(const std::string& type_name) {
    return std::make_shared<ReductionFactory<ReductionT>>(type_name);
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

#undef CREATE_PAD_FACTORY
#define CREATE_PAD_FACTORY(type_name) CreatePadFactory(#type_name)

#undef CREATE_BATCH_TO_SPACE_FACTORY
#define CREATE_BATCH_TO_SPACE_FACTORY(type_name) CreateBatchToSpaceFactory(#type_name)

#undef CREATE_SPACE_TO_BATCH_FACTORY
#define CREATE_SPACE_TO_BATCH_FACTORY(type_name) CreateSpaceToBatchFactory(#type_name)

#undef CREATE_REDUCTION_FACTORY
#define CREATE_REDUCTION_FACTORY(type_name) CreateReductionFactory<type_name>(#type_name)
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

struct TestCase;
struct ModelDescription;
using TestParams = tuple<size_t /* idx num_main_ops */, size_t /* idx main_op */, TestCase>;
using CreateGraphF = function<shared_ptr<ov::Model>(size_t main_op_idx, const ModelDescription&, size_t, const OutputVector&)>;

// Describes a model to test.
// Expects to be used in such a scenario:
// 1st Preprocessing inserts Transpose/Gather to the inputs
// of the main node.
// Factory contains the rules how to create the main testing node.
// 2nd Preprocessing inserts Transpose/Gather to the outputs
// of the main node.
// model_template is a function which uses the arguments above.
// Examples of the scenarios:
// ModelDescription model: Param -> (Transpose inserted by 1st Preprocessing) -> Abs (main_node) -> Result
// ModelDescription reference: Param -> Abs (main_node) -> (Transpose inserted by 2nd Preprocessing) -> Result
struct ModelDescription {
    Preprocessing preprocess_inputs_to_main;
    // @parameterized with multiple values
    vector<FactoryPtr> main_op;
    Preprocessing preprocess_outputs_of_main;
    CreateGraphF model_template;
};

struct TestCase {
    OutputVector inputs_to_main;
    // @parameterized with multiple values
    vector<size_t> num_main_ops;

    ModelDescription model;
    ModelDescription model_ref;
    PassFactoryPtr transformation;
};

class TransposeSinkingTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
public:
    static string get_test_name(testing::TestParamInfo<TestParams> obj) {
        size_t num_main_ops_idx;
        size_t main_op_idx;
        TestCase test_case;
        tie(num_main_ops_idx, main_op_idx, test_case) = obj.param;

        ostringstream test_name;
        test_name << "Factory=" << test_case.model.main_op[main_op_idx]->getTypeName() << "/";
        test_name << "NumOps=" << test_case.num_main_ops[num_main_ops_idx] << "/";
        test_name << "Transformation=" << test_case.transformation->getTypeName() << "/";
        return test_name.str();
    }
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

std::vector<FactoryPtr> reduction_factories = {CREATE_BINARY_FACTORY(ReduceMax),
                                                CREATE_BINARY_FACTORY(ReduceMin),
                                                CREATE_BINARY_FACTORY(ReduceMean),
                                                CREATE_BINARY_FACTORY(ReduceSum),
                                                CREATE_BINARY_FACTORY(ReduceProd),
                                                //CREATE_BINARY_FACTORY(ReduceLogicalOr),
                                                //CREATE_BINARY_FACTORY(ReduceLogicalAnd),
                                                CREATE_BINARY_FACTORY(ReduceL1),
                                                CREATE_BINARY_FACTORY(ReduceL2),
                                                //CREATE_BINARY_FACTORY(Squeeze),
                                                //CREATE_BINARY_FACTORY(Unsqueeze),
                                                };


TEST_P(TransposeSinkingTestFixture, CompareFunctions) {
    int num_main_ops_idx;
    int main_op_idx;
    TestCase test_case;
    tie(num_main_ops_idx, main_op_idx, test_case) = this->GetParam();
    model = test_case.model.model_template(main_op_idx,
                                           test_case.model,
                                           test_case.num_main_ops[num_main_ops_idx],
                                           test_case.inputs_to_main);

    model_ref = test_case.model_ref.model_template(main_op_idx,
                                                   test_case.model_ref,
                                                   test_case.num_main_ops[num_main_ops_idx],
                                                   test_case.inputs_to_main);
    test_case.transformation->registerPass(manager);
    // TODO: enable accuracy testing. The current issues: div by 0
    // comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

namespace transpose_sinking {
namespace common {

shared_ptr<ov::Model> create_model(size_t main_node_idx,
                                   const ModelDescription& model_desc,
                                   size_t num_ops,
                                   const OutputVector& inputs_to_main) {
    auto new_inputs = model_desc.preprocess_inputs_to_main.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, model_desc.main_op[main_node_idx]);
    auto outputs = model_desc.preprocess_outputs_of_main.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}
}
}

auto wrapper = [](const TestCase& test_case) {
    OPENVINO_ASSERT(test_case.model.main_op.size() == test_case.model_ref.main_op.size(),
                    "The number of main op (testing op) creator have to be the same for the testing model and for"
                    "the reference model.");
    return ::testing::Combine(::testing::Range<size_t>(0, test_case.num_main_ops.size()),
                              ::testing::Range<size_t>(0, test_case.model.main_op.size()),
                              ::testing::Values(test_case));
};

auto test_forward_unary = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingUnaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
            parameter(element::f32, {1, 96, 55, 55}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = unary_factories;
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    test_case.model_ref.main_op = unary_factories;
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnaryForward,
                         TransposeSinkingTestFixture,
                         test_forward_unary());

auto test_forward_binary = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingBinaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
            parameter(element::f32, {1, 96, 55, 55}),
            parameter(element::f32, {55, 55, 96, 1}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = binary_factories;
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{1}}};
    test_case.model_ref.main_op = binary_factories;
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBinaryForward,
                         TransposeSinkingTestFixture,
                         test_forward_binary());


auto test_forward_concat = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingConcatForward);
    test_case.num_main_ops = {1, 3};
    test_case.inputs_to_main = {
            parameter(element::f32, {1, 96, 55, 55}),
            parameter(element::f32, {55, 55, 96, 1}),
            parameter(element::f32, {55, 55, 96, 1}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_CONCAT_FACTORY(Concat)};
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{1, 2}}};
    test_case.model_ref.main_op = {CREATE_CONCAT_REF_FACTORY(Concat)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonConcatForward,
                         TransposeSinkingTestFixture,
                         test_forward_concat());

auto test_forward_split = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingSplitForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
            parameter(element::f32, {1, 9, 55, 55}),
            constant(element::i32, {}, {2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_SPLIT_FACTORY(Split)};
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{1});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_SPLIT_FACTORY(Split)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0, 1, 2}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSplitForward,
                         TransposeSinkingTestFixture,
                         test_forward_split());

auto test_forward_pad = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingDataMovementForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
            parameter(element::f32, {1, 3, 55, 55}),
            constant(element::i32, {4}, {1, 2, 3, 4}),
            constant(element::i32, {4}, {1, 2, 3, 4}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_PAD_FACTORY(Pad)};
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2}}};
    test_case.model_ref.main_op = {CREATE_PAD_FACTORY(Pad)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonPadForward,
                         TransposeSinkingTestFixture,
                         test_forward_pad());

auto test_forward_batch_to_space = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingDataMovementForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
            parameter(element::f32, {128, 55, 3, 128}),
            constant(element::i32, {4}, {1, 2, 2, 2}),
            constant(element::i32, {4}, {1, 2, 2, 2}),
            constant(element::i32, {4}, {1, 2, 2, 2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_BATCH_TO_SPACE_FACTORY(BatchToSpace)};
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2, 3}}};
    test_case.model_ref.main_op = {CREATE_BATCH_TO_SPACE_FACTORY(BatchToSpace)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBatchToSpaceForward,
                         TransposeSinkingTestFixture,
                         test_forward_batch_to_space());

auto test_forward_space_to_batch = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingDataMovementForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
            parameter(element::f32, {64, 9, 8, 1}),
            constant(element::i32, {4}, {1, 2, 3, 4}),
            constant(element::i32, {4}, {1, 2, 3, 4}),
            constant(element::i32, {4}, {1, 2, 3, 4}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_SPACE_TO_BATCH_FACTORY(SpaceToBatch)};
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2, 3}}};
    test_case.model_ref.main_op = {CREATE_SPACE_TO_BATCH_FACTORY(SpaceToBatch)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSpaceToBatchForward,
                         TransposeSinkingTestFixture,
                         test_forward_space_to_batch());

auto test_forward_reduction = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TransposeSinkingReductionForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
            parameter(element::f32, {32, 4, 2, 1}),
            constant(element::i32, {2}, {1, 3}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = reduction_factories;
    test_case.model.model_template = transpose_sinking::common::create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{2, 0});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = reduction_factories;
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = transpose_sinking::common::create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReductionForward,
                         TransposeSinkingTestFixture,
                         test_forward_reduction());
