// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/transpose_sinking/ts_binary.hpp"
#include "transformations/transpose_sinking/ts_concat.hpp"
#include "transformations/transpose_sinking/ts_cumsum.hpp"
#include "transformations/transpose_sinking/ts_data_movement.hpp"
#include "transformations/transpose_sinking/ts_interpolate.hpp"
#include "transformations/transpose_sinking/ts_reduction.hpp"
#include "transformations/transpose_sinking/ts_slice.hpp"
#include "transformations/transpose_sinking/ts_split.hpp"
#include "transformations/transpose_sinking/ts_squeeze.hpp"
#include "transformations/transpose_sinking/ts_tile.hpp"
#include "transformations/transpose_sinking/ts_unary.hpp"
#include "transformations/transpose_sinking/ts_unsqueeze.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing;
using namespace transpose_sinking::testing::utils;

namespace transpose_sinking {
namespace testing {
namespace common {

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
        return std::make_shared<ov::op::v1::Pad>(parent_nodes[0],
                                                 parent_nodes[1],
                                                 parent_nodes[2],
                                                 ov::op::PadMode::CONSTANT);
    }
};
FactoryPtr CreatePadFactory(const std::string& type_name) {
    return std::make_shared<PadFactory>(type_name);
}

class Pad12Factory : public IFactory {
public:
    explicit Pad12Factory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<ov::op::v12::Pad>(parent_nodes[0],
                                                  parent_nodes[1],
                                                  parent_nodes[2],
                                                  ov::op::PadMode::CONSTANT);
    }
};
FactoryPtr CreatePad12Factory(const std::string& type_name) {
    return std::make_shared<Pad12Factory>(type_name);
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
        return std::make_shared<ReductionT>(parent_nodes[0], parent_nodes[1], true);
    }
};

template <typename ReductionT>
FactoryPtr CreateReductionFactory(const std::string& type_name) {
    return std::make_shared<ReductionFactory<ReductionT>>(type_name);
}

class InterpolateFactory : public IFactory {
public:
    explicit InterpolateFactory(const std::string& type_name, bool is_reference)
        : IFactory(type_name),
          m_is_reference(is_reference) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        std::vector<size_t> pads_begin{1, 2, 3, 4};
        std::vector<size_t> pads_end{1, 2, 3, 4};
        if (m_is_reference) {
            pads_begin = {4, 3, 2, 1};
            pads_end = {4, 3, 2, 1};
        }
        const Interpolate::InterpolateAttrs attrs{Interpolate::InterpolateMode::NEAREST,
                                                  Interpolate::ShapeCalcMode::SCALES,
                                                  pads_begin,
                                                  pads_end,
                                                  Interpolate::CoordinateTransformMode::HALF_PIXEL,
                                                  Interpolate::NearestMode::ROUND_PREFER_FLOOR,
                                                  false,
                                                  -0.75};
        return std::make_shared<Interpolate>(parent_nodes[0], parent_nodes[1], parent_nodes[2], parent_nodes[3], attrs);
    }

private:
    bool m_is_reference = false;
};

FactoryPtr CreateInterpolateFactory(const std::string& type_name, bool is_reference) {
    return std::make_shared<InterpolateFactory>(type_name, is_reference);
}

class CumSumFactory : public IFactory {
public:
    explicit CumSumFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<CumSum>(parent_nodes[0], parent_nodes[1]);
    }
};

FactoryPtr CreateCumSumFactory(const std::string& type_name) {
    return std::make_shared<CumSumFactory>(type_name);
}

class TileFactory : public IFactory {
public:
    explicit TileFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Tile>(parent_nodes[0], parent_nodes[1]);
    }
};

FactoryPtr CreateTileFactory(const std::string& type_name) {
    return std::make_shared<TileFactory>(type_name);
}

class SliceFactory : public IFactory {
public:
    explicit SliceFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Slice>(parent_nodes[0],
                                       parent_nodes[1],
                                       parent_nodes[2],
                                       parent_nodes[3],
                                       parent_nodes[4]);
    }
};

FactoryPtr CreateSliceFactory(const std::string& type_name) {
    return std::make_shared<SliceFactory>(type_name);
}

class ReshapeFactory : public IFactory {
public:
    explicit ReshapeFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<Reshape>(parent_nodes[0], parent_nodes[1], false);
    }
};

FactoryPtr CreateReshapeFactory(const std::string& type_name) {
    return std::make_shared<ReshapeFactory>(type_name);
}

class FakeQuantizeFactory : public IFactory {
public:
    explicit FakeQuantizeFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        return std::make_shared<FakeQuantize>(parent_nodes[0],
                                              parent_nodes[1],
                                              parent_nodes[2],
                                              parent_nodes[3],
                                              parent_nodes[4],
                                              128);
    }
};

FactoryPtr CreateFakeQuantizeFactory(const std::string& type_name) {
    return std::make_shared<FakeQuantizeFactory>(type_name);
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

#undef CREATE_PAD12_FACTORY
#define CREATE_PAD12_FACTORY(type_name) CreatePad12Factory(#type_name)

#undef CREATE_BATCH_TO_SPACE_FACTORY
#define CREATE_BATCH_TO_SPACE_FACTORY(type_name) CreateBatchToSpaceFactory(#type_name)

#undef CREATE_SPACE_TO_BATCH_FACTORY
#define CREATE_SPACE_TO_BATCH_FACTORY(type_name) CreateSpaceToBatchFactory(#type_name)

#undef CREATE_REDUCTION_FACTORY
#define CREATE_REDUCTION_FACTORY(type_name) CreateReductionFactory<type_name>(#type_name)

#undef CREATE_INTERPOLATE_FACTORY
#define CREATE_INTERPOLATE_FACTORY(type_name, reference_flag) CreateInterpolateFactory(#type_name, reference_flag)

#undef CREATE_CUMSUM_FACTORY
#define CREATE_CUMSUM_FACTORY(type_name) CreateCumSumFactory(#type_name)

#undef CREATE_TILE_FACTORY
#define CREATE_TILE_FACTORY(type_name) CreateTileFactory(#type_name)

#undef CREATE_SLICE_FACTORY
#define CREATE_SLICE_FACTORY(type_name) CreateSliceFactory(#type_name)

#undef CREATE_RESHAPE_FACTORY
#define CREATE_RESHAPE_FACTORY(type_name) CreateReshapeFactory(#type_name)

#undef CREATE_FQ_FACTORY
#define CREATE_FQ_FACTORY(type_name) common::CreateFakeQuantizeFactory(#type_name)

// ----------------------------------------------------------------------------

vector<FactoryPtr> unary_factories = {
    CREATE_UNARY_FACTORY(Abs),     CREATE_UNARY_FACTORY(Acos),     CREATE_UNARY_FACTORY(Acosh),
    CREATE_UNARY_FACTORY(Asin),    CREATE_UNARY_FACTORY(Asinh),    CREATE_UNARY_FACTORY(Atan),
    CREATE_UNARY_FACTORY(Atanh),   CREATE_UNARY_FACTORY(Ceiling),  CREATE_UNARY_FACTORY(Clamp),
    CREATE_UNARY_FACTORY(Cos),     CREATE_UNARY_FACTORY(Cosh),     CREATE_UNARY_FACTORY(Convert),
    CREATE_UNARY_FACTORY(Erf),     CREATE_UNARY_FACTORY(Elu),      CREATE_UNARY_FACTORY(Exp),
    CREATE_UNARY_FACTORY(Floor),   CREATE_UNARY_FACTORY(Gelu),     CREATE_UNARY_FACTORY(HSigmoid),
    CREATE_UNARY_FACTORY(HSwish),  CREATE_UNARY_FACTORY(Log),      CREATE_UNARY_FACTORY(LogicalNot),
    CREATE_UNARY_FACTORY(Mish),    CREATE_UNARY_FACTORY(Negative), CREATE_UNARY_FACTORY(Relu),
    CREATE_UNARY_FACTORY(Sigmoid), CREATE_UNARY_FACTORY(Sign),     CREATE_UNARY_FACTORY(Sin),
    CREATE_UNARY_FACTORY(Sinh),    CREATE_UNARY_FACTORY(SoftPlus), CREATE_UNARY_FACTORY(SoftSign),
    CREATE_UNARY_FACTORY(Sqrt),    CREATE_UNARY_FACTORY(Tan),      CREATE_UNARY_FACTORY(Tanh)};

vector<FactoryPtr> logical_unary_factories = {CREATE_UNARY_FACTORY(IsFinite),
                                              CREATE_UNARY_FACTORY(IsInf),
                                              CREATE_UNARY_FACTORY(IsNaN)};

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

std::vector<FactoryPtr> reduction_factories = {
    CREATE_REDUCTION_FACTORY(ReduceMax),
    CREATE_REDUCTION_FACTORY(ReduceMin),
    CREATE_REDUCTION_FACTORY(ReduceMean),
    CREATE_REDUCTION_FACTORY(ReduceSum),
    CREATE_REDUCTION_FACTORY(ReduceProd),
    CREATE_REDUCTION_FACTORY(ReduceL1),
    CREATE_REDUCTION_FACTORY(ReduceL2),
};

auto wrapper = [](const TestCase& test_case) {
    OPENVINO_ASSERT(test_case.model.main_op.size() == test_case.model_ref.main_op.size(),
                    "The number of main op (testing op) creator have to be the same for the testing model and for"
                    "the reference model.");
    return ::testing::Combine(::testing::Range<size_t>(0, test_case.num_main_ops.size()),
                              ::testing::Range<size_t>(0, test_case.model.main_op.size()),
                              ::testing::Values(test_case));
};

TEST_P(TSTestFixture, CompareFunctions) {
    size_t num_main_ops_idx;
    size_t main_op_idx;
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
    if (test_case.model.main_op[0]->getTypeName() == "Split") {
        disable_result_friendly_names_check();
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

shared_ptr<ov::Model> create_model(size_t main_node_idx,
                                   const ModelDescription& model_desc,
                                   size_t num_ops,
                                   const OutputVector& inputs_to_main) {
    auto new_inputs = model_desc.preprocess_inputs_to_main.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, model_desc.main_op[main_node_idx]);
    auto outputs = model_desc.preprocess_outputs_of_main.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}

auto test_forward_unary = [](const vector<FactoryPtr>& factories, const vector<size_t>& num_main_ops) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryForward);
    test_case.num_main_ops = num_main_ops;
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = factories;
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.main_op = factories;
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnaryForward,
                         TSTestFixture,
                         test_forward_unary(unary_factories, {1, 10}));
INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonLogicalUnaryForward,
                         TSTestFixture,
                         test_forward_unary(logical_unary_factories, {1}));

auto test_forward_binary = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSBinaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {55, 55, 96, 1}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = binary_factories;
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{1}}};
    test_case.model_ref.main_op = binary_factories;
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBinaryForward, TSTestFixture, test_forward_binary());

auto test_forward_binary_broadcasted = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSBinaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
        parameter(element::f32, {96, 55, 55}),
        parameter(element::f32, {1, 1, 1, 1}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = binary_factories;
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_transpose = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        shared_ptr<Node> order;
        if (std::string(out_vec[idxs[0]].get_node_shared_ptr()->get_type_name()) == "PRelu") {
            order = make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{2, 1, 0});
        } else {
            order = make_shared<Constant>(element::i32, Shape{4}, std::vector<int64_t>{0, 3, 2, 1});
        }
        new_out_vec[idxs[0]] = make_shared<Transpose>(out_vec[idxs[0]], order);
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_transpose}, {{1}}};
    test_case.model_ref.main_op = binary_factories;
    test_case.model_ref.preprocess_outputs_of_main = {{new_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBinaryForwardBroadcasted,
                         TSTestFixture,
                         test_forward_binary_broadcasted());

auto test_forward_fq = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSBinaryForward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {55, 55, 96, 1}),
        parameter(element::f32, {1}),
        parameter(element::f32, {55, 1, 1, 1}),
        parameter(element::f32, {55, 55, 1, 1}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_FQ_FACTORY(FakeQuantize)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_unsqueeze_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        auto indices = make_shared<Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2});
        new_out_vec[2] = make_shared<Unsqueeze>(out_vec[2], indices);
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_unsqueeze_for, set_transpose_for}, {{2}, {1, 2, 3, 4}}};
    test_case.model_ref.main_op = {CREATE_FQ_FACTORY(FakeQuantize)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonFQForward, TSTestFixture, test_forward_fq());

auto test_forward_concat = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSConcatForward);
    test_case.num_main_ops = {1, 3};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {55, 55, 96, 1}),
        parameter(element::f32, {55, 55, 96, 1}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_CONCAT_FACTORY(Concat)};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{1, 2}}};
    test_case.model_ref.main_op = {CREATE_CONCAT_REF_FACTORY(Concat)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonConcatForward, TSTestFixture, test_forward_concat());

auto test_forward_split = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSplitForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 9, 55, 55}),
        constant<int64_t>(element::i32, {}, {2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_SPLIT_FACTORY(Split)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{1});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_SPLIT_FACTORY(Split)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0, 1, 2}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSplitForward, TSTestFixture, test_forward_split());

auto test_forward_pad = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 3, 55, 55}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_PAD_FACTORY(Pad)};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2}}};
    test_case.model_ref.main_op = {CREATE_PAD_FACTORY(Pad)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonPadForward, TSTestFixture, test_forward_pad());

auto test_negative_forward_pad = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 3, 55, 55}),
        constant<int64_t>(element::i32, {4}, {1, -2, -3, -4}),
        constant<int64_t>(element::i32, {4}, {1, -2, -3, -4}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_PAD12_FACTORY(Pad12)};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2}}};
    test_case.model_ref.main_op = {CREATE_PAD12_FACTORY(Pad12)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonNegativePad12Forward, TSTestFixture, test_negative_forward_pad());

auto test_forward_batch_to_space = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementForward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
        parameter(element::f32, {128, 55, 3, 128}),
        constant<int64_t>(element::i32, {4}, {1, 2, 2, 2}),
        constant<int64_t>(element::i32, {4}, {1, 2, 2, 2}),
        constant<int64_t>(element::i32, {4}, {1, 2, 2, 2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_BATCH_TO_SPACE_FACTORY(BatchToSpace)};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2, 3}}};
    test_case.model_ref.main_op = {CREATE_BATCH_TO_SPACE_FACTORY(BatchToSpace)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBatchToSpaceForward, TSTestFixture, test_forward_batch_to_space());

auto test_forward_space_to_batch = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {64, 9, 8, 1}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_SPACE_TO_BATCH_FACTORY(SpaceToBatch)};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2, 3}}};
    test_case.model_ref.main_op = {CREATE_SPACE_TO_BATCH_FACTORY(SpaceToBatch)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSpaceToBatchForward, TSTestFixture, test_forward_space_to_batch());

auto test_forward_reduction = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSReductionForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {32, 4, 2, 1}),
        constant<int64_t>(element::i32, {2}, {1, 3}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = reduction_factories;
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{2, 0});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = reduction_factories;
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReductionForward, TSTestFixture, test_forward_reduction());

auto test_forward_interpolate = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSInterpolateForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 2, 48, 80}),
        constant<int64_t>(element::i32, {2}, {24, 160}),
        constant<float>(element::f32, {2}, {0.5, 2.}),
        constant<int64_t>(element::i32, {2}, {1, 2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_INTERPOLATE_FACTORY(Interpolate, false)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(data, out, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_specific_gather_for}, {{3}}};
    test_case.model_ref.main_op = {CREATE_INTERPOLATE_FACTORY(Interpolate, true)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonInterpolateForward, TSTestFixture, test_forward_interpolate());

auto test_forward_cumsum = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSCumSumForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {parameter(element::f32, {1, 2, 48, 80}),
                                constant<int64_t>(element::i64, {}, std::vector<int64_t>{0})};

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_CUMSUM_FACTORY(CumSum)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(data, out, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_specific_gather_for}, {{1}}};
    test_case.model_ref.main_op = {CREATE_CUMSUM_FACTORY(CumSum)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonCumSumForward, TSTestFixture, test_forward_cumsum());

auto test_forward_tile = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSTileForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {parameter(element::f32, {1, 2, 48, 80}),
                                constant<int64_t>(element::i64, {4}, std::vector<int64_t>{1, 2, 3, 4})};

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_TILE_FACTORY(Tile)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(out, data, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_specific_gather_for}, {{1}}};
    test_case.model_ref.main_op = {CREATE_TILE_FACTORY(Tile)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonTileForward, TSTestFixture, test_forward_tile());

auto test_forward_squeeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSqueezeForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {32, 1, 2, 1}),
        constant<int64_t>(element::i32, {2}, {0, 2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_BINARY_FACTORY(Squeeze)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{3, 1});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_BINARY_FACTORY(Squeeze)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSqueezeForward, TSTestFixture, test_forward_squeeze());

auto test_forward_unsqueeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnsqueezeForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {32, 3, 2, 1}),
        constant<int64_t>(element::i32, {2}, {0, 2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_BINARY_FACTORY(Unsqueeze)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{0, 2});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_BINARY_FACTORY(Unsqueeze)};

    auto new_transpose = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        auto order = make_shared<Constant>(element::i32, Shape{6}, std::vector<int64_t>{0, 5, 2, 4, 3, 1});
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        return new_out_vec;
    };
    test_case.model_ref.preprocess_outputs_of_main = {{new_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnsqueezeForward, TSTestFixture, test_forward_unsqueeze());

auto test_forward_slice = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSliceForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {6, 4, 5, 3}),
        constant<int64_t>(element::i32, {3}, {1, 2, 3}),
        constant<int64_t>(element::i32, {3}, {0, 4, 11}),
        constant<int64_t>(element::i32, {3}, {1, 2, -1}),
        constant<int64_t>(element::i32, {3}, {0, 1, 2}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_SLICE_FACTORY(SliceFactory)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(data, out, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_specific_gather_for}, {{4}}};
    test_case.model_ref.main_op = {CREATE_SLICE_FACTORY(Slice)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSliceForward, TSTestFixture, test_forward_slice());

auto test_forward_reshape_squeeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSqueezeForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {6, 1, 5, 1, 4}),
        constant<int64_t>(element::i32, {3}, {4, 5, 6}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{6, 5, 4});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReshapeSqueezeForward, TSTestFixture, test_forward_reshape_squeeze());

auto test_forward_reshape_unsqueeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnsqueezeForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {6, 5, 4}),
        constant<int64_t>(element::i32, {5}, {4, 1, 5, 1, 6}),
    };

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_transpose = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        auto order = make_shared<Constant>(element::i32, Shape{5}, std::vector<int64_t>{4, 1, 2, 3, 0});
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        return new_out_vec;
    };
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = make_shared<Constant>(out_vec[1].get_element_type(),
                                               out_vec[1].get_shape(),
                                               std::vector<int64_t>{6, 1, 5, 1, 4});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_constant}, {{1}}};
    test_case.model_ref.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model_ref.preprocess_outputs_of_main = {{new_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReshapeUnsqueezeForward,
                         TSTestFixture,
                         test_forward_reshape_unsqueeze());

// ------------------ BACKWARD --------------------

auto test_backward_unary = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
    };

    // Test model description:
    test_case.model.main_op = unary_factories;
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.main_op = unary_factories;
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnaryBackward, TSTestFixture, test_backward_unary());

auto test_backward_binary = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSBinaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {1, 96, 55, 55}),
    };

    // Test model description:
    test_case.model.main_op = binary_factories;
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{0, 1}}};
    test_case.model_ref.main_op = binary_factories;
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBinaryBackward, TSTestFixture, test_backward_binary());

auto test_backward_fq = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSBinaryBackward);
    test_case.num_main_ops = {1, 10};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {1}),
        parameter(element::f32, {1, 96, 55, 1}),
        parameter(element::f32, {1, 96, 1, 1}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_FQ_FACTORY(FakeQuantize)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    auto set_unsqueeze_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        auto indices = make_shared<Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2});
        new_out_vec[2] = make_shared<Unsqueeze>(out_vec[2], indices);
        return new_out_vec;
    };

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_unsqueeze_for, set_transpose_for}, {{2}, {0, 1, 2, 3, 4}}};
    test_case.model_ref.main_op = {CREATE_FQ_FACTORY(FakeQuantize)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonFQBackward, TSTestFixture, test_backward_fq());

auto test_backward_concat = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSConcatBackward);
    test_case.num_main_ops = {1, 3};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {1, 96, 55, 55}),
        parameter(element::f32, {1, 96, 55, 55}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_CONCAT_FACTORY(Concat)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for}, {{0, 1, 2}}};
    test_case.model_ref.main_op = {CREATE_CONCAT_REF_FACTORY(Concat)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonConcatBackward, TSTestFixture, test_backward_concat());

auto test_backward_split = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSplitBackward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 9, 55, 55}),
        constant<int64_t>(element::i32, {}, {1}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_SPLIT_FACTORY(Split)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0, 1, 2}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{2});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, new_constant}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_SPLIT_FACTORY(Split)};
    test_case.model_ref.model_template = create_model;
    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSplitBackward, TSTestFixture, test_backward_split());

auto test_backward_pad = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementBackward);
    test_case.num_main_ops = {1, 2};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 3, 55, 55}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_PAD_FACTORY(Pad)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_gather_for}, {{0}, {1, 2}}};
    test_case.model_ref.main_op = {CREATE_PAD_FACTORY(Pad)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonPadBackward, TSTestFixture, test_backward_pad());

auto test_backward_batch_to_space = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {128, 55, 3, 128}),
        constant<int64_t>(element::i32, {4}, {1, 2, 2, 2}),
        constant<int64_t>(element::i32, {4}, {1, 2, 2, 2}),
        constant<int64_t>(element::i32, {4}, {1, 2, 2, 2}),
    };

    // Reference model description:
    test_case.model.main_op = {CREATE_BATCH_TO_SPACE_FACTORY(BatchToSpace)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Test model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_gather_for}, {{0}, {1, 2, 3}}};
    test_case.model_ref.main_op = {CREATE_BATCH_TO_SPACE_FACTORY(BatchToSpace)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonBatchToSpaceBackward, TSTestFixture, test_backward_batch_to_space());

auto test_backward_space_to_batch = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSDataMovementBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 8, 9, 64}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
        constant<int64_t>(element::i32, {4}, {1, 2, 3, 4}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_SPACE_TO_BATCH_FACTORY(SpaceToBatch)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_gather_for}, {{0}, {1, 2, 3}}};
    test_case.model_ref.main_op = {CREATE_SPACE_TO_BATCH_FACTORY(SpaceToBatch)};
    test_case.model_ref.model_template = create_model;
    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSpaceToBatchBackward, TSTestFixture, test_backward_space_to_batch());

auto test_backward_reduction = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSReductionBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {32, 4, 2, 1}),
        constant<int64_t>(element::i32, {2}, {1, 3}),
    };

    // Test model description:
    test_case.model.main_op = reduction_factories;
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{2, 0});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, new_constant}, {{0}, {1}}};
    test_case.model_ref.main_op = reduction_factories;
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReductionBackward, TSTestFixture, test_backward_reduction());

auto test_backward_interpolate = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSInterpolateBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {1, 2, 48, 80}),
        constant<int64_t>(element::i32, {2}, {24, 160}),
        constant<float>(element::f32, {2}, {0.5, 2.}),
        constant<int64_t>(element::i32, {2}, {1, 2}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_INTERPOLATE_FACTORY(Interpolate, true)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(data, out, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_specific_gather_for}, {{0}, {3}}};
    test_case.model_ref.main_op = {CREATE_INTERPOLATE_FACTORY(Interpolate, false)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonInterpolateBackward, TSTestFixture, test_backward_interpolate());

auto test_backward_cumsum = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSCumSumBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {parameter(element::f32, {1, 2, 48, 80}),
                                constant<int64_t>(element::i64, {}, std::vector<int64_t>{0})};

    // Test model description:
    test_case.model.main_op = {CREATE_CUMSUM_FACTORY(CumSum)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(data, out, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_specific_gather_for}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_CUMSUM_FACTORY(CumSum)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonCumSumBackward, TSTestFixture, test_backward_cumsum());

auto test_backward_tile = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSTileBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {parameter(element::f32, {1, 2, 48, 80}),
                                constant<int64_t>(element::i64, {4}, std::vector<int64_t>{1, 2, 3, 4})};

    // Test model description:
    test_case.model.main_op = {CREATE_TILE_FACTORY(Tile)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(out, data, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_specific_gather_for}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_TILE_FACTORY(Tile)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonTileBackward, TSTestFixture, test_backward_tile());

auto test_backward_tile_tf_case = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSTileBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {parameter(element::f32, {2, 1, 1, 128}),
                                constant<int64_t>(element::i64, {4}, std::vector<int64_t>{1, 1, 88, 1})};

    // Test model description:
    test_case.model.main_op = {CREATE_TILE_FACTORY(Tile)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(out, data, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_specific_gather_for}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_TILE_FACTORY(Tile)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonTileBackwardTfCase, TSTestFixture, test_backward_tile_tf_case());

auto test_backward_unsqueeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnsqueezeBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {32, 3, 2, 1}),
        constant<int64_t>(element::i32, {2}, {0, 2}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_BINARY_FACTORY(Unsqueeze)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{5, 3});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, new_constant}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_BINARY_FACTORY(Unsqueeze)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnsqueezeBackward, TSTestFixture, test_backward_unsqueeze());

auto test_backward_slice = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSliceBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {6, 4, 5, 3}),
        constant<int64_t>(element::i32, {3}, {1, 2, 3}),
        constant<int64_t>(element::i32, {3}, {0, 4, 11}),
        constant<int64_t>(element::i32, {3}, {1, 2, -1}),
        constant<int64_t>(element::i32, {3}, {0, 1, 2}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_SLICE_FACTORY(Slice)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto set_specific_gather_for = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];
            vector<int64_t> transpose_order(out_vec[0].get_shape().size());
            iota(transpose_order.begin(), transpose_order.end(), 0);
            reverse(transpose_order.begin(), transpose_order.end());
            auto data = make_shared<Constant>(element::i32, Shape{transpose_order.size()}, transpose_order);
            auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
            auto transpose = make_shared<Gather>(data, out, axis);
            result[idx] = transpose;
        }
        return result;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, set_specific_gather_for}, {{0}, {4}}};
    test_case.model_ref.main_op = {CREATE_SLICE_FACTORY(SliceFactory)};
    test_case.model_ref.model_template = create_model;
    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonSliceBackward, TSTestFixture, test_backward_slice());

auto test_backward_reshape_squeeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSSqueezeBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {4, 1, 5, 1, 6}),
        constant<int64_t>(element::i32, {3}, {4, 5, 6}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_transpose = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        auto order = make_shared<Constant>(element::i32, Shape{5}, std::vector<int64_t>{4, 1, 2, 3, 0});
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        new_out_vec[1] = out_vec[1];
        return new_out_vec;
    };
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] =
            make_shared<Constant>(out_vec[1].get_element_type(), out_vec[1].get_shape(), std::vector<int64_t>{6, 5, 4});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{new_transpose, new_constant}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReshapeSqueezeBackward, TSTestFixture, test_backward_reshape_squeeze());

auto test_backward_reshape_unsqueeze = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnsqueezeBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, {4, 5, 6}),
        constant<int64_t>(element::i32, {5}, {4, 1, 5, 1, 6}),
    };

    // Test model description:
    test_case.model.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto new_constant = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        new_out_vec[0] = out_vec[0];
        new_out_vec[1] = make_shared<Constant>(out_vec[1].get_element_type(),
                                               out_vec[1].get_shape(),
                                               std::vector<int64_t>{6, 1, 5, 1, 4});
        return new_out_vec;
    };
    test_case.model_ref.preprocess_inputs_to_main = {{set_transpose_for, new_constant}, {{0}, {1}}};
    test_case.model_ref.main_op = {CREATE_RESHAPE_FACTORY(Reshape)};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonReshapeUnsqueezeBackward,
                         TSTestFixture,
                         test_backward_reshape_unsqueeze());

auto test_backward_unsqueeze_dyn_rank = []() {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSUnsqueezeBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = {
        parameter(element::f32, PartialShape::dynamic()),
        constant<int64_t>(element::i32, {2}, {-1}),
    };

    auto dyn_transpose = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector result = out_vec;
        for (const auto& idx : idxs) {
            const auto& out = out_vec[idx];

            // fill the order const with the stub values {-1, -2}
            auto order = make_shared<Constant>(element::i32, Shape{2}, vector<int64_t>{-1, -2});
            auto transpose = make_shared<Transpose>(out, order);
            result[idx] = transpose;
        }
        return result;
    };

    // Test model description:
    test_case.model.main_op = {CREATE_BINARY_FACTORY(Unsqueeze)};
    test_case.model.preprocess_outputs_of_main = {{dyn_transpose}, {{0}}};
    test_case.model.model_template = create_model;

    // Ref model description, the same as the original model, the transformation is not applied
    // it's expected.
    test_case.model_ref.main_op = {CREATE_BINARY_FACTORY(Unsqueeze)};
    test_case.model_ref.preprocess_outputs_of_main = {{dyn_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;
    return wrapper(test_case);
};

INSTANTIATE_TEST_SUITE_P(TransposeSinkingCommonUnsqueezeBackwardDynRank,
                         TSTestFixture,
                         test_backward_unsqueeze_dyn_rank());
}  // namespace common
}  // namespace testing
}  // namespace transpose_sinking
