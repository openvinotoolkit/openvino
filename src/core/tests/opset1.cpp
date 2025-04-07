// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"

#include <locale>
#include <memory>
#include <type_traits>

#include "gtest/gtest.h"

using namespace std;
using namespace ov;

namespace {
string capitalize(string name) {
    locale loc;
    string munged_name(name);
    if (munged_name.size() >= 2) {
        munged_name[1] = std::toupper(munged_name[1], loc);
    }
    return munged_name;
}
}  // namespace

#define CHECK_OPSET(op1, op2)                                                                                \
    EXPECT_TRUE(is_type<op1>(make_shared<op2>()));                                                           \
    EXPECT_TRUE((std::is_same<op1, op2>::value));                                                            \
    EXPECT_TRUE((get_opset1().contains_type<op2>()));                                                        \
    {                                                                                                        \
        shared_ptr<Node> op(get_opset1().create(op2::get_type_info_static().name));                          \
        ASSERT_TRUE(op);                                                                                     \
        EXPECT_TRUE(is_type<op2>(op));                                                                       \
        shared_ptr<Node> opi(get_opset1().create_insensitive(capitalize(op2::get_type_info_static().name))); \
        ASSERT_TRUE(opi);                                                                                    \
        EXPECT_TRUE(is_type<op2>(opi));                                                                      \
    }

TEST(opset, check_opset1) {
    CHECK_OPSET(op::v0::Abs, opset1::Abs)
    CHECK_OPSET(op::v0::Acos, opset1::Acos)
    CHECK_OPSET(op::v1::Add, opset1::Add)
    CHECK_OPSET(op::v0::Asin, opset1::Asin)
    CHECK_OPSET(op::v1::LogicalAnd, opset1::LogicalAnd)
    CHECK_OPSET(op::v0::Atan, opset1::Atan)
    CHECK_OPSET(op::v1::AvgPool, opset1::AvgPool)
    CHECK_OPSET(op::v0::BatchNormInference, opset1::BatchNormInference)
    CHECK_OPSET(op::v1::Broadcast, opset1::Broadcast)
    CHECK_OPSET(op::v0::Ceiling, opset1::Ceiling)
    CHECK_OPSET(op::v0::Concat, opset1::Concat)
    CHECK_OPSET(op::v0::Constant, opset1::Constant)
    CHECK_OPSET(op::v0::Convert, opset1::Convert)
    CHECK_OPSET(op::v1::Convolution, opset1::Convolution)
    CHECK_OPSET(op::v0::Cos, opset1::Cos)
    CHECK_OPSET(op::v0::Cosh, opset1::Cosh)
    CHECK_OPSET(op::v0::CTCGreedyDecoder, opset1::CTCGreedyDecoder)
    CHECK_OPSET(op::v1::DeformablePSROIPooling, opset1::DeformablePSROIPooling)
    CHECK_OPSET(op::v0::DepthToSpace, opset1::DepthToSpace)
    CHECK_OPSET(op::v0::DetectionOutput, opset1::DetectionOutput)
    CHECK_OPSET(op::v1::Divide, opset1::Divide)
    CHECK_OPSET(op::v0::Elu, opset1::Elu)
    CHECK_OPSET(op::v1::Equal, opset1::Equal)
    CHECK_OPSET(op::v0::Erf, opset1::Erf)
    CHECK_OPSET(op::v0::Exp, opset1::Exp)
    CHECK_OPSET(op::v0::FakeQuantize, opset1::FakeQuantize)
    CHECK_OPSET(op::v0::Floor, opset1::Floor)
    CHECK_OPSET(op::v1::Gather, opset1::Gather)
    CHECK_OPSET(op::v1::Greater, opset1::Greater)
    CHECK_OPSET(op::v1::GreaterEqual, opset1::GreaterEqual)
    CHECK_OPSET(op::v0::GRN, opset1::GRN)
    CHECK_OPSET(op::v1::GroupConvolution, opset1::GroupConvolution)
    CHECK_OPSET(op::v1::GroupConvolutionBackpropData, opset1::GroupConvolutionBackpropData)
    CHECK_OPSET(op::v0::HardSigmoid, opset1::HardSigmoid)
    CHECK_OPSET(op::v0::Interpolate, opset1::Interpolate)
    CHECK_OPSET(op::v1::Less, opset1::Less)
    CHECK_OPSET(op::v1::LessEqual, opset1::LessEqual)
    CHECK_OPSET(op::v0::Log, opset1::Log)
    CHECK_OPSET(op::v1::LogicalAnd, opset1::LogicalAnd)
    CHECK_OPSET(op::v1::LogicalNot, opset1::LogicalNot)
    CHECK_OPSET(op::v1::LogicalOr, opset1::LogicalOr)
    CHECK_OPSET(op::v1::LogicalXor, opset1::LogicalXor)
    CHECK_OPSET(op::v0::LRN, opset1::LRN)
    CHECK_OPSET(op::v0::MatMul, opset1::MatMul)
    CHECK_OPSET(op::v1::Maximum, opset1::Maximum)
    CHECK_OPSET(op::v1::MaxPool, opset1::MaxPool)
    CHECK_OPSET(op::v1::Minimum, opset1::Minimum)
    CHECK_OPSET(op::v1::Multiply, opset1::Multiply)
    CHECK_OPSET(op::v0::Negative, opset1::Negative)
    CHECK_OPSET(op::v0::NormalizeL2, opset1::NormalizeL2)
    CHECK_OPSET(op::v1::NotEqual, opset1::NotEqual)
    CHECK_OPSET(op::v1::OneHot, opset1::OneHot)
    CHECK_OPSET(op::v1::Pad, opset1::Pad)
    CHECK_OPSET(op::v0::Parameter, opset1::Parameter)
    CHECK_OPSET(op::v1::Power, opset1::Power)
    CHECK_OPSET(op::v0::PRelu, opset1::PRelu)
    CHECK_OPSET(op::v0::PriorBox, opset1::PriorBox)
    CHECK_OPSET(op::v0::PriorBoxClustered, opset1::PriorBoxClustered)
    CHECK_OPSET(op::v0::Proposal, opset1::Proposal)
    CHECK_OPSET(op::v0::PSROIPooling, opset1::PSROIPooling)
    CHECK_OPSET(op::v1::ReduceLogicalAnd, opset1::ReduceLogicalAnd)
    CHECK_OPSET(op::v1::ReduceLogicalOr, opset1::ReduceLogicalOr)
    CHECK_OPSET(op::v1::ReduceMax, opset1::ReduceMax)
    CHECK_OPSET(op::v1::ReduceMean, opset1::ReduceMean)
    CHECK_OPSET(op::v1::ReduceMin, opset1::ReduceMin)
    CHECK_OPSET(op::v1::ReduceProd, opset1::ReduceProd)
    CHECK_OPSET(op::v1::ReduceSum, opset1::ReduceSum)
    CHECK_OPSET(op::v0::RegionYolo, opset1::RegionYolo)
    CHECK_OPSET(op::v0::Relu, opset1::Relu)
    CHECK_OPSET(op::v1::Reshape, opset1::Reshape)
    CHECK_OPSET(op::v0::Result, opset1::Result)
    CHECK_OPSET(op::v1::Reverse, opset1::Reverse)
    CHECK_OPSET(op::v0::ReverseSequence, opset1::ReverseSequence)
    CHECK_OPSET(op::v0::RNNCell, opset1::RNNCell)
    CHECK_OPSET(op::v1::Select, opset1::Select)
    CHECK_OPSET(op::v0::Selu, opset1::Selu)
    CHECK_OPSET(op::v0::ShapeOf, opset1::ShapeOf)
    CHECK_OPSET(op::v0::ShuffleChannels, opset1::ShuffleChannels)
    CHECK_OPSET(op::v0::Sigmoid, opset1::Sigmoid)
    CHECK_OPSET(op::v0::Sign, opset1::Sign)
    CHECK_OPSET(op::v0::Sin, opset1::Sin)
    CHECK_OPSET(op::v0::Sinh, opset1::Sinh)
    CHECK_OPSET(op::v1::Softmax, opset1::Softmax)
    CHECK_OPSET(op::v0::SpaceToDepth, opset1::SpaceToDepth)
    CHECK_OPSET(op::v1::Split, opset1::Split)
    CHECK_OPSET(op::v0::Sqrt, opset1::Sqrt)
    CHECK_OPSET(op::v0::SquaredDifference, opset1::SquaredDifference)
    CHECK_OPSET(op::v0::Squeeze, opset1::Squeeze)
    CHECK_OPSET(op::v1::StridedSlice, opset1::StridedSlice)
    CHECK_OPSET(op::v1::Subtract, opset1::Subtract)
    CHECK_OPSET(op::v0::Tan, opset1::Tan)
    CHECK_OPSET(op::v0::Tanh, opset1::Tanh)
    CHECK_OPSET(op::v0::TensorIterator, opset1::TensorIterator)
    CHECK_OPSET(op::v0::Tile, opset1::Tile)
    CHECK_OPSET(op::v1::TopK, opset1::TopK)
    CHECK_OPSET(op::v1::Transpose, opset1::Transpose)
    CHECK_OPSET(op::v0::Unsqueeze, opset1::Unsqueeze)
    CHECK_OPSET(op::v1::VariadicSplit, opset1::VariadicSplit)
    CHECK_OPSET(op::v0::Xor, opset1::Xor)
}

class NewOp : public op::Op {
public:
    OPENVINO_OP("NewOp", 0);

    NewOp() = default;
    void validate_and_infer_types() override{};

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& /* new_args */) const override {
        return make_shared<NewOp>();
    };
};

TEST(opset, new_op) {
    // Copy opset1; don't bash the real thing in a test
    OpSet opset1_copy(get_opset1());
    opset1_copy.insert<NewOp>();
    ASSERT_TRUE(opset1_copy.contains_type<NewOp>());
    {
        shared_ptr<Node> op(opset1_copy.create(NewOp::get_type_info_static().name));
        ASSERT_TRUE(op);
        EXPECT_TRUE(is_type<NewOp>(op));
    }
    shared_ptr<Node> fred;
    fred = shared_ptr<Node>(opset1_copy.create("Fred"));
    EXPECT_FALSE(fred);
    opset1_copy.insert<NewOp>("Fred");
    // Make sure we copied
    fred = shared_ptr<Node>(get_opset1().create("Fred"));
    ASSERT_FALSE(fred);
    // Fred should be in the copy
    fred = shared_ptr<Node>(opset1_copy.create("Fred"));
    EXPECT_TRUE(fred);
    // FReD should also be in the copy
    fred = shared_ptr<Node>(opset1_copy.create_insensitive("FReD"));
    EXPECT_TRUE(fred);
    // Fred should not be in the registry
    ASSERT_FALSE(get_opset1().contains_type(NewOp::get_type_info_static()));
    ASSERT_FALSE(get_opset1().contains_type<NewOp>());
}

TEST(opset, dump) {
    OpSet opset1_copy(get_opset1());
    cout << "All opset1 operations: ";
    for (const auto& t : opset1_copy.get_types_info()) {
        std::cout << t.name << " ";
    }
    cout << endl;
}
