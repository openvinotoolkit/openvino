// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"
#include "op/convolution.hpp"
#include "op/group_conv.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

namespace
{
    void op_is_Abs()
    {
        op::Abs node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Acos()
    {
        op::Acos node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Asin()
    {
        op::Asin node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Atan()
    {
        op::Atan node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_AvgPool()
    {
        op::AvgPool node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_BatchNormInference()
    {
        op::v0::BatchNormInference node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Broadcast()
    {
        op::v1::Broadcast node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Ceiling()
    {
        op::Ceiling node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Clamp()
    {
        op::Clamp node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Concat()
    {
        op::Concat node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Constant()
    {
        op::Constant node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Convert()
    {
        op::Convert node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Convolution()
    {
        op::v0::Convolution node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ConvolutionBackpropData()
    {
        op::v0::ConvolutionBackpropData node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Cos()
    {
        op::Cos node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Cosh()
    {
        op::Cosh node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_CumSum()
    {
        op::CumSum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_DepthToSpace()
    {
        op::DepthToSpace node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Elu()
    {
        op::Elu node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Erf()
    {
        op::Erf node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Exp()
    {
        op::Exp node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_FakeQuantize()
    {
        op::FakeQuantize node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Floor()
    {
        op::Floor node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GRN()
    {
        op::GRN node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Gather()
    {
        op::v1::Gather node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Gelu()
    {
        op::Gelu node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GroupConvolution()
    {
        op::v0::GroupConvolution node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GroupConvolutionBackpropData()
    {
        op::v0::GroupConvolutionBackpropData node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_HardSigmoid()
    {
        op::HardSigmoid node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Interpolate()
    {
        op::v0::Interpolate node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Log()
    {
        op::Log node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_LRN()
    {
        op::LRN node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_LSTMSequence()
    {
        op::v0::LSTMSequence node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_MatMul()
    {
        op::MatMul node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_NormalizeL2()
    {
        op::NormalizeL2 node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_MVN()
    {
        op::MVN node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Negative()
    {
        op::Negative node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Parameter()
    {
        op::Parameter node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_PRelu()
    {
        op::PRelu node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_PriorBox()
    {
        op::PriorBox node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Range()
    {
        op::Range node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Relu()
    {
        op::Relu node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Result()
    {
        op::Result node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ReverseSequence()
    {
        op::ReverseSequence node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Selu()
    {
        op::Selu node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ShapeOf()
    {
        op::ShapeOf node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ShuffleChannels()
    {
        op::ShuffleChannels node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sigmoid()
    {
        op::Sigmoid node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sign()
    {
        op::Sign node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sin()
    {
        op::Sin node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sinh()
    {
        op::Sinh node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_SpaceToDepth()
    {
        op::SpaceToDepth node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sqrt()
    {
        op::Sqrt node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_SquaredDifference()
    {
        op::SquaredDifference node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Squeeze()
    {
        op::Squeeze node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Tan()
    {
        op::Tan node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Tanh()
    {
        op::Tanh node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_TensorIterator()
    {
        op::TensorIterator node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Tile()
    {
        op::v0::Tile node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Unsqueeze()
    {
        op::v0::Unsqueeze node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Xor()
    {
        op::Xor node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_TRUE(op::is_binary_elementwise_logical(&node));
    }
} // namespace

TEST(op_is, check)
{
    NGRAPH_SUPPRESS_DEPRECATED_START
#define NGRAPH_OP(a, b) op_is_##a();
#include "opset0_tbl.hpp"
#undef NGRAPH_OP
    NGRAPH_SUPPRESS_DEPRECATED_END
}
