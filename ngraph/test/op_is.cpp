//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"
#include "op/avg_pool.hpp"
#include "op/convolution.hpp"
#include "op/group_conv.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

namespace
{
    void op_is_Abs()
    {
        op::v0::Abs node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Acos()
    {
        op::v0::Acos node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Add()
    {
        op::v0::Add node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Any()
    {
        op::v0::Any node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Asin()
    {
        op::v0::Asin node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Atan()
    {
        op::v0::Atan node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_AvgPool()
    {
        op::v0::AvgPool node;
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
        op::v0::Broadcast node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_BroadcastLike()
    {
        op::v0::BroadcastLike node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Ceiling()
    {
        op::v0::Ceiling node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Clamp()
    {
        op::v0::Clamp node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Concat()
    {
        op::v0::Concat node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Constant()
    {
        op::v0::Constant node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Convert()
    {
        op::v0::Convert node;
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
        op::v0::Cos node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Cosh()
    {
        op::v0::Cosh node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_CumSum()
    {
        op::v0::CumSum node;
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

    void op_is_Dequantize()
    {
        op::v0::Dequantize node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Divide()
    {
        op::v0::Divide node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Dot()
    {
        op::v0::Dot node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Elu()
    {
        op::v0::Elu node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_EmbeddingBagOffsetsSum()
    {
        op::v3::EmbeddingBagOffsetsSum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_EmbeddingBagPackedSum()
    {
        op::v3::EmbeddingBagPackedSum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_EmbeddingSegmentsSum()
    {
        op::v3::EmbeddingSegmentsSum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Equal()
    {
        op::v0::Equal node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Erf()
    {
        op::v0::Erf node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Exp()
    {
        op::v0::Exp node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ExtractImagePatches()
    {
        op::v3::ExtractImagePatches node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
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
        op::v0::Floor node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GRN()
    {
        op::v0::GRN node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GRUCell()
    {
        op::GRUCell node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Gather()
    {
        op::v0::Gather node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GatherND()
    {
        op::v0::GatherND node;
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

    void op_is_Greater()
    {
        op::v0::Greater node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_GreaterEq()
    {
        op::v0::GreaterEq node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_comparison(&node));
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

    void op_is_Less()
    {
        op::v0::Less node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_LessEq()
    {
        op::v0::LessEq node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Log()
    {
        op::v0::Log node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_LRN()
    {
        op::v0::LRN node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_LSTMCell()
    {
        op::LSTMCell node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_LSTMSequence()
    {
        op::LSTMSequence node;
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

    void op_is_Max()
    {
        op::v0::Max node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Maximum()
    {
        op::v0::Maximum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Min()
    {
        op::v0::Min node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Minimum()
    {
        op::v0::Minimum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Multiply()
    {
        op::v0::Multiply node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_MVN()
    {
        op::v0::MVN node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Negative()
    {
        op::v0::Negative node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Not()
    {
        op::v0::Not node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_NotEqual()
    {
        op::v0::NotEqual node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_OneHot()
    {
        op::v0::OneHot node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Or()
    {
        op::v0::Or node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_TRUE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Pad()
    {
        op::v0::Pad node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Parameter()
    {
        op::v0::Parameter node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Passthrough()
    {
        op::v0::Passthrough node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Power()
    {
        op::v0::Power node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
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

    void op_is_Product()
    {
        op::v0::Product node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Quantize()
    {
        op::v0::Quantize node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_QuantizedConvolution()
    {
        op::v0::QuantizedConvolution node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_QuantizedDot()
    {
        op::v0::QuantizedDot node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Range()
    {
        op::v0::Range node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Relu()
    {
        op::v0::Relu node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ReplaceSlice()
    {
        op::v0::ReplaceSlice node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Reshape()
    {
        op::v0::Reshape node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Result()
    {
        op::v0::Result node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Reverse()
    {
        op::v0::Reverse node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_ReverseSequence()
    {
        op::v0::ReverseSequence node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_RNNCell()
    {
        op::RNNCell node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Round()
    {
        op::v0::Round node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Select()
    {
        op::v0::Select node;
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
        op::v0::ShapeOf node;
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
        op::v0::Sigmoid node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sign()
    {
        op::v0::Sign node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sin()
    {
        op::v0::Sin node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sinh()
    {
        op::v0::Sinh node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Slice()
    {
        op::v0::Slice node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Softmax()
    {
        op::v0::Softmax node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
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

    void op_is_Split()
    {
        op::v0::Split node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sqrt()
    {
        op::v0::Sqrt node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_SquaredDifference()
    {
        op::SquaredDifference node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Squeeze()
    {
        op::v0::Squeeze node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_StopGradient()
    {
        op::v0::StopGradient node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Subtract()
    {
        op::v0::Subtract node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_TRUE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Sum()
    {
        op::v0::Sum node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Tan()
    {
        op::v0::Tan node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Tanh()
    {
        op::v0::Tanh node;
        EXPECT_TRUE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_TensorIterator()
    {
        op::v0::TensorIterator node;
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

    void op_is_TopK()
    {
        op::v0::TopK node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Unsqueeze()
    {
        op::Unsqueeze node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_FALSE(op::is_binary_elementwise_logical(&node));
    }

    void op_is_Xor()
    {
        op::v0::Xor node;
        EXPECT_FALSE(op::is_unary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_arithmetic(&node));
        EXPECT_FALSE(op::is_binary_elementwise_comparison(&node));
        EXPECT_TRUE(op::is_binary_elementwise_logical(&node));
    }
}

TEST(op_is, check)
{
#define NGRAPH_OP(a, b) op_is_##a();
#include "opset0_tbl.hpp"
#undef NGRAPH_OP
}
