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
#include "ngraph/validation_util.hpp"
#include "op/atan2.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

namespace
{
    void op_is_Abs()
    {
        op::Abs node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Acos()
    {
        op::Acos node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Add()
    {
        op::Add node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_All()
    {
        op::All node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_AllReduce()
    {
        op::AllReduce node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_And()
    {
        op::And node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_TRUE(node.is_binary_elementwise_logical());
    }

    void op_is_Any()
    {
        op::Any node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ArgMax()
    {
        op::ArgMax node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ArgMin()
    {
        op::ArgMin node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Asin()
    {
        op::Asin node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Atan()
    {
        op::Atan node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Atan2()
    {
        op::v0::Atan2 node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_AvgPool()
    {
        op::AvgPool node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchMatMul()
    {
        op::BatchMatMul node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchMatMulTranspose()
    {
        op::BatchMatMulTranspose node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchNormInference()
    {
        op::BatchNormInference node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BatchNormTraining()
    {
        op::BatchNormTraining node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Broadcast()
    {
        op::Broadcast node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BroadcastDistributed()
    {
        op::BroadcastDistributed node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_BroadcastLike()
    {
        op::BroadcastLike node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Ceiling()
    {
        op::Ceiling node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Clamp()
    {
        op::Clamp node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Concat()
    {
        op::Concat node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Constant()
    {
        op::Constant node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Convert()
    {
        op::Convert node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Convolution()
    {
        op::Convolution node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBackpropData()
    {
        op::ConvolutionBackpropData node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBias()
    {
        op::ConvolutionBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ConvolutionBiasAdd()
    {
        op::ConvolutionBiasAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Cos()
    {
        op::Cos node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Cosh()
    {
        op::Cosh node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CrossEntropy()
    {
        op::CrossEntropy node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CropAndResize()
    {
        op::CropAndResize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_CumSum()
    {
        op::CumSum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DepthToSpace()
    {
        op::DepthToSpace node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Dequantize()
    {
        op::Dequantize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Divide()
    {
        op::Divide node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Dot()
    {
        op::Dot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynBroadcast()
    {
        op::DynBroadcast node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynPad()
    {
        op::DynPad node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynReplaceSlice()
    {
        op::DynReplaceSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_DynSlice()
    {
        op::DynSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Elu()
    {
        op::Elu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_EmbeddingBagOffsetsSum()
    {
        op::EmbeddingBagOffsetsSum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_EmbeddingBagPackedSum()
    {
        op::EmbeddingBagPackedSum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_EmbeddingLookup()
    {
        op::EmbeddingLookup node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_EmbeddingSegmentsSum()
    {
        op::EmbeddingSegmentsSum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Equal()
    {
        op::Equal node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Erf()
    {
        op::Erf node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Exp()
    {
        op::Exp node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ExtractImagePatches()
    {
        op::ExtractImagePatches node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_FakeQuantize()
    {
        op::FakeQuantize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Floor()
    {
        op::Floor node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GRN()
    {
        op::GRN node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GRUCell()
    {
        op::GRUCell node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Gather()
    {
        op::Gather node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GatherND()
    {
        op::GatherND node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Gelu()
    {
        op::Gelu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Gemm()
    {
        op::Gemm node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GenerateMask()
    {
        op::GenerateMask node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GetOutputElement()
    {
        op::GetOutputElement node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Greater()
    {
        op::Greater node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GreaterEq()
    {
        op::GreaterEq node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GroupConvolution()
    {
        op::GroupConvolution node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_GroupConvolutionBackpropData()
    {
        op::GroupConvolutionBackpropData node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_HardSigmoid()
    {
        op::HardSigmoid node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Interpolate()
    {
        op::Interpolate node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LayerNorm()
    {
        op::LayerNorm node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Less()
    {
        op::Less node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LessEq()
    {
        op::LessEq node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Log()
    {
        op::Log node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LRN()
    {
        op::LRN node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LSTMCell()
    {
        op::LSTMCell node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_LSTMSequence()
    {
        op::LSTMSequence node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MatMul()
    {
        op::MatMul node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_NormalizeL2()
    {
        op::NormalizeL2 node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Max()
    {
        op::Max node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Maximum()
    {
        op::Maximum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MaxPool()
    {
        op::MaxPool node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Min()
    {
        op::Min node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Minimum()
    {
        op::Minimum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Multiply()
    {
        op::Multiply node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_MVN()
    {
        op::MVN node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Negative()
    {
        op::Negative node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Not()
    {
        op::Not node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_NotEqual()
    {
        op::NotEqual node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_OneHot()
    {
        op::OneHot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Or()
    {
        op::Or node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_TRUE(node.is_binary_elementwise_logical());
    }

    void op_is_Pad()
    {
        op::Pad node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Parameter()
    {
        op::Parameter node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_PartialSlice()
    {
        op::PartialSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Passthrough()
    {
        op::Passthrough node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Power()
    {
        op::Power node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_PRelu()
    {
        op::PRelu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Product()
    {
        op::Product node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Quantize()
    {
        op::Quantize node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolution()
    {
        op::QuantizedConvolution node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionBias()
    {
        op::QuantizedConvolutionBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionBiasAdd()
    {
        op::QuantizedConvolutionBiasAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionBiasSignedAdd()
    {
        op::QuantizedConvolutionBiasSignedAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedConvolutionRelu()
    {
        op::QuantizedConvolutionRelu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedDot()
    {
        op::QuantizedDot node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_QuantizedDotBias()
    {
        op::QuantizedDotBias node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_RandomUniform()
    {
        op::RandomUniform node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Recv()
    {
        op::Recv node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Range()
    {
        op::Range node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Relu()
    {
        op::Relu node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ReplaceSlice()
    {
        op::ReplaceSlice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Reshape()
    {
        op::Reshape node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Result()
    {
        op::Result node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Reverse()
    {
        op::Reverse node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ReverseSequence()
    {
        op::ReverseSequence node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_RNNCell()
    {
        op::RNNCell node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Round()
    {
        op::Round node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScalarConstantLike()
    {
        op::ScalarConstantLike node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScaleShift()
    {
        op::ScaleShift node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScatterAdd()
    {
        op::ScatterAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScatterND()
    {
        op::ScatterND node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ScatterNDAdd()
    {
        op::ScatterNDAdd node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Select()
    {
        op::Select node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Selu()
    {
        op::Selu node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Send()
    {
        op::Send node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ShapeOf()
    {
        op::ShapeOf node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_ShuffleChannels()
    {
        op::ShuffleChannels node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sigmoid()
    {
        op::Sigmoid node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sign()
    {
        op::Sign node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sin()
    {
        op::Sin node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sinh()
    {
        op::Sinh node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Slice()
    {
        op::Slice node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Softmax()
    {
        op::Softmax node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SoftmaxCrossEntropy()
    {
        op::SoftmaxCrossEntropy node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SpaceToDepth()
    {
        op::SpaceToDepth node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Split()
    {
        op::Split node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sqrt()
    {
        op::Sqrt node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_SquaredDifference()
    {
        op::SquaredDifference node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Squeeze()
    {
        op::Squeeze node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_StopGradient()
    {
        op::StopGradient node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Stack()
    {
        op::Stack node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Subtract()
    {
        op::Subtract node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_TRUE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Sum()
    {
        op::Sum node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Tan()
    {
        op::Tan node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Tanh()
    {
        op::Tanh node;
        EXPECT_TRUE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_TensorIterator()
    {
        op::TensorIterator node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Tile()
    {
        op::Tile node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_TopK()
    {
        op::TopK node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Unsqueeze()
    {
        op::Unsqueeze node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_FALSE(node.is_binary_elementwise_logical());
    }

    void op_is_Xor()
    {
        op::Xor node;
        EXPECT_FALSE(node.is_unary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_arithmetic());
        EXPECT_FALSE(node.is_binary_elementwise_comparison());
        EXPECT_TRUE(node.is_binary_elementwise_logical());
    }
}

TEST(op_is, check)
{
#define NGRAPH_OP(a, b) op_is_##a();
#include "opset0_tbl.hpp"
#undef NGRAPH_OP
}
