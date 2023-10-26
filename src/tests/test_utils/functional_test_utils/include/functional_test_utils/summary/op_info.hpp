// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace functional {

// {{ type_info, real_version }}
const std::map<ov::NodeTypeInfo, size_t> not_aligned_op_version = {
    // opset 1
    {ov::op::v0::Abs::get_type_info_static(), 0},
    {ov::op::v0::Acos::get_type_info_static(), 0},
    {ov::op::v0::Asin::get_type_info_static(), 0},
    {ov::op::v0::Atan::get_type_info_static(), 0},
    {ov::op::v0::BatchNormInference::get_type_info_static(), 0},
    {ov::op::v0::CTCGreedyDecoder::get_type_info_static(), 0},
    {ov::op::v0::Ceiling::get_type_info_static(), 0},
    {ov::op::v0::Clamp::get_type_info_static(), 0},
    {ov::op::v0::Concat::get_type_info_static(), 0},
    {ov::op::v0::Constant::get_type_info_static(), 0},
    {ov::op::v0::Convert::get_type_info_static(), 0},
    {ov::op::v0::Cos::get_type_info_static(), 0},
    {ov::op::v0::Cosh::get_type_info_static(), 0},
    {ov::op::v0::DepthToSpace::get_type_info_static(), 0},
    {ov::op::v0::DetectionOutput::get_type_info_static(), 0},
    {ov::op::v0::Elu::get_type_info_static(), 0},
    {ov::op::v0::Erf::get_type_info_static(), 0},
    {ov::op::v0::Exp::get_type_info_static(), 0},
    {ov::op::v0::FakeQuantize::get_type_info_static(), 0},
    {ov::op::v0::Floor::get_type_info_static(), 0},
    {ov::op::v0::GRN::get_type_info_static(), 0},
    {ov::op::v0::HardSigmoid::get_type_info_static(), 0},
    {ov::op::v0::Interpolate::get_type_info_static(), 0},
    {ov::op::v0::Log::get_type_info_static(), 0},
    {ov::op::v0::LRN::get_type_info_static(), 0},
    {ov::op::v0::LSTMCell::get_type_info_static(), 0},
    {ov::op::v0::LSTMSequence::get_type_info_static(), 0},
    {ov::op::v0::MatMul::get_type_info_static(), 0},
    {ov::op::v0::Negative::get_type_info_static(), 0},
    {ov::op::v0::NormalizeL2::get_type_info_static(), 0},
    {ov::op::v0::PRelu::get_type_info_static(), 0},
    {ov::op::v0::PSROIPooling::get_type_info_static(), 0},
    {ov::op::v0::Parameter::get_type_info_static(), 0},
    {ov::op::v0::PriorBox::get_type_info_static(), 0},
    {ov::op::v0::PriorBoxClustered::get_type_info_static(), 0},
    {ov::op::v0::Proposal::get_type_info_static(), 0},
    {ov::op::v0::Range::get_type_info_static(), 0},
    {ov::op::v0::Relu::get_type_info_static(), 0},
    {ov::op::v0::RegionYolo::get_type_info_static(), 0},
    {ov::op::v0::Result::get_type_info_static(), 0},
    {ov::op::v0::ReverseSequence::get_type_info_static(), 0},
    {ov::op::v0::RNNCell::get_type_info_static(), 0},
    {ov::op::v0::Selu::get_type_info_static(), 0},
    {ov::op::v0::ShapeOf::get_type_info_static(), 0},
    {ov::op::v0::ShuffleChannels::get_type_info_static(), 0},
    {ov::op::v0::Sign::get_type_info_static(), 0},
    {ov::op::v0::Sigmoid::get_type_info_static(), 0},
    {ov::op::v0::Sin::get_type_info_static(), 0},
    {ov::op::v0::Sinh::get_type_info_static(), 0},
    {ov::op::v0::Sqrt::get_type_info_static(), 0},
    {ov::op::v0::SpaceToDepth::get_type_info_static(), 0},
    {ov::op::v0::SquaredDifference::get_type_info_static(), 0},
    {ov::op::v0::Squeeze::get_type_info_static(), 0},
    {ov::op::v0::Tan::get_type_info_static(), 0},
    {ov::op::v0::Tanh::get_type_info_static(), 0},
    {ov::op::v0::TensorIterator::get_type_info_static(), 0},
    {ov::op::v0::Tile::get_type_info_static(), 0},
    {ov::op::v0::Unsqueeze::get_type_info_static(), 0},
    {ov::op::v0::Xor::get_type_info_static(), 0},
    // opset 2
    {ov::op::v0::MVN::get_type_info_static(), 0},
    {ov::op::v0::ReorgYolo::get_type_info_static(), 0},
    {ov::op::v0::ROIPooling::get_type_info_static(), 0},
    {ov::op::v0::Gelu::get_type_info_static(), 0},
    {ov::op::v1::BatchToSpace::get_type_info_static(), 1},
    {ov::op::v1::SpaceToBatch::get_type_info_static(), 1},
    // opset 3
    {ov::op::v0::RNNCell::get_type_info_static(), 0},
    {ov::op::v0::ShuffleChannels::get_type_info_static(), 0},
    // opset 4
    {ov::op::v3::Acosh::get_type_info_static(), 3},
    {ov::op::v3::Asinh::get_type_info_static(), 3},
    {ov::op::v3::Atanh::get_type_info_static(), 3},
};

// todo: reuse in summary
std::string get_node_version(const std::shared_ptr<ov::Node>& node, const std::string& postfix = "");
std::string get_node_version(const ov::NodeTypeInfo& node_type_info);

}  // namespace functional
}  // namespace test
}  // namespace ov
