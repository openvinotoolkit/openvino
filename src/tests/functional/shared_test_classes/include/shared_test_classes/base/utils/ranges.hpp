// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/type_ranges.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/ops.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

namespace ov {
namespace test {
namespace utils {

// NOTE: Default ranges are collected by data type and have resolution 1(for real types too)
// to set up correct ranges and resolutions, please, configure range for Op in inputRanges structure
struct Range {
    std::vector<ov::test::utils::InputGenerateData> int_port_ranges;
    std::vector<ov::test::utils::InputGenerateData> real_port_ranges;

    Range(const std::vector<ov::test::utils::InputGenerateData>& int_ranges = {},
          const std::vector<ov::test::utils::InputGenerateData>& real_ranges = {})
        : int_port_ranges(int_ranges),
          real_port_ranges(real_ranges) {
        size_t max_known_port = std::max(real_port_ranges.size(), int_port_ranges.size());
        max_known_port = std::max(static_cast<int>(max_known_port), 1);
        for (size_t port = 0; port < max_known_port; port++) {
            std::map<ov::element::Type, ov::test::utils::InputGenerateData> type_map;
            for (const auto& type : get_known_types()) {
                ov::test::utils::InputGenerateData new_range = rangeByType.get_range(type);
                if (type.is_real() && port < real_port_ranges.size()) {
                    new_range.correct_range(real_port_ranges.at(port));
                    new_range.input_attribute = real_port_ranges.at(port).input_attribute;
                } else if (type.is_integral() && port < int_port_ranges.size()) {
                    new_range.correct_range(int_port_ranges.at(port));
                    new_range.input_attribute = int_port_ranges.at(port).input_attribute;
                }
                type_map[type] = new_range;
            }
            data.push_back(type_map);
        }
    }

    std::vector<std::map<ov::element::Type, ov::test::utils::InputGenerateData>> data;

    ov::test::utils::InputGenerateData get_data(size_t port, ov::element::Type type) {
        if (port < data.size()) {
            return data.at(port).at(type);
        } else {
            return data.at(0).at(type);
        }
    }
};

static std::map<ov::NodeTypeInfo, Range> inputRanges = {
    {ov::op::v0::Erf::get_type_info_static(), Range({{-3, 6}}, {{-3, 6, 10}})},
    {ov::op::v1::Divide::get_type_info_static(), Range({{101, 100}}, {{2, 2, 128}})},
    {ov::op::v1::FloorMod::get_type_info_static(), Range({{2, 4}}, {{2, 2, 128}})},
    {ov::op::v1::Mod::get_type_info_static(), Range({{2, 4}}, {{2, 2, 128}})},
    {ov::op::v1::ReduceMax::get_type_info_static(), Range({{0, 5}}, {{-5, 5, 1000}})},
    {ov::op::v1::ReduceMean::get_type_info_static(), Range({{0, 5, 1000}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceMin::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceProd::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceSum::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceSum::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::ReduceSum::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v1::Power::get_type_info_static(), Range({{2, 4}}, {{2, 2, 128}})},
    {ov::op::v4::Proposal::get_type_info_static(), Range({{0, 255, 1, 8234231}}, {{0, 1, 1000, 8234231}})},
    {ov::op::v4::ReduceL1::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v4::ReduceL2::get_type_info_static(), Range({{0, 5}}, {{0, 5, 1000}})},
    {ov::op::v7::DFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v9::RDFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v1::LogicalAnd::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v1::LogicalOr::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v1::LogicalNot::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v1::LogicalXor::get_type_info_static(), Range({{0, 2}}, {{0, 2, 1}})},
    {ov::op::v7::IDFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v9::IRDFT::get_type_info_static(), Range({{0, 1}}, {{0, 1, 1000000}})},
    {ov::op::v0::Sigmoid::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Tanh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Relu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::PRelu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Exp::get_type_info_static(), Range({{0, 15}}, {{-10, 20, 32768}})},
    {ov::op::v0::Log::get_type_info_static(), Range({{0, 15}}, {{1, 20, 32768}})},
    {ov::op::v0::Sign::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Abs::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Clamp::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Negative::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Acos::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v3::Acosh::get_type_info_static(), Range({{1, 15}}, {{1, 200, 32768}})},
    {ov::op::v0::Asin::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v3::Asinh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Atan::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v3::Atanh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Cos::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Cosh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Floor::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sin::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sinh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sqrt::get_type_info_static(), Range({{0, 15}}, {{1, 20, 32768}})},
    {ov::op::v0::Tan::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Elu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Erf::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::HardSigmoid::get_type_info_static(),
     Range({{0, 15}}, {{-1, 2, 32768}, {0.2, 0, 1, 1, true}, {0.5, 0, 1, 1, true}})},
    {ov::op::v0::Selu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sigmoid::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Tanh::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Relu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Exp::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Log::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Sign::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Abs::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Gelu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v0::Ceiling::get_type_info_static(), Range({{0, 15}}, {{-1000, 2000, 32768}})},
    {ov::op::v4::Mish::get_type_info_static(), Range({{0, 15}}, {{-10, 60, 32768}})},
    {ov::op::v4::HSwish::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v4::SoftPlus::get_type_info_static(), Range({{0, 15}}, {{-100, 200, 32768}})},
    {ov::op::v4::Swish::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v5::HSigmoid::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v5::Round::get_type_info_static(), Range({{0, 15}}, {{-10, 20, 4}})},
    {ov::op::v7::Gelu::get_type_info_static(), Range({{0, 15}}, {{-1, 2, 32768}})},
    {ov::op::v14::MaxPool::get_type_info_static(), Range({{0, 10, 1, 1}}, {{0, 10, 1, 1}})},
    {ov::op::v8::MaxPool::get_type_info_static(), Range({{0, 10, 1, 1}}, {{0, 10, 1, 1}})},
    {ov::op::v1::MaxPool::get_type_info_static(), Range({{0, 10, 1, 1}}, {{0, 10, 1, 1}})},
    {ov::op::v1::AvgPool::get_type_info_static(), Range({{0, 10, 1, 1}}, {{0, 10, 1, 1}})},
    {ov::op::v14::AvgPool::get_type_info_static(), Range({{0, 10, 1, 1}}, {{0, 10, 1, 1}})},
    {ov::op::v9::SoftSign::get_type_info_static(), Range({{0, 15}}, {{-100, 200, 32768}})},
    // new temp
    {ov::op::v1::Convolution::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::ConvolutionBackpropData::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::GroupConvolution::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::GroupConvolutionBackpropData::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v12::ScatterElementsUpdate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::ScatterUpdate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::Unsqueeze::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::RegionYolo::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::MatMul::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v11::Interpolate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v4::Interpolate::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::LRN::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Pad::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::Broadcast::get_type_info_static(), Range({{0, 200}, {0, 10, 1, 1, true}, {0, 10, 1, 1, true}}, {{0, 2000, 32768}})},
    {ov::op::v5::NonMaxSuppression::get_type_info_static(),
        Range({{0, 15}, {0, 1, 1000, 1, true}}, {{0, 8, 32}, {0, 1, 1000, 1, true}})},
    {ov::op::v9::NonMaxSuppression::get_type_info_static(),
        Range({{0, 15}, {0, 1, 1000, 1, true}}, {{0, 8, 32}, {0, 1, 1000, 1, true}})},
    {ov::op::v8::MatrixNms::get_type_info_static(),
        Range({{0, 15}, {0, 1, 1000, 1, true}}, {{0, 8, 32}, {0, 1, 1000, 1, true}})},
    {ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::get_type_info_static(),
        Range({{1, 0, 1, 1}}, {{1, 0, 1, 1}})},
    {ov::op::v6::ExperimentalDetectronPriorGridGenerator::get_type_info_static(),
        Range({{0, 0, 1}}, {{-100, 200, 2, 1}, {0, 0, 1, 1, true}, {0, 0, 1, 1, true}})},
    {ov::op::v8::DeformableConvolution::get_type_info_static(),
        Range({{0, 15}, {0, 2, 10, 1, true}, {0, 1, 20, 1, true}},
              {{0, 8, 32}, {0, 2, 10, 1, true}, {0, 1, 20, 1, true}})},
    {ov::op::v5::GRUSequence::get_type_info_static(), Range({{0, 15}, {0, 15}, {0, 10, 1, 1, true}}, {{0, 8, 32}})},
    {ov::op::v5::BatchNormInference::get_type_info_static(), Range({{0, 3}}, {{0, 3, 1}})},
    {ov::op::v5::RNNSequence::get_type_info_static(),
        Range({{0, 15}, {0, 15}, {0, 10, 1, 1, true}}, {{0, 8, 32}, {0, 8, 32}, {0, 10, 1, 1, true}})},
    {ov::op::v1::LogicalAnd::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::LogicalNot::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::LogicalOr::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::LogicalXor::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::ReduceLogicalAnd::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::ReduceLogicalOr::get_type_info_static(), Range({{0, 2}}, {{0, 2}})},
    {ov::op::v1::Reshape::get_type_info_static(), Range({{-1000, 2000}, {0, 256, 1, 1, true}}, {{-100, 200, 32768}})},
    {ov::op::v3::TopK::get_type_info_static(), Range({{-1000, 2000}, {0, 1000, 1, 1, true}}, {{-1000, 2000, 32768}})},
    {ov::op::v11::TopK::get_type_info_static(), Range({{-1000, 2000}, {0, 1000, 1, 1, true}}, {{-1000, 2000, 32768}})},
    {ov::op::v4::Range::get_type_info_static(),
        Range({{0, 15}, {1, 1000, 1, 1, true}}, {{-1000, 2000, 32768}, {1, 1000, 1, 1, true}})},
    {ov::op::v3::ROIAlign::get_type_info_static(),
        Range({{0, 15}, {0, 1000, 1, 1, true}, {0, 1000, 1, 1, true}},
              {{-1000, 2000, 32768}, {0, 1000, 1, 1, true}, {0, 1000, 1, 1, true}})},
    {ov::op::v9::ROIAlign::get_type_info_static(),
        Range({{0, 15}, {0, 1000, 1, 1, true}, {0, 1000, 1, 1, true}},
              {{-1000, 2000, 32768}, {0, 1000, 1, 1, true}, {0, 1000, 1, 1, true}})},
    {ov::op::v0::Convert::get_type_info_static(), Range({{0, 1000}}, {{-100, 200, 32768}})},
    {ov::op::v0::FakeQuantize::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::FakeQuantize::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Select::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Multiply::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::StridedSlice::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v5::LSTMSequence::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::VariadicSplit::get_type_info_static(), Range({{0, 10}}, {{0, 8, 32}})},
    {ov::op::v1::Subtract::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::SpaceToBatch::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::GatherND::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::Gather::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::DepthToSpace::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v7::Einsum::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::RandomUniform::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v9::Eye::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::CumSum::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::MVN::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v6::MVN::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::GRUCell::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v5::GRUSequence::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::If::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::TensorIterator::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v12::GroupNormalization::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::ReverseSequence::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::GatherTree::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::DeformablePSROIPooling::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Softmax::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::Softmax::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::PSROIPooling::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::internal::AUGRUSequence::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::internal::AUGRUCell::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v7::Roll::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v4::LSTMCell::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v13::ScaledDotProductAttention::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Transpose::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v5::Loop::get_type_info_static(), Range({{1, 10, 1, 1, true}, {0, 2, 1, 1, true}, {0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::SquaredDifference::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v4::CTCLoss::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v9::GridSample::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v13::Multinomial::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::EmbeddingBagOffsetsSum::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v15::EmbeddingBagOffsets::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v9::GenerateProposals::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::ROIPooling::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::ShuffleChannels::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v8::Slice::get_type_info_static(),
        Range({{0, 15}, {0, 15, 1, 1, true}, {0, 15, 1, 1, true}, {1, 5, 1, 1, true}, {0, 15, 1, 1, true}}, {{0, 8, 32}})},
    {ov::op::v3::EmbeddingBagPackedSum::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v3::EmbeddingSegmentsSum::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v15::EmbeddingBagPacked::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v0::GRN::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::Add::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v15::ROIAlignRotated::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v1::BatchToSpace::get_type_info_static(), Range({{0, 15}}, {{0, 8, 32}})},
    {ov::op::v15::BitwiseLeftShift::get_type_info_static(), Range({{0, 5}, {0, 4}}, {})},
    {ov::op::v15::BitwiseRightShift::get_type_info_static(), Range({{0, 5}, {0, 4}}, {})},
    {ov::op::v15::STFT::get_type_info_static(), Range({{16, 24}, {1, 16}}, {{0, 1, 10000}, {0, 1, 10000}})},
    {ov::op::v16::ISTFT::get_type_info_static(), Range({{}, {}, {16, 0, 1}, {4, 0, 1}, {64, 0, 1}}, {{0, 1, 10000}, {0, 1, 10000}})},
};

class ModelRange {
    // key for map calculated in get_range_id and contais [Parameter Name]_[parameter type]
    std::map<std::string, std::shared_ptr<ov::test::utils::InputGenerateData>> node_ranges;

public:
    void find_mode_ranges(const std::shared_ptr<ov::Model>& function);
    std::string get_range_id(const std::shared_ptr<ov::Node>& node);
    ov::Tensor generate_input(std::shared_ptr<ov::Node> node, size_t port, const ov::Shape& targetShape);

    const std::shared_ptr<ov::test::utils::InputGenerateData> get_range_for_param(
        const std::shared_ptr<ov::Node>& node);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
