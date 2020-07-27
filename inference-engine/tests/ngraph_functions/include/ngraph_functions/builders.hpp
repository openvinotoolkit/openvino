// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>

#include "ngraph_functions/utils/data_utils.hpp"

namespace ngraph {
namespace builder {

ngraph::ParameterVector makeParams(const element::Type &type, const std::vector<std::vector<size_t>> &shapes);

ngraph::ParameterVector
makeParams(const element::Type &type, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs);

std::shared_ptr<ngraph::Node> makeConstant(const element::Type &type, const std::vector<size_t> &shape,
                                           const std::vector<float> &data, bool random = false);

std::shared_ptr<ngraph::Node> makeInputLayer(const element::Type& type, ngraph::helpers::InputLayerType inputType,
                                             const std::vector<size_t>& shape);

std::shared_ptr<ngraph::Node> makeConvolution(const ngraph::Output<Node> &in,
                                              const element::Type &type,
                                              const std::vector<size_t> &filterSize,
                                              const std::vector<size_t> &strides,
                                              const std::vector<ptrdiff_t> &padsBegin,
                                              const std::vector<ptrdiff_t> &padsEnd,
                                              const std::vector<size_t> &dilations,
                                              const op::PadType &autoPad,
                                              size_t numOutChannels,
                                              bool addBiases = false,
                                              const std::vector<float> &filterWeights = {},
                                              const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolution(const ngraph::Output<Node> &in,
                                                   const element::Type &type,
                                                   const std::vector<size_t> &filterSize,
                                                   const std::vector<size_t> &strides,
                                                   const std::vector<ptrdiff_t> &padsBegin,
                                                   const std::vector<ptrdiff_t> &padsEnd,
                                                   const std::vector<size_t> &dilations,
                                                   const op::PadType &autoPad,
                                                   size_t numOutChannels,
                                                   size_t numGroups,
                                                   bool addBiases = false,
                                                   const std::vector<float> &filterWeights = {},
                                                   const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolution(const ngraph::Output<Node> &in,
                                                   const ngraph::Output<Node> &weights,
                                                   const element::Type &type,
                                                   const std::vector<size_t> &strides,
                                                   const std::vector<ptrdiff_t> &padsBegin,
                                                   const std::vector<ptrdiff_t> &padsEnd,
                                                   const std::vector<size_t> &dilations,
                                                   const op::PadType &autoPad,
                                                   bool addBiases = false,
                                                   const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                          const element::Type &type,
                                                          const std::vector<size_t> &filterSize,
                                                          const std::vector<size_t> &strides,
                                                          const std::vector<ptrdiff_t> &padsBegin,
                                                          const std::vector<ptrdiff_t> &padsEnd,
                                                          const std::vector<size_t> &dilations,
                                                          const op::PadType &autoPad,
                                                          size_t numOutChannels,
                                                          bool addBiases = false,
                                                          const std::vector<float> &filterWeights = {},
                                                          const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                          const ngraph::Output<Node> &weights,
                                                          const element::Type &type,
                                                          const std::vector<size_t> &strides,
                                                          const std::vector<ptrdiff_t> &padsBegin,
                                                          const std::vector<ptrdiff_t> &padsEnd,
                                                          const std::vector<size_t> &dilations,
                                                          const op::PadType &autoPad,
                                                          bool addBiases = false,
                                                          const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                               const element::Type &type,
                                                               const std::vector<size_t> &filterSize,
                                                               const std::vector<size_t> &strides,
                                                               const std::vector<ptrdiff_t> &padsBegin,
                                                               const std::vector<ptrdiff_t> &padsEnd,
                                                               const std::vector<size_t> &dilations,
                                                               const op::PadType &autoPad,
                                                               size_t numOutChannels,
                                                               size_t numGroups,
                                                               bool addBiases = false,
                                                               const std::vector<float> &filterWeights = {},
                                                               const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                               const ngraph::Output<Node> &weights,
                                                               const element::Type &type,
                                                               const std::vector<size_t> &strides,
                                                               const std::vector<ptrdiff_t> &padsBegin,
                                                               const std::vector<ptrdiff_t> &padsEnd,
                                                               const std::vector<size_t> &dilations,
                                                               const op::PadType &autoPad,
                                                               bool addBiases = false,
                                                               const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeBinaryConvolution(const ngraph::Output<Node> &in,
                                                    const std::vector<size_t> &filterSize,
                                                    const std::vector<size_t> &strides,
                                                    const std::vector<ptrdiff_t> &padsBegin,
                                                    const std::vector<ptrdiff_t> &padsEnd,
                                                    const std::vector<size_t> &dilations,
                                                    const op::PadType &autoPad,
                                                    size_t numOutChannels,
                                                    float padValue,
                                                    const std::vector<int8_t> &filterWeihgts = {});

std::shared_ptr<ngraph::Node> makeSplit(const ngraph::Output<Node> &in,
                                        const element::Type &type,
                                        size_t numSplits,
                                        size_t axis);

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType,
                                             std::vector<size_t> inShape = {});

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::ParameterVector &parameters,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType);

std::shared_ptr<ngraph::Node> makeEltwise(const ngraph::Output<Node> &in0,
                                          const ngraph::Output<Node> &in1,
                                          ngraph::helpers::EltwiseTypes eltwiseType);

std::shared_ptr<ngraph::Node> makeBatchToSpace(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<size_t> &blockShape,
                                               const std::vector<size_t> &cropsBegin,
                                               const std::vector<size_t> &cropsEnd);

std::shared_ptr<ngraph::Node> makeSpaceToBatch(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<size_t> &blockShape,
                                               const std::vector<size_t> &padsBegin,
                                               const std::vector<size_t> &padsEnd);

std::shared_ptr<ngraph::Node> makeStridedSlice(const ngraph::Output<Node> &in,
                                               const std::vector<int64_t> &begin,
                                               const std::vector<int64_t> &end,
                                               const std::vector<int64_t> &stride,
                                               const element::Type &type,
                                               const std::vector<int64_t> &begin_mask,
                                               const std::vector<int64_t> &end_mask,
                                               const std::vector<int64_t> &new_axis_mask = std::vector<int64_t>{},
                                               const std::vector<int64_t> &shrink_mask = std::vector<int64_t>{},
                                               const std::vector<int64_t> &ellipsis_mask = std::vector<int64_t>{});

std::shared_ptr<ngraph::Node> makeMVN(const ngraph::Output<Node> &in,
                                      bool acrossChannels,
                                      bool normalizeVariance,
                                      double eps);

std::shared_ptr<ngraph::Node> makeSqueezeUnsqueeze(const ngraph::Output<Node> &in,
                                                   const element::Type &type,
                                                   const std::vector<int> &squeeze_indices,
                                                   ngraph::helpers::SqueezeOpType opType);

std::shared_ptr<ngraph::Node> makeMinMax(const ngraph::Output<Node> &in1,
                                         const ngraph::Output<Node> &in2,
                                         ngraph::helpers::MinMaxOpType opType);

std::shared_ptr<ngraph::Node> makeProposal(const ngraph::Output<Node> &class_probs,
                                           const ngraph::Output<Node> &class_logits,
                                           const ngraph::Output<Node> &image_shape,
                                           const element::Type &type,
                                           size_t base_size,
                                           size_t pre_nms_topn,
                                           size_t post_nms_topn,
                                           float nms_thresh,
                                           size_t feat_stride,
                                           size_t min_size,
                                           const std::vector<float> &ratio,
                                           const std::vector<float> &scale,
                                           bool clip_before_nms,
                                           bool clip_after_nms,
                                           bool normalize,
                                           float box_size_scale,
                                           float box_coordinate_scale,
                                           std::string framework);

std::shared_ptr<ngraph::Node> makeSelect(std::vector<ngraph::Output<Node>> &in,
                                         const ngraph::op::AutoBroadcastSpec &auto_broadcast);

std::shared_ptr<Node> makeFakeQuantize(const ngraph::Output<Node> &in,
                                       const element::Type &type,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes,
                                       const std::vector<float> &inputLowData,
                                       const std::vector<float> &inputHighData,
                                       const std::vector<float> &outputLowData,
                                       const std::vector<float> &outputHighData);

std::shared_ptr<Node> makeFakeQuantize(const ngraph::Output<Node> &in,
                                       const element::Type &type,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes);

std::shared_ptr<ngraph::Node> makeCumSum(const ngraph::Output<Node> &in,
                                         const ngraph::Output<Node> &axis,
                                         bool exclusive,
                                         bool reverse);

std::shared_ptr<ngraph::Node> makeEmbeddingBagOffsetsSum(
        const element::Type &dataType,
        const element::Type &indicesType,
        const ngraph::Output<Node> &emb_table_node,
        const std::vector<size_t> &indices,
        const std::vector<size_t> &offsets,
        size_t default_index,
        bool with_weights,
        bool with_default_index);

std::shared_ptr<ngraph::Node> makeEmbeddingBagPackedSum(
        const element::Type &dataType,
        const element::Type &indicesType,
        const ngraph::Output<Node> &emb_table_node,
        const std::vector<std::vector<size_t>> &indices,
        bool with_weights);

std::shared_ptr<ngraph::Node> makeEmbeddingSegmentsSum(
        const element::Type &dataType,
        const element::Type &indicesType,
        const ngraph::Output<Node> &emb_table_node,
        const std::vector<size_t> &indices,
        const std::vector<size_t> &segment_ids,
        size_t num_segments,
        size_t default_index,
        bool with_weights,
        bool with_default_index);

std::shared_ptr<ngraph::Node> makeDepthToSpace(const ngraph::Output<Node> &in,
                                               ngraph::opset3::DepthToSpace::DepthToSpaceMode mode,
                                               size_t blockSize);

std::shared_ptr<ngraph::Node> makeSpaceToDepth(const ngraph::Output<Node> &in,
                                               ngraph::opset3::SpaceToDepth::SpaceToDepthMode mode,
                                               size_t blockSize);

std::shared_ptr<Node> makeShuffleChannels(const ngraph::Output<Node> &in,
                                          int axis,
                                          int group);

std::shared_ptr<Node> makeMatMul(const Output<Node> &A,
                                 const Output<Node> &B,
                                 bool transpose_a = false,
                                 bool transpose_b = false);

std::shared_ptr<ngraph::Node> makeReduce(std::vector<ngraph::Output<Node>> &in,
                                         const std::vector<int> &reductionAxes,
                                         bool keepDims,
                                         ngraph::helpers::ReductionType reductionType);

std::shared_ptr<Node> makePooling(const ngraph::Output<Node> &in,
                                  const std::vector<size_t> &strides,
                                  const std::vector<size_t> &padsBegin,
                                  const std::vector<size_t> &padsEnd,
                                  const std::vector<size_t> &kernel,
                                  const op::RoundingType &roundingType,
                                  const op::PadType &padType,
                                  bool excludePad,
                                  const ngraph::helpers::PoolingTypes &poolType);

std::shared_ptr<ngraph::Node> makeScatterUpdate(const ngraph::Output<Node> &in,
                                                const element::Type& indicesType,
                                                const std::vector<size_t>& indicesShape,
                                                const std::vector<size_t>& indices,
                                                const ngraph::Output<Node> &update,
                                                std::size_t axis);

std::shared_ptr<ngraph::Node> makeScatterElementsUpdate(const ngraph::Output<Node> &in,
                                                        const element::Type& indicesType,
                                                        const std::vector<size_t>& indicesShape,
                                                        const std::vector<size_t>& indices,
                                                        const ngraph::Output<Node> &update,
                                                        int axis);

std::shared_ptr<ngraph::Node> makeScatterNDUpdate(const ngraph::Output<Node> &in,
                                                  const element::Type& indicesType,
                                                  const std::vector<size_t>& indicesShape,
                                                  const std::vector<size_t>& indices,
                                                  const ngraph::Output<Node> &update);

std::shared_ptr<ngraph::Node> makeComparison(const ngraph::Output<Node> &in0,
                                             const ngraph::Output<Node> &in1,
                                             ngraph::helpers::ComparisonTypes comparisonType);

std::shared_ptr<ngraph::Node> makeLogical(const ngraph::Output<Node> &in0,
                                          const ngraph::Output<Node> &in1,
                                          ngraph::helpers::LogicalTypes logicalType);

}  // namespace builder
}  // namespace ngraph
