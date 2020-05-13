// Copyright (C) 2019 Intel Corporation
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
ngraph::ParameterVector makeParams(const element::Type &type, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs);

std::shared_ptr<ngraph::Node> makeConstant(const element::Type &type, const std::vector<size_t> &shape,
                                           const std::vector<float> &data, bool random = false);

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

std::shared_ptr<ngraph::Node> makeSplit(const ngraph::Output<Node> &in,
                                        const element::Type &type,
                                        size_t numSplits,
                                        size_t axis);

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType);

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

std::shared_ptr<ngraph::Node> makeSqueeze(const ngraph::Output<Node> &in,
                                          const element::Type &type,
                                          const std::vector<size_t> &squeeze_indices);

std::shared_ptr<ngraph::Node> makeUnsqueeze(const ngraph::Output<Node> &in,
                                            const element::Type &type,
                                            const std::vector<size_t> &squeeze_indices);

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
}  // namespace builder
}  // namespace ngraph
