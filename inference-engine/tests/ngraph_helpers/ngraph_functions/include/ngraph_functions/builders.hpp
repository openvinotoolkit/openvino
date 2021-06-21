// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>

#include "ngraph_functions/utils/data_utils.hpp"

namespace ngraph {
namespace builder {

ngraph::ParameterVector makeParams(const element::Type &type, const std::vector<std::vector<size_t>> &shapes);

ngraph::ParameterVector
makeParams(const element::Type &type, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs);

template<typename T>
std::shared_ptr<Node> makeConstant(const element::Type &type, const std::vector<size_t> &shape,
                                   const std::vector<T> &data, bool random = false,
                                   T upTo = 10, T startFrom = 1, const int seed = 1) {
    std::shared_ptr<ngraph::Node> weightsNode;

#define makeNode(TYPE) \
        case TYPE: \
            weightsNode = std::make_shared<ngraph::opset1::Constant>( \
                    type, shape, \
                    random ? NGraphFunctions::Utils::generateVector<TYPE>(ngraph::shape_size(shape), upTo, startFrom, seed) : \
                             NGraphFunctions::Utils::castVector<T, ngraph::helpers::nGraphTypesTrait<TYPE>::value_type >(data)); \
            break;
    switch (type) {
        case ngraph::element::Type_t::bf16:
            weightsNode = std::make_shared<ngraph::opset1::Constant>(
                    type, shape,
                    random ? NGraphFunctions::Utils::generateBF16Vector(ngraph::shape_size(shape), upTo, startFrom) :
                    NGraphFunctions::Utils::castVector<T, ngraph::bfloat16>(data));
            break;
        case ngraph::element::Type_t::f16:
            weightsNode = std::make_shared<ngraph::opset1::Constant>(
                    type, shape,
                    random ? NGraphFunctions::Utils::generateF16Vector(ngraph::shape_size(shape), upTo, startFrom) :
                    NGraphFunctions::Utils::castVector<T, ngraph::float16>(data));
            break;
        makeNode(ngraph::element::Type_t::f32);
        makeNode(ngraph::element::Type_t::f64);
        makeNode(ngraph::element::Type_t::i8);
        makeNode(ngraph::element::Type_t::i16);
        makeNode(ngraph::element::Type_t::i32);
        makeNode(ngraph::element::Type_t::i64);
        makeNode(ngraph::element::Type_t::u8);
        makeNode(ngraph::element::Type_t::u16);
        makeNode(ngraph::element::Type_t::u32);
        makeNode(ngraph::element::Type_t::u64);
        makeNode(ngraph::element::Type_t::boolean);
#undef makeNode
        default:
            throw std::runtime_error("Unhandled precision");
    }
    return weightsNode;
}

std::shared_ptr<ngraph::Node> makeInputLayer(const element::Type& type, ngraph::helpers::InputLayerType inputType,
                                             const std::vector<size_t>& shape);

std::shared_ptr<ngraph::Node> makeBroadcast(const ngraph::Output<Node> &in,
                                            const ngraph::Output<Node> &target_shape,
                                            const ngraph::op::BroadcastType& mode,
                                            const ngraph::AxisSet& axis_set = {});

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
                                                          const std::vector<ptrdiff_t> &outputPadding = {},
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
                                                          const std::vector<ptrdiff_t> &outputPadding = {},
                                                          const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                          const ngraph::Output<Node> &outputShape,
                                                          const element::Type &type,
                                                          const std::vector<size_t> &filterSize,
                                                          const std::vector<size_t> &strides,
                                                          const std::vector<ptrdiff_t> &padsBegin,
                                                          const std::vector<ptrdiff_t> &padsEnd,
                                                          const std::vector<size_t> &dilations,
                                                          const op::PadType &autoPad,
                                                          size_t numOutChannels,
                                                          bool addBiases = false,
                                                          const std::vector<ptrdiff_t> &outputPadding = {},
                                                          const std::vector<float> &filterWeights = {},
                                                          const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoder(
        const ngraph::Output<Node>& inputData,
        const bool mergeRepeated);

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoderSeqLen(
        const ngraph::Output<Node>& inputData,
        const ngraph::Output<Node>& sequenceLength,
        int blankIndex,
        bool mergeRepeated,
        const element::Type& idxPrecision = element::i32);

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoderSeqLen(
        const ngraph::Output<Node>& inputData,
        int blankIndex,
        bool mergeRepeated,
        const element::Type& idxPrecision = element::i32);

std::shared_ptr<ngraph::Node> makeCTCLoss(
        const ngraph::Output<Node>& logitsNode,
        std::vector<int>& logitsLength,
        std::vector<std::vector<int>>& labels,
        std::vector<int>& labelsLength,
        int blankIndex,
        const element::Type& fType,
        const element::Type& iType,
        const bool preprocessCollapseRepeated,
        const bool ctcMergeRepeated,
        const bool unique);

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
                                        int64_t axis);

std::shared_ptr<ngraph::Node> makeVariadicSplit(const ngraph::Output<Node> &in,
                                                const std::vector<size_t> numSplits,
                                                size_t axis);

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType,
                                             std::vector<size_t> inShape = {},
                                             std::vector<float> constantsValue = {});

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::ParameterVector &parameters,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType);

std::shared_ptr<ngraph::Node> makeEltwise(const ngraph::Output<Node> &in0,
                                          const ngraph::Output<Node> &in1,
                                          ngraph::helpers::EltwiseTypes eltwiseType);

std::shared_ptr<ngraph::Node> makeBatchToSpace(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<int64_t> &blockShape,
                                               const std::vector<int64_t> &cropsBegin,
                                               const std::vector<int64_t> &cropsEnd);

std::shared_ptr<ngraph::Node> makeSpaceToBatch(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<int64_t> &blockShape,
                                               const std::vector<int64_t> &padsBegin,
                                               const std::vector<int64_t> &padsEnd);

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

std::shared_ptr<ngraph::Node> makeMVN6(const Output<Node>& in,
                                       const Output<Node>& axesNode,
                                       bool normalizeVariance,
                                       float eps,
                                       std::string& epsMode);

std::shared_ptr<ngraph::Node> makeSqueezeUnsqueeze(const ngraph::Output<Node> &in,
                                                   const element::Type &type,
                                                   const std::vector<int> &squeeze_indices,
                                                   ngraph::helpers::SqueezeOpType opType);

std::shared_ptr<ngraph::Node> makeMinMax(const ngraph::Output<Node> &in1,
                                         const ngraph::Output<Node> &in2,
                                         ngraph::helpers::MinMaxOpType opType);

std::shared_ptr<ngraph::Node> makeProposal(const ngraph::Output<Node> &class_probs,
                                           const ngraph::Output<Node> &class_logits,
                                           const std::vector<float>& image_info,
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
                                       std::vector<size_t> constShapes,
                                       const int32_t  seed = 1);

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

std::shared_ptr<ngraph::Node> makeReduce(const ngraph::Output<Node>& data,
                                         const ngraph::Output<Node>& axes,
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

std::shared_ptr<Node> makeROIPooling(const Output<Node>& input,
                                     const Output<Node>& coords,
                                     const Shape& output_size,
                                     const float spatial_scale,
                                     const ngraph::helpers::ROIPoolingTypes& roi_pool_type);

std::shared_ptr<ngraph::Node> makeScatterUpdate(const ngraph::Output<Node> &in,
                                                const element::Type& indicesType,
                                                const std::vector<size_t>& indicesShape,
                                                const std::vector<int64_t>& indices,
                                                const ngraph::Output<Node> &update,
                                                int64_t axis);

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

std::shared_ptr<ngraph::Node> makeDetectionOutput(const ngraph::OutputVector &inputs,
                                                  const ngraph::op::DetectionOutputAttrs& attrs);

std::shared_ptr<ngraph::Node> makeFullyConnected(const ngraph::Output<Node>& in,
                                                 const element::Type& type,
                                                 const size_t outputSize,
                                                 bool addBias = true,
                                                 const ngraph::Shape& weightsShape = {},
                                                 const std::vector<float>& weights = {},
                                                 const std::vector<float>& biasWeights = {});

std::shared_ptr<ngraph::Node> makeConcat(const std::vector<ngraph::Output<Node>>& in,
                                         const int& axis);

std::shared_ptr<ngraph::Node> makePad(const ngraph::Output<Node>& data,
                                      const std::vector<int64_t>& padsBegin,
                                      const std::vector<int64_t>& padsEnd,
                                      float argPadValue,
                                      ngraph::helpers::PadMode padMode);

std::shared_ptr<ngraph::Node> makeBatchNormInference(const ngraph::Output<Node>& data,
                                                     double epsilon);

std::shared_ptr<ngraph::Node> makeLSTM(const OutputVector& in,
                                           const std::vector<ngraph::Shape>& constants,
                                           std::size_t hidden_size,
                                           const std::vector<std::string>& activations =
                                           std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                           const std::vector<float>& activations_alpha = {},
                                           const std::vector<float>& activations_beta = {},
                                           float clip = 0.f,
                                           bool make_sequence = false,
                                           ngraph::op::RecurrentSequenceDirection direction = ngraph::op::RecurrentSequenceDirection::FORWARD,
                                           ngraph::helpers::SequenceTestsMode mode = ngraph::helpers::SequenceTestsMode::PURE_SEQ);

std::shared_ptr<ngraph::Node> makeGRU(const OutputVector& in,
                                      const std::vector<ngraph::Shape>& constants,
                                      std::size_t hidden_size,
                                      const std::vector<std::string>& activations =
                                      std::vector<std::string>{"sigmoid", "tanh"},
                                      const std::vector<float>& activations_alpha = {},
                                      const std::vector<float>& activations_beta = {},
                                      float clip = 0.f,
                                      bool linear_before_reset = false,
                                      bool make_sequence = false,
                                      ngraph::op::RecurrentSequenceDirection direction = ngraph::op::RecurrentSequenceDirection::FORWARD,
                                      ngraph::helpers::SequenceTestsMode mode = ngraph::helpers::SequenceTestsMode::PURE_SEQ);

std::shared_ptr<ngraph::Node> makeRNN(const OutputVector& in,
                                      const std::vector<ngraph::Shape>& constants,
                                      std::size_t hidden_size,
                                      const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
                                      const std::vector<float>& activations_alpha = {},
                                      const std::vector<float>& activations_beta = {},
                                      float clip = 0.f,
                                      bool make_sequence = false,
                                      ngraph::op::RecurrentSequenceDirection direction = ngraph::op::RecurrentSequenceDirection::FORWARD,
                                      ngraph::helpers::SequenceTestsMode mode = ngraph::helpers::SequenceTestsMode::PURE_SEQ);

std::shared_ptr<ngraph::Node> makeGatherElements(
                                      const ngraph::Output<Node>& dataNode,
                                      const ngraph::Shape& indicesShape,
                                      const element::Type& indicesType,
                                      const int axis);

std::shared_ptr<ngraph::Node> makeGatherND(
                                      const ngraph::Output<Node>& dataNode,
                                      const ngraph::Shape& indicesShape,
                                      const element::Type& indicesType,
                                      const std::size_t batchDims);

std::shared_ptr<ngraph::Node> makeTile(const ngraph::Output<Node>& in,
                                       const std::vector<int64_t>& repeats);

std::shared_ptr<ngraph::Node> makeNormalizeL2(const ngraph::Output<Node>& data,
                                              const std::vector<int64_t>& axes,
                                              float eps,
                                              ngraph::op::EpsMode epsMode);

std::shared_ptr<ngraph::Node> makeNms(const ngraph::Output<Node> &boxes,
                                      const ngraph::Output<Node> &scores,
                                      const element::Type& maxBoxesPrec,
                                      const element::Type& thrPrec,
                                      const int32_t &maxOutBoxesPerClass,
                                      const float &iouThr,
                                      const float &scoreThr,
                                      const float &softNmsSigma,
                                      const ngraph::op::v5::NonMaxSuppression::BoxEncodingType &boxEncoding,
                                      const bool &sortResDescend,
                                      const ngraph::element::Type& outType);

std::shared_ptr<ngraph::Node> makeOneHot(const ngraph::Output<Node>& indices,
                                         const element::Type& depth_type,
                                         const int64_t& depth_val,
                                         const element::Type& set_type,
                                         const float& on_val,
                                         const float& off_val,
                                         const int64_t& axis);

std::shared_ptr<ngraph::Node> makeRoll(const ngraph::Output<Node>& dataNode,
                                       const ngraph::Output<Node>& shiftNode,
                                       const ngraph::Output<Node>& axesNode);

std::shared_ptr<ngraph::Node> makeDFT(const ngraph::Output<Node> &dataNode,
                                      const std::vector<int64_t> &axes,
                                      const std::vector<int64_t> &signalSize,
                                      const ngraph::helpers::DFTOpType opType);

}  // namespace builder
}  // namespace ngraph
