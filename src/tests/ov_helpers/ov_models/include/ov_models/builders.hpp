// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

// TODO: Temporary solution to fix compilation of plugin tests
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ov_models/utils/ov_helpers.hpp>
// TODO: Temporary solution to fix compilation of plugin tests

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "ov_models/utils/data_utils.hpp"

namespace ngraph {
namespace builder {

template <typename T>
std::shared_ptr<ov::Node> makeConstant(const ov::element::Type& type,
                                       const std::vector<size_t>& shape,
                                       const std::vector<T>& data,
                                       bool random = false,
                                       T upTo = 10,
                                       T startFrom = 1,
                                       const int seed = 1) {
    std::shared_ptr<ov::Node> weightsNode;

#define makeNode(TYPE)                                                                                               \
    case TYPE:                                                                                                       \
        weightsNode = std::make_shared<ov::op::v0::Constant>(                                                        \
            type,                                                                                                    \
            shape,                                                                                                   \
            random                                                                                                   \
                ? NGraphFunctions::Utils::generateVector<TYPE>(ov::shape_size(shape),                                \
                                                               ov::element_type_traits<TYPE>::value_type(upTo),      \
                                                               ov::element_type_traits<TYPE>::value_type(startFrom), \
                                                               seed)                                                 \
                : NGraphFunctions::Utils::castVector<T, ov::element_type_traits<TYPE>::value_type>(data));           \
        break;
    switch (type) {
        makeNode(ov::element::Type_t::bf16);
        makeNode(ov::element::Type_t::f16);
        makeNode(ov::element::Type_t::f32);
        makeNode(ov::element::Type_t::f64);
        makeNode(ov::element::Type_t::i8);
        makeNode(ov::element::Type_t::i16);
        makeNode(ov::element::Type_t::i32);
        makeNode(ov::element::Type_t::i64);
        makeNode(ov::element::Type_t::u8);
        makeNode(ov::element::Type_t::u16);
        makeNode(ov::element::Type_t::u32);
        makeNode(ov::element::Type_t::u64);
        makeNode(ov::element::Type_t::boolean);
        makeNode(ov::element::Type_t::nf4);
        makeNode(ov::element::Type_t::u4);
        makeNode(ov::element::Type_t::i4);
#undef makeNode
    default:
        throw std::runtime_error("Unhandled precision");
    }
    return weightsNode;
}

std::shared_ptr<ov::Node> makeInputLayer(const element::Type& type,
                                         ov::test::utils::InputLayerType inputType,
                                         const std::vector<size_t>& shape);

std::shared_ptr<ov::Node> makeDynamicInputLayer(const element::Type& type,
                                                ov::test::utils::InputLayerType inputType,
                                                const ov::PartialShape& shape);

std::shared_ptr<ov::Node> makeBroadcast(const ov::Output<Node>& in,
                                        const ov::Output<Node>& target_shape,
                                        const ov::op::BroadcastType& mode,
                                        const ov::AxisSet& axis_set = {});

std::shared_ptr<ov::Node> makeConvolution(const ov::Output<Node>& in,
                                          const element::Type& type,
                                          const std::vector<size_t>& filterSize,
                                          const std::vector<size_t>& strides,
                                          const std::vector<ptrdiff_t>& padsBegin,
                                          const std::vector<ptrdiff_t>& padsEnd,
                                          const std::vector<size_t>& dilations,
                                          const op::PadType& autoPad,
                                          size_t numOutChannels,
                                          bool addBiases = false,
                                          const std::vector<float>& filterWeights = {},
                                          const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeConvolution(const ov::Output<Node>& in_data,
                                          const ov::Output<Node>& in_weights,
                                          const element::Type& type,
                                          const std::vector<size_t>& filterSize,
                                          const std::vector<size_t>& strides,
                                          const std::vector<ptrdiff_t>& padsBegin,
                                          const std::vector<ptrdiff_t>& padsEnd,
                                          const std::vector<size_t>& dilations,
                                          const op::PadType& autoPad,
                                          size_t numOutChannels,
                                          bool addBiases = false,
                                          const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeGroupConvolution(const ov::Output<Node>& in,
                                               const element::Type& type,
                                               const std::vector<size_t>& filterSize,
                                               const std::vector<size_t>& strides,
                                               const std::vector<ptrdiff_t>& padsBegin,
                                               const std::vector<ptrdiff_t>& padsEnd,
                                               const std::vector<size_t>& dilations,
                                               const op::PadType& autoPad,
                                               size_t numOutChannels,
                                               size_t numGroups,
                                               bool addBiases = false,
                                               const std::vector<float>& filterWeights = {},
                                               const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeGroupConvolution(const ov::Output<Node>& in,
                                               const ov::Output<Node>& weights,
                                               const element::Type& type,
                                               const std::vector<size_t>& strides,
                                               const std::vector<ptrdiff_t>& padsBegin,
                                               const std::vector<ptrdiff_t>& padsEnd,
                                               const std::vector<size_t>& dilations,
                                               const op::PadType& autoPad,
                                               bool addBiases = false,
                                               const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeConvolutionBackpropData(const ov::Output<Node>& in,
                                                      const element::Type& type,
                                                      const std::vector<size_t>& filterSize,
                                                      const std::vector<size_t>& strides,
                                                      const std::vector<ptrdiff_t>& padsBegin,
                                                      const std::vector<ptrdiff_t>& padsEnd,
                                                      const std::vector<size_t>& dilations,
                                                      const op::PadType& autoPad,
                                                      size_t numOutChannels,
                                                      bool addBiases = false,
                                                      const std::vector<ptrdiff_t>& outputPadding = {},
                                                      const std::vector<float>& filterWeights = {},
                                                      const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeConvolutionBackpropData(const ov::Output<Node>& in,
                                                      const ov::Output<Node>& weights,
                                                      const element::Type& type,
                                                      const std::vector<size_t>& strides,
                                                      const std::vector<ptrdiff_t>& padsBegin,
                                                      const std::vector<ptrdiff_t>& padsEnd,
                                                      const std::vector<size_t>& dilations,
                                                      const op::PadType& autoPad,
                                                      bool addBiases = false,
                                                      const std::vector<ptrdiff_t>& outputPadding = {},
                                                      const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeConvolutionBackpropData(const ov::Output<Node>& in,
                                                      const ov::Output<Node>& outputShape,
                                                      const element::Type& type,
                                                      const std::vector<size_t>& filterSize,
                                                      const std::vector<size_t>& strides,
                                                      const std::vector<ptrdiff_t>& padsBegin,
                                                      const std::vector<ptrdiff_t>& padsEnd,
                                                      const std::vector<size_t>& dilations,
                                                      const op::PadType& autoPad,
                                                      size_t numOutChannels,
                                                      bool addBiases = false,
                                                      const std::vector<ptrdiff_t>& outputPadding = {},
                                                      const std::vector<float>& filterWeights = {},
                                                      const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeCTCGreedyDecoder(const ov::Output<Node>& inputData, const bool mergeRepeated);

std::shared_ptr<ov::Node> makeCTCGreedyDecoderSeqLen(const ov::Output<Node>& inputData,
                                                     const ov::Output<Node>& sequenceLength,
                                                     int blankIndex,
                                                     bool mergeRepeated,
                                                     const element::Type& idxPrecision = element::i32);

std::shared_ptr<ov::Node> makeCTCGreedyDecoderSeqLen(const ov::Output<Node>& inputData,
                                                     int blankIndex,
                                                     bool mergeRepeated,
                                                     const element::Type& idxPrecision = element::i32);

std::shared_ptr<ov::Node> makeCTCLoss(const ov::Output<Node>& logitsNode,
                                      std::vector<int>& logitsLength,
                                      std::vector<std::vector<int>>& labels,
                                      std::vector<int>& labelsLength,
                                      int blankIndex,
                                      const element::Type& fType,
                                      const element::Type& iType,
                                      const bool preprocessCollapseRepeated,
                                      const bool ctcMergeRepeated,
                                      const bool unique);

std::shared_ptr<ov::Node> makeGroupConvolutionBackpropData(const ov::Output<Node>& in,
                                                           const element::Type& type,
                                                           const std::vector<size_t>& filterSize,
                                                           const std::vector<size_t>& strides,
                                                           const std::vector<ptrdiff_t>& padsBegin,
                                                           const std::vector<ptrdiff_t>& padsEnd,
                                                           const std::vector<size_t>& dilations,
                                                           const op::PadType& autoPad,
                                                           size_t numOutChannels,
                                                           size_t numGroups,
                                                           bool addBiases = false,
                                                           const std::vector<ptrdiff_t>& outputPadding = {},
                                                           const std::vector<float>& filterWeights = {},
                                                           const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeGroupConvolutionBackpropData(const ov::Output<Node>& in,
                                                           const ov::Output<Node>& weights,
                                                           const element::Type& type,
                                                           const std::vector<size_t>& strides,
                                                           const std::vector<ptrdiff_t>& padsBegin,
                                                           const std::vector<ptrdiff_t>& padsEnd,
                                                           const std::vector<size_t>& dilations,
                                                           const op::PadType& autoPad,
                                                           bool addBiases = false,
                                                           const std::vector<ptrdiff_t>& outputPadding = {},
                                                           const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeGroupConvolutionBackpropData(const ov::Output<Node>& in,
                                                           const ov::Output<Node>& outputShape,
                                                           const element::Type& type,
                                                           const std::vector<size_t>& filterSize,
                                                           const std::vector<size_t>& strides,
                                                           const std::vector<ptrdiff_t>& padsBegin,
                                                           const std::vector<ptrdiff_t>& padsEnd,
                                                           const std::vector<size_t>& dilations,
                                                           const op::PadType& autoPad,
                                                           size_t numOutChannels,
                                                           size_t numGroups,
                                                           bool addBiases = false,
                                                           const std::vector<ptrdiff_t>& outputPadding = {},
                                                           const std::vector<float>& filterWeights = {},
                                                           const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> makeBinaryConvolution(const ov::Output<Node>& in,
                                                const std::vector<size_t>& filterSize,
                                                const std::vector<size_t>& strides,
                                                const std::vector<ptrdiff_t>& padsBegin,
                                                const std::vector<ptrdiff_t>& padsEnd,
                                                const std::vector<size_t>& dilations,
                                                const op::PadType& autoPad,
                                                size_t numOutChannels,
                                                float padValue,
                                                const std::vector<int8_t>& filterWeihgts = {});

std::shared_ptr<ov::Node> makeSplit(const ov::Output<Node>& in,
                                    const element::Type& type,
                                    size_t numSplits,
                                    int64_t axis);

std::shared_ptr<ov::Node> makeVariadicSplit(const ov::Output<Node>& in,
                                            const std::vector<size_t> numSplits,
                                            int64_t axis);

std::shared_ptr<ov::Node> makeActivation(const ov::Output<Node>& in,
                                         const element::Type& type,
                                         ov::test::utils::ActivationTypes activationType,
                                         std::vector<size_t> inShape = {},
                                         std::vector<float> constantsValue = {});

std::shared_ptr<ov::Node> makeActivation(const ov::ParameterVector& parameters,
                                         const element::Type& type,
                                         ov::test::utils::ActivationTypes activationType);

std::shared_ptr<ov::Node> makeEltwise(const ov::Output<Node>& in0,
                                      const ov::Output<Node>& in1,
                                      ov::test::utils::EltwiseTypes eltwiseType);

std::shared_ptr<ov::Node> makeBatchToSpace(const ov::Output<Node>& in,
                                           const element::Type& type,
                                           const std::vector<int64_t>& blockShape,
                                           const std::vector<int64_t>& cropsBegin,
                                           const std::vector<int64_t>& cropsEnd);

std::shared_ptr<ov::Node> makeSpaceToBatch(const ov::Output<Node>& in,
                                           const element::Type& type,
                                           const std::vector<int64_t>& blockShape,
                                           const std::vector<int64_t>& padsBegin,
                                           const std::vector<int64_t>& padsEnd);

std::shared_ptr<ov::Node> makeStridedSlice(const ov::Output<Node>& in,
                                           const std::vector<int64_t>& begin,
                                           const std::vector<int64_t>& end,
                                           const std::vector<int64_t>& stride,
                                           const element::Type& type,
                                           const std::vector<int64_t>& begin_mask,
                                           const std::vector<int64_t>& end_mask,
                                           const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                                           const std::vector<int64_t>& shrink_mask = std::vector<int64_t>{},
                                           const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

std::shared_ptr<ov::Node> makeStridedSlice(const ov::Output<Node>& in,
                                           const ov::Output<Node>& beginNode,
                                           const ov::Output<Node>& endNode,
                                           const ov::Output<Node>& strideNode,
                                           const element::Type& type,
                                           const std::vector<int64_t>& begin_mask,
                                           const std::vector<int64_t>& end_mask,
                                           const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                                           const std::vector<int64_t>& shrink_mask = std::vector<int64_t>{},
                                           const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

std::shared_ptr<ov::Node> makeSlice(const ov::Output<Node>& in,
                                    const std::vector<int64_t>& begin,
                                    const std::vector<int64_t>& end,
                                    const std::vector<int64_t>& stride,
                                    const std::vector<int64_t>& axes,
                                    const element::Type& type);

std::shared_ptr<ov::Node> makeSlice(const ov::Output<Node>& in,
                                    const ov::Output<Node>& begin,
                                    const ov::Output<Node>& end,
                                    const ov::Output<Node>& stride,
                                    const ov::Output<Node>& axes);

std::shared_ptr<ov::Node> makeSlice(const ov::Output<Node>& in,
                                    const ov::Output<Node>& begin,
                                    const ov::Output<Node>& end,
                                    const ov::Output<Node>& stride);

std::shared_ptr<ov::Node> makeMVN(const ov::Output<Node>& in, bool acrossChannels, bool normalizeVariance, double eps);

std::shared_ptr<ov::Node> makeMVN(const ov::Output<Node>& in,
                                  const ov::AxisSet& axes,
                                  bool normalizeVariance,
                                  double eps);

OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
std::shared_ptr<ov::Node> makeMVN6(const Output<Node>& in,
                                   const Output<Node>& axesNode,
                                   bool normalizeVariance,
                                   float eps,
                                   std::string& epsMode);

OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
std::shared_ptr<ov::Node> makeSqueezeUnsqueeze(const ov::Output<Node>& in,
                                               const element::Type& type,
                                               const std::vector<int>& squeeze_indices,
                                               ov::test::utils::SqueezeOpType opType);

OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
std::shared_ptr<ov::Node> makeMinMax(const ov::Output<Node>& in1,
                                     const ov::Output<Node>& in2,
                                     ov::test::utils::MinMaxOpType opType);

OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
std::shared_ptr<ov::Node> makeProposal(const ov::Output<Node>& class_probs,
                                       const ov::Output<Node>& class_logits,
                                       const std::vector<float>& image_info,
                                       const element::Type& type,
                                       size_t base_size,
                                       size_t pre_nms_topn,
                                       size_t post_nms_topn,
                                       float nms_thresh,
                                       size_t feat_stride,
                                       size_t min_size,
                                       const std::vector<float>& ratio,
                                       const std::vector<float>& scale,
                                       bool clip_before_nms,
                                       bool clip_after_nms,
                                       bool normalize,
                                       float box_size_scale,
                                       float box_coordinate_scale,
                                       std::string framework);

std::shared_ptr<Node> makeFakeQuantize(const ov::Output<Node>& in,
                                       const element::Type& type,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes,
                                       const std::vector<float>& inputLowData,
                                       const std::vector<float>& inputHighData,
                                       const std::vector<float>& outputLowData,
                                       const std::vector<float>& outputHighData);

std::shared_ptr<Node> makeFakeQuantize(const ov::Output<Node>& in,
                                       const element::Type& type,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes,
                                       const int32_t seed = 1);

std::shared_ptr<ov::Node> makeEmbeddingBagOffsetsSum(const element::Type& dataType,
                                                     const element::Type& indicesType,
                                                     const ov::Output<Node>& emb_table_node,
                                                     const std::vector<size_t>& indices,
                                                     const std::vector<size_t>& offsets,
                                                     size_t default_index,
                                                     bool with_weights,
                                                     bool with_default_index);

std::shared_ptr<ov::Node> makeEmbeddingBagPackedSum(const element::Type& dataType,
                                                    const element::Type& indicesType,
                                                    const ov::Output<Node>& emb_table_node,
                                                    const std::vector<std::vector<size_t>>& indices,
                                                    bool with_weights);

std::shared_ptr<ov::Node> makeEmbeddingSegmentsSum(const element::Type& dataType,
                                                   const element::Type& indicesType,
                                                   const ov::Output<Node>& emb_table_node,
                                                   const std::vector<size_t>& indices,
                                                   const std::vector<size_t>& segment_ids,
                                                   size_t num_segments,
                                                   size_t default_index,
                                                   bool with_weights,
                                                   bool with_default_index);

std::shared_ptr<ov::Node> makeReduce(const ov::Output<Node>& data,
                                     const ov::Output<Node>& axes,
                                     bool keepDims,
                                     ov::test::utils::ReductionType reductionType);

OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
std::shared_ptr<Node> makePooling(const ov::Output<Node>& in,
                                  const std::vector<size_t>& strides,
                                  const std::vector<size_t>& padsBegin,
                                  const std::vector<size_t>& padsEnd,
                                  const std::vector<size_t>& kernel,
                                  const op::RoundingType& roundingType,
                                  const op::PadType& padType,
                                  bool excludePad,
                                  const ov::test::utils::PoolingTypes& poolType);

std::shared_ptr<ov::Node> makeScatterNDUpdate(const ov::Output<Node>& in,
                                              const element::Type& indicesType,
                                              const std::vector<size_t>& indicesShape,
                                              const std::vector<size_t>& indices,
                                              const ov::Output<Node>& update);

std::shared_ptr<ov::Node> makeComparison(const ov::Output<Node>& in0,
                                         const ov::Output<Node>& in1,
                                         ov::test::utils::ComparisonTypes comparisonType);

std::shared_ptr<ov::Node> makeConversion(const ov::Output<Node>& in,
                                         const element::Type& type,
                                         const ov::test::utils::ConversionTypes& conversionType);

std::shared_ptr<ov::Node> makeLogical(const ov::Output<Node>& in0,
                                      const ov::Output<Node>& in1,
                                      ov::test::utils::LogicalTypes logicalType);

std::shared_ptr<ov::Node> makeLogical(const ov::ParameterVector& inputs, ov::test::utils::LogicalTypes logicalType);

std::shared_ptr<ov::Node> makeDetectionOutput(const ov::OutputVector& inputs,
                                              const ov::op::v0::DetectionOutput::Attributes& attrs);

std::shared_ptr<ov::Node> makeFullyConnected(const ov::Output<Node>& in,
                                             const element::Type& type,
                                             const size_t outputSize,
                                             bool addBias = true,
                                             const ov::Shape& weightsShape = {},
                                             const std::vector<float>& weights = {},
                                             const std::vector<float>& biasWeights = {});

std::shared_ptr<ov::Node> makeConcat(const std::vector<ov::Output<Node>>& in, const int& axis);

std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& data,
                                  const std::vector<int64_t>& padsBegin,
                                  const std::vector<int64_t>& padsEnd,
                                  float argPadValue,
                                  ov::test::utils::PadMode padMode,
                                  const bool allow_negative_pad = false);

std::shared_ptr<ov::Node> makePad(const ov::Output<Node>& in,
                                  const ov::Output<Node>& beginNode,
                                  const ov::Output<Node>& endNode,
                                  const ov::Output<Node>& valueNode,
                                  ov::test::utils::PadMode padMode,
                                  const bool allow_negative_pad = false);

std::shared_ptr<ov::Node> makeBatchNormInference(const ov::Output<Node>& data, double epsilon);

std::shared_ptr<ov::Node> makeLSTM(
    const OutputVector& in,
    const std::vector<ov::Shape>& constants,
    std::size_t hidden_size,
    const std::vector<std::string>& activations = std::vector<std::string>{"sigmoid", "tanh", "tanh"},
    const std::vector<float>& activations_alpha = {},
    const std::vector<float>& activations_beta = {},
    float clip = 0.f,
    bool make_sequence = false,
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
    ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ,
    float WRB_range = 0.f);

std::shared_ptr<ov::Node> makeGRU(
    const OutputVector& in,
    const std::vector<ov::Shape>& constants,
    std::size_t hidden_size,
    const std::vector<std::string>& activations = std::vector<std::string>{"sigmoid", "tanh"},
    const std::vector<float>& activations_alpha = {},
    const std::vector<float>& activations_beta = {},
    float clip = 0.f,
    bool linear_before_reset = false,
    bool make_sequence = false,
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
    ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ);

std::shared_ptr<ov::Node> makeAUGRU(
    const OutputVector& in,
    const std::vector<ov::Shape>& constants,
    std::size_t hidden_size,
    bool make_sequence = false,
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
    ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ);

std::shared_ptr<ov::Node> makeRNN(
    const OutputVector& in,
    const std::vector<ov::Shape>& constants,
    std::size_t hidden_size,
    const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
    const std::vector<float>& activations_alpha = {},
    const std::vector<float>& activations_beta = {},
    float clip = 0.f,
    bool make_sequence = false,
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
    ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ);

std::shared_ptr<ov::Node> makeGatherND(const ov::Output<Node>& dataNode,
                                       const ov::Shape& indicesShape,
                                       const element::Type& indicesType,
                                       const std::size_t batchDims);

std::shared_ptr<ov::Node> makeGatherND8(const ov::Output<Node>& dataNode,
                                        const ov::Shape& indicesShape,
                                        const element::Type& indicesType,
                                        const std::size_t batchDims);

enum class NmsVersion { NmsVersion5, NmsVersion9 };

OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
std::shared_ptr<ov::Node> makeNms(const ov::Output<Node>& boxes,
                                  const ov::Output<Node>& scores,
                                  const element::Type& maxBoxesPrec,
                                  const element::Type& thrPrec,
                                  const int32_t& maxOutBoxesPerClass,
                                  const float& iouThr,
                                  const float& scoreThr,
                                  const float& softNmsSigma,
                                  const bool isCenter,
                                  const bool& sortResDescend,
                                  const ov::element::Type& outType,
                                  const NmsVersion nmsVersion = NmsVersion::NmsVersion5);

std::shared_ptr<ov::Node> makeDFT(const ov::Output<Node>& dataNode,
                                  const std::vector<int64_t>& axes,
                                  const std::vector<int64_t>& signalSize,
                                  const ov::test::utils::DFTOpType opType);

std::shared_ptr<ov::Node> makeRDFT(const ov::Output<Node>& dataNode,
                                   const std::vector<int64_t>& axes,
                                   const std::vector<int64_t>& signalSize,
                                   const ov::test::utils::DFTOpType opType);
}  // namespace builder
}  // namespace ngraph
