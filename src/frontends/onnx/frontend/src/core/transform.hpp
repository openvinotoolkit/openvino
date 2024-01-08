// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <onnx/onnx_pb.h>

namespace ngraph {
namespace onnx_import {
namespace transform {

static const std::vector<std::string> onnx_functions_to_expand = {"Bernoulli",
                                                                  "Celu",
                                                                  "NegativeLogLikelihoodLoss",
                                                                  "SoftmaxCrossEntropyLoss",
                                                                  "LayerNormalization"};

/// \brief Replace nodes with expanded body of ONNX functions
///
/// Some ONNX operators are specified as functions, which can be expanded to
/// a subgraph or more primitive operations. This functions modifies the ONNX
/// model by replacing operations of types listed in onnx_functions_to_expand
/// with their expanded subgraphs.
///
/// \param model_proto Protobuf message with ONNX model to transform.
void expand_onnx_functions(ONNX_NAMESPACE::ModelProto& model_proto);

static const std::vector<std::string> legacy_ops_to_fixup = {"DeformableConv2D",
                                                             "DetectionOutput",
                                                             "ExperimentalDetectronDetectionOutput",
                                                             "ExperimentalDetectronGenerateProposalsSingleImage",
                                                             "ExperimentalDetectronGroupNorm",
                                                             "ExperimentalDetectronPriorGridGenerator",
                                                             "ExperimentalDetectronROIFeatureExtractor",
                                                             "ExperimentalDetectronTopKROIs",
                                                             "FakeQuantize",
                                                             "GenerateProposals",
                                                             "GroupNorm",
                                                             "Normalize",
                                                             "PriorBox",
                                                             "PriorBoxClustered",
                                                             "Swish"};

/// \brief Add support for models with custom operators mistakenly registered in
///        "ai.onnx" domain.
///
/// Some legacy models use custom operators (listed in legacy_ops_to_fixup vector) which
/// were registered in the default ONNX domain. This function updates nodes with these
/// operations to use OPENVINO_ONNX_DOMAIN in order to process them correctly
/// in the nGraph ONNX Importer.
///
/// \param model_proto Protobuf message with ONNX model to transform.
void fixup_legacy_operators(ONNX_NAMESPACE::ModelProto& model_proto);

}  // namespace transform
}  // namespace onnx_import
}  // namespace ngraph
