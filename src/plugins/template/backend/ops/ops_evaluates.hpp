// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "evaluate_node.hpp"
#include "openvino/op/rms_norm.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "ov_ops/rms.hpp"

extern template bool evaluate_node<ov::op::v0::Abs>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::BatchNormInference>(std::shared_ptr<ov::Node> node,
                                                                   ov::TensorVector& outputs,
                                                                   const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Ceiling>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Convert>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::CTCGreedyDecoder>(std::shared_ptr<ov::Node> node,
                                                                 ov::TensorVector& outputs,
                                                                 const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::CumSum>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::DetectionOutput>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Elu>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Gelu>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v7::Gelu>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::GRN>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::HardSigmoid>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Interpolate>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v11::Interpolate>(std::shared_ptr<ov::Node> node,
                                                             ov::TensorVector& outputs,
                                                             const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::LRN>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::LSTMCell>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::ReduceMean>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::MVN>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::NormalizeL2>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Proposal>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::PSROIPooling>(std::shared_ptr<ov::Node> node,
                                                             ov::TensorVector& outputs,
                                                             const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::RegionYolo>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Relu>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::ReorgYolo>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::ReverseSequence>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::RNNCell>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Selu>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Sign>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::SquaredDifference>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::TensorIterator>(std::shared_ptr<ov::Node> node,
                                                               ov::TensorVector& outputs,
                                                               const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::ROIPooling>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::AvgPool>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v14::AvgPool>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::BinaryConvolution>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::ConvertLike>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Convolution>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::ConvolutionBackpropData>(std::shared_ptr<ov::Node> node,
                                                                        ov::TensorVector& outputs,
                                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::DeformablePSROIPooling>(std::shared_ptr<ov::Node> node,
                                                                       ov::TensorVector& outputs,
                                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Divide>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Equal>(std::shared_ptr<ov::Node> node,
                                                      ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Greater>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::GroupConvolution>(std::shared_ptr<ov::Node> node,
                                                                 ov::TensorVector& outputs,
                                                                 const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::GroupConvolutionBackpropData>(std::shared_ptr<ov::Node> node,
                                                                             ov::TensorVector& outputs,
                                                                             const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::DeformableConvolution>(std::shared_ptr<ov::Node> node,
                                                                      ov::TensorVector& outputs,
                                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Mod>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Multiply>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::NonMaxSuppression>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::Pad>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v1::GatherTree>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::Assign>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::Bucketize>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::EmbeddingBagOffsetsSum>(std::shared_ptr<ov::Node> node,
                                                                       ov::TensorVector& outputs,
                                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::EmbeddingBagPackedSum>(std::shared_ptr<ov::Node> node,
                                                                      ov::TensorVector& outputs,
                                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::ExtractImagePatches>(std::shared_ptr<ov::Node> node,
                                                                    ov::TensorVector& outputs,
                                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::EmbeddingSegmentsSum>(std::shared_ptr<ov::Node> node,
                                                                     ov::TensorVector& outputs,
                                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::GRUCell>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::NonMaxSuppression>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::ReadValue>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v3::ScatterNDUpdate>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v4::CTCLoss>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v4::LSTMCell>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v4::NonMaxSuppression>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v4::Proposal>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::BatchNormInference>(std::shared_ptr<ov::Node> node,
                                                                   ov::TensorVector& outputs,
                                                                   const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::GatherND>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::GRUSequence>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::LogSoftmax>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::LSTMSequence>(std::shared_ptr<ov::Node> node,
                                                             ov::TensorVector& outputs,
                                                             const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::NonMaxSuppression>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::internal::RMSNorm>(std::shared_ptr<ov::Node> node,
                                                              ov::TensorVector& outputs,
                                                              const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v5::RNNSequence>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::CTCGreedyDecoderSeqLen>(std::shared_ptr<ov::Node> node,
                                                                       ov::TensorVector& outputs,
                                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::ExperimentalDetectronDetectionOutput>(std::shared_ptr<ov::Node> node,
                                                                                     ov::TensorVector& outputs,
                                                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(
    std::shared_ptr<ov::Node> node,
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::GenerateProposals>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(std::shared_ptr<ov::Node> node,
                                                                                        ov::TensorVector& outputs,
                                                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(
    std::shared_ptr<ov::Node> node,
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::ExperimentalDetectronTopKROIs>(std::shared_ptr<ov::Node> node,
                                                                              ov::TensorVector& outputs,
                                                                              const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::GatherElements>(std::shared_ptr<ov::Node> node,
                                                               ov::TensorVector& outputs,
                                                               const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v6::MVN>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v7::DFT>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v7::Einsum>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v7::IDFT>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v7::Roll>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::AdaptiveAvgPool>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::AdaptiveMaxPool>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::Gather>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::MatrixNms>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::MulticlassNms>(std::shared_ptr<ov::Node> node,
                                                              ov::TensorVector& outputs,
                                                              const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::DeformableConvolution>(std::shared_ptr<ov::Node> node,
                                                                      ov::TensorVector& outputs,
                                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::If>(std::shared_ptr<ov::Node> node,
                                                   ov::TensorVector& outputs,
                                                   const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::GatherND>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::DetectionOutput>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::NV12toRGB>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::NV12toBGR>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::I420toRGB>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v8::I420toBGR>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Sigmoid>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Tanh>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Exp>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::Log>(std::shared_ptr<ov::Node> node,
                                                    ov::TensorVector& outputs,
                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v0::PRelu>(std::shared_ptr<ov::Node> node,
                                                      ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::GridSample>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::RDFT>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::NonMaxSuppression>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::IRDFT>(std::shared_ptr<ov::Node> node,
                                                      ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::ROIAlign>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::SoftSign>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v9::MulticlassNms>(std::shared_ptr<ov::Node> node,
                                                              ov::TensorVector& outputs,
                                                              const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v10::IsFinite>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v10::IsInf>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v10::IsNaN>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v10::Unique>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v12::GroupNormalization>(std::shared_ptr<ov::Node> node,
                                                                    ov::TensorVector& outputs,
                                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v13::BitwiseAnd>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v13::BitwiseNot>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v13::BitwiseOr>(std::shared_ptr<ov::Node> node,
                                                           ov::TensorVector& outputs,
                                                           const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v13::BitwiseXor>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::BitwiseLeftShift>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::BitwiseRightShift>(std::shared_ptr<ov::Node> node,
                                                                   ov::TensorVector& outputs,
                                                                   const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v13::NMSRotated>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v13::Multinomial>(std::shared_ptr<ov::Node> node,
                                                             ov::TensorVector& outputs,
                                                             const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v14::Inverse>(std::shared_ptr<ov::Node> node,
                                                         ov::TensorVector& outputs,
                                                         const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::Col2Im>(std::shared_ptr<ov::Node> node,
                                                        ov::TensorVector& outputs,
                                                        const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::ROIAlignRotated>(std::shared_ptr<ov::Node> node,
                                                                 ov::TensorVector& outputs,
                                                                 const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::EmbeddingBagOffsets>(std::shared_ptr<ov::Node> node,
                                                                     ov::TensorVector& outputs,
                                                                     const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::EmbeddingBagPacked>(std::shared_ptr<ov::Node> node,
                                                                    ov::TensorVector& outputs,
                                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::SliceScatter>(std::shared_ptr<ov::Node> node,
                                                              ov::TensorVector& outputs,
                                                              const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::STFT>(std::shared_ptr<ov::Node> node,
                                                      ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v16::ISTFT>(std::shared_ptr<ov::Node> node,
                                                       ov::TensorVector& outputs,
                                                       const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::internal::AUGRUCell>(std::shared_ptr<ov::Node> node,
                                                                ov::TensorVector& outputs,
                                                                const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::internal::AUGRUSequence>(std::shared_ptr<ov::Node> node,
                                                                    ov::TensorVector& outputs,
                                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::internal::RMS>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::StringTensorUnpack>(std::shared_ptr<ov::Node> node,
                                                                    ov::TensorVector& outputs,
                                                                    const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::StringTensorPack>(std::shared_ptr<ov::Node> node,
                                                                  ov::TensorVector& outputs,
                                                                  const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v15::SearchSorted>(std::shared_ptr<ov::Node> node,
                                                              ov::TensorVector& outputs,
                                                              const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v16::Identity>(std::shared_ptr<ov::Node> node,
                                                          ov::TensorVector& outputs,
                                                          const ov::TensorVector& inputs);

extern template bool evaluate_node<ov::op::v16::SegmentMax>(std::shared_ptr<ov::Node> node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs);
