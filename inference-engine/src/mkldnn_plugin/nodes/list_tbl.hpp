// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MKLDNN_EXTENSION_NODE
# warning "MKLDNN_EXTENSION_NODE is not defined"
# define MKLDNN_EXTENSION_NODE(__prim, __type)
#endif

MKLDNN_EXTENSION_NODE(EmbeddingBagOffsetsSumImpl, EmbeddingBagOffsetsSum);
MKLDNN_EXTENSION_NODE(EmbeddingBagPackedSumImpl, EmbeddingBagPackedSum);
MKLDNN_EXTENSION_NODE(EmbeddingSegmentsSumImpl, EmbeddingSegmentsSum);
MKLDNN_EXTENSION_NODE(CTCLossImpl, CTCLoss);
MKLDNN_EXTENSION_NODE(PriorBoxImpl, PriorBox);
MKLDNN_EXTENSION_NODE(MathImpl, Abs);
MKLDNN_EXTENSION_NODE(MathImpl, Acos);
MKLDNN_EXTENSION_NODE(MathImpl, Acosh);
MKLDNN_EXTENSION_NODE(MathImpl, Asin);
MKLDNN_EXTENSION_NODE(MathImpl, Asinh);
MKLDNN_EXTENSION_NODE(MathImpl, Atan);
MKLDNN_EXTENSION_NODE(MathImpl, Atanh);
MKLDNN_EXTENSION_NODE(MathImpl, Ceil);
MKLDNN_EXTENSION_NODE(MathImpl, Ceiling);
MKLDNN_EXTENSION_NODE(MathImpl, Cos);
MKLDNN_EXTENSION_NODE(MathImpl, Cosh);
MKLDNN_EXTENSION_NODE(MathImpl, Erf);
MKLDNN_EXTENSION_NODE(MathImpl, Floor);
MKLDNN_EXTENSION_NODE(MathImpl, HardSigmoid);
MKLDNN_EXTENSION_NODE(MathImpl, Log);
MKLDNN_EXTENSION_NODE(MathImpl, Neg);
MKLDNN_EXTENSION_NODE(MathImpl, Reciprocal);
MKLDNN_EXTENSION_NODE(MathImpl, Selu);
MKLDNN_EXTENSION_NODE(MathImpl, Sign);
MKLDNN_EXTENSION_NODE(MathImpl, Sin);
MKLDNN_EXTENSION_NODE(MathImpl, Sinh);
MKLDNN_EXTENSION_NODE(MathImpl, SoftPlus);
MKLDNN_EXTENSION_NODE(MathImpl, Softsign);
MKLDNN_EXTENSION_NODE(MathImpl, Tan);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronTopKROIsImpl, ExperimentalDetectronTopKROIs);
MKLDNN_EXTENSION_NODE(ExtractImagePatchesImpl, ExtractImagePatches);
MKLDNN_EXTENSION_NODE(ReverseSequenceImpl, ReverseSequence);
MKLDNN_EXTENSION_NODE(DetectionOutputImpl, DetectionOutput);
MKLDNN_EXTENSION_NODE(ArgMaxImpl, ArgMax);
MKLDNN_EXTENSION_NODE(UnsqueezeImpl, Unsqueeze);
MKLDNN_EXTENSION_NODE(StridedSliceImpl, StridedSlice);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronDetectionOutputImpl, ExperimentalDetectronDetectionOutput);
MKLDNN_EXTENSION_NODE(RegionYoloImpl, RegionYolo);
MKLDNN_EXTENSION_NODE(LogSoftmaxImpl, LogSoftmax);
MKLDNN_EXTENSION_NODE(ReorgYoloImpl, ReorgYolo);
MKLDNN_EXTENSION_NODE(SqueezeImpl, Squeeze);
MKLDNN_EXTENSION_NODE(ConvertImpl, Convert);
MKLDNN_EXTENSION_NODE(FillImpl, Fill);
MKLDNN_EXTENSION_NODE(UniqueImpl, Unique);
MKLDNN_EXTENSION_NODE(PSROIPoolingImpl, PSROIPooling);
MKLDNN_EXTENSION_NODE(DepthToSpaceImpl, DepthToSpace);
MKLDNN_EXTENSION_NODE(OneHotImpl, OneHot);
MKLDNN_EXTENSION_NODE(BroadcastImpl, Broadcast);
MKLDNN_EXTENSION_NODE(ExperimentalSparseWeightedReduceImpl, ExperimentalSparseWeightedSum);
MKLDNN_EXTENSION_NODE(SparseToDenseImpl, SparseToDense);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronROIFeatureExtractorImpl, ExperimentalDetectronROIFeatureExtractor);
MKLDNN_EXTENSION_NODE(ONNXCustomProposalImpl, ExperimentalDetectronGenerateProposalsSingleImage);
MKLDNN_EXTENSION_NODE(NonMaxSuppressionImpl, NonMaxSuppression);
MKLDNN_EXTENSION_NODE(TopKImpl, TopK);
MKLDNN_EXTENSION_NODE(ShuffleChannelsImpl, ShuffleChannels);
MKLDNN_EXTENSION_NODE(SpaceToDepthImpl, SpaceToDepth);
MKLDNN_EXTENSION_NODE(PowerFileImpl, PowerFile);
MKLDNN_EXTENSION_NODE(BatchToSpaceImpl, BatchToSpace);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronPriorGridGeneratorImpl, ExperimentalDetectronPriorGridGenerator);
MKLDNN_EXTENSION_NODE(SimplerNMSImpl, SimplerNMS);
MKLDNN_EXTENSION_NODE(GRNImpl, GRN);
MKLDNN_EXTENSION_NODE(SparseFillEmptyRowsImpl, SparseFillEmptyRows);
MKLDNN_EXTENSION_NODE(BucketizeImpl, Bucketize);
MKLDNN_EXTENSION_NODE(CTCGreedyDecoderImpl, CTCGreedyDecoder);
MKLDNN_EXTENSION_NODE(GatherImpl, Gather);
MKLDNN_EXTENSION_NODE(GatherNDImpl, GatherND);
MKLDNN_EXTENSION_NODE(ProposalImpl, Proposal);
MKLDNN_EXTENSION_NODE(RangeImpl, Range);
MKLDNN_EXTENSION_NODE(SelectImpl, Select);
MKLDNN_EXTENSION_NODE(GatherTreeImpl, GatherTree);
MKLDNN_EXTENSION_NODE(PriorBoxClusteredImpl, PriorBoxClustered);
MKLDNN_EXTENSION_NODE(SpaceToBatchImpl, SpaceToBatch);
MKLDNN_EXTENSION_NODE(SparseSegmentReduceImpl, SparseSegmentMean);
MKLDNN_EXTENSION_NODE(SparseSegmentReduceImpl, SparseSegmentSqrtN);
MKLDNN_EXTENSION_NODE(SparseSegmentReduceImpl, SparseSegmentSum);
MKLDNN_EXTENSION_NODE(CumSumImpl, CumSum);
