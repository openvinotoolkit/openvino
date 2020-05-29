// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MKLDNN_EXTENSION_NODE
# warning "MKLDNN_EXTENSION_NODE is not defined"
# define MKLDNN_EXTENSION_NODE(__prim, __type)
#endif

#if GraphGen(Gen_Unsqueeze)
MKLDNN_EXTENSION_NODE(PriorBoxImpl, PriorBox);
#endif

#if GraphGen(Gen_Abs)
MKLDNN_EXTENSION_NODE(MathImpl, Abs);
#endif

#if GraphGen(Gen_Acos)
MKLDNN_EXTENSION_NODE(MathImpl, Acos);
#endif

#if GraphGen(Gen_Acosh)
MKLDNN_EXTENSION_NODE(MathImpl, Acosh);
#endif

#if GraphGen(Gen_Asin)
MKLDNN_EXTENSION_NODE(MathImpl, Asin);
#endif

#if GraphGen(Gen_Asinh)
MKLDNN_EXTENSION_NODE(MathImpl, Asinh);
#endif

#if GraphGen(Gen_Atan)
MKLDNN_EXTENSION_NODE(MathImpl, Atan);
#endif

#if GraphGen(Gen_Atanh)
MKLDNN_EXTENSION_NODE(MathImpl, Atanh);
#endif

#if GraphGen(Gen_Ceil)
MKLDNN_EXTENSION_NODE(MathImpl, Ceil);
#endif

#if GraphGen(Gen_Cos)
MKLDNN_EXTENSION_NODE(MathImpl, Cos);
#endif

#if GraphGen(Gen_Cosh)
MKLDNN_EXTENSION_NODE(MathImpl, Cosh);
#endif

#if GraphGen(Gen_Erf)
MKLDNN_EXTENSION_NODE(MathImpl, Erf);
#endif

#if GraphGen(Gen_Floor)
MKLDNN_EXTENSION_NODE(MathImpl, Floor);
#endif

#if GraphGen(Gen_HardSigmoid)
MKLDNN_EXTENSION_NODE(MathImpl, HardSigmoid);
#endif

#if GraphGen(Gen_Log)
MKLDNN_EXTENSION_NODE(MathImpl, Log);
#endif

#if GraphGen(Gen_Neg)
MKLDNN_EXTENSION_NODE(MathImpl, Neg);
#endif

#if GraphGen(Gen_Reciprocal)
MKLDNN_EXTENSION_NODE(MathImpl, Reciprocal);
#endif

#if GraphGen(Gen_Selu)
MKLDNN_EXTENSION_NODE(MathImpl, Selu);
#endif

#if GraphGen(Gen_Sign)
MKLDNN_EXTENSION_NODE(MathImpl, Sign);
#endif

#if GraphGen(Gen_Sin)
MKLDNN_EXTENSION_NODE(MathImpl, Sin);
#endif

#if GraphGen(Gen_Sinh)
MKLDNN_EXTENSION_NODE(MathImpl, Sinh);
#endif

#if GraphGen(Gen_Softplus)
MKLDNN_EXTENSION_NODE(MathImpl, Softplus);
#endif

#if GraphGen(Gen_Softsign)
MKLDNN_EXTENSION_NODE(MathImpl, Softsign);
#endif

#if GraphGen(Gen_Tan)
MKLDNN_EXTENSION_NODE(MathImpl, Tan);
#endif

#if GraphGen(Gen_ExperimentalDetectronTopKROIs)
MKLDNN_EXTENSION_NODE(ExperimentalDetectronTopKROIsImpl, ExperimentalDetectronTopKROIs);
#endif

#if GraphGen(ExtractImagePatches)
MKLDNN_EXTENSION_NODE(ExtractImagePatchesImpl, ExtractImagePatches);
#endif

#if GraphGen(Gen_ReverseSequence)
MKLDNN_EXTENSION_NODE(ReverseSequenceImpl, ReverseSequence);
#endif

#if GraphGen(Gen_DetectionOutput)
MKLDNN_EXTENSION_NODE(DetectionOutputImpl, DetectionOutput);
#endif

#if GraphGen(Gen_ArgMax)
MKLDNN_EXTENSION_NODE(ArgMaxImpl, ArgMax);
#endif

#if GraphGen(Gen_Unsqueeze)
MKLDNN_EXTENSION_NODE(UnsqueezeImpl, Unsqueeze);
#endif

#if GraphGen(Gen_StridedSlice)
MKLDNN_EXTENSION_NODE(StridedSliceImpl, StridedSlice);
#endif

#if GraphGen(Gen_ExperimentalDetectronDetectionOutput)
MKLDNN_EXTENSION_NODE(ExperimentalDetectronDetectionOutputImpl, ExperimentalDetectronDetectionOutput);
#endif

#if GraphGen(Gen_RegionYolo)
MKLDNN_EXTENSION_NODE(RegionYoloImpl, RegionYolo);
#endif

#if GraphGen(Gen_LogSoftmax)
MKLDNN_EXTENSION_NODE(LogSoftmaxImpl, LogSoftmax);
#endif

#if GraphGen(Gen_ReorgYolo)
MKLDNN_EXTENSION_NODE(ReorgYoloImpl, ReorgYolo);
#endif

#if GraphGen(Gen_Squeeze)
MKLDNN_EXTENSION_NODE(SqueezeImpl, Squeeze);
#endif

#if GraphGen(Gen_Convert)
MKLDNN_EXTENSION_NODE(ConvertImpl, Convert);
#endif

#if GraphGen(Gen_Fill)
MKLDNN_EXTENSION_NODE(FillImpl, Fill);
#endif

#if GraphGen(Gen_Unique)
MKLDNN_EXTENSION_NODE(UniqueImpl, Unique);
#endif

#if GraphGen(Gen_PSROIPooling)
MKLDNN_EXTENSION_NODE(PSROIPoolingImpl, PSROIPooling);
#endif

#if GraphGen(Gen_DepthToSpace)
MKLDNN_EXTENSION_NODE(DepthToSpaceImpl, DepthToSpace);
#endif

#if GraphGen(Gen_ScatterUpdate)
MKLDNN_EXTENSION_NODE(ScatterImpl, ScatterUpdate);
#endif

#if GraphGen(Gen_OneHot)
MKLDNN_EXTENSION_NODE(OneHotImpl, OneHot);
#endif

#if GraphGen(Gen_Broadcast)
MKLDNN_EXTENSION_NODE(BroadcastImpl, Broadcast);
#endif

#if GraphGen(Gen_ExperimentalSparseWeightedSum)
MKLDNN_EXTENSION_NODE(ExperimentalSparseWeightedReduceImpl, ExperimentalSparseWeightedSum);
#endif

#if GraphGen(Gen_SparseToDense)
MKLDNN_EXTENSION_NODE(SparseToDenseImpl, SparseToDense);
#endif

#if GraphGen(Gen_ExperimentalDetectronROIFeatureExtractor)
MKLDNN_EXTENSION_NODE(ExperimentalDetectronROIFeatureExtractorImpl, ExperimentalDetectronROIFeatureExtractor);
#endif

#if GraphGen(Gen_ExperimentalDetectronGenerateProposalsSingleImage)
MKLDNN_EXTENSION_NODE(ONNXCustomProposalImpl, ExperimentalDetectronGenerateProposalsSingleImage);
#endif

#if GraphGen(Gen_NonMaxSuppression)
MKLDNN_EXTENSION_NODE(NonMaxSuppressionImpl, NonMaxSuppression);
#endif

#if GraphGen(Gen_TopK)
MKLDNN_EXTENSION_NODE(TopKImpl, TopK);
#endif

#if GraphGen(Gen_ShuffleChannels)
MKLDNN_EXTENSION_NODE(ShuffleChannelsImpl, ShuffleChannels);
#endif

#if GraphGen(Gen_SpaceToDepth)
MKLDNN_EXTENSION_NODE(SpaceToDepthImpl, SpaceToDepth);
#endif

#if GraphGen(Gen_PowerFile)
MKLDNN_EXTENSION_NODE(PowerFileImpl, PowerFile);
#endif

#if GraphGen(Gen_Interp)
MKLDNN_EXTENSION_NODE(InterpImpl, Interp);
#endif

#if GraphGen(Gen_BatchToSpace)
MKLDNN_EXTENSION_NODE(BatchToSpaceImpl, BatchToSpace);
#endif

#if GraphGen(Gen_ExperimentalDetectronPriorGridGenerator)
MKLDNN_EXTENSION_NODE(ExperimentalDetectronPriorGridGeneratorImpl, ExperimentalDetectronPriorGridGenerator);
#endif

#if GraphGen(Gen_SimplerNMS)
MKLDNN_EXTENSION_NODE(SimplerNMSImpl, SimplerNMS);
#endif

#if GraphGen(Gen_Pad)
MKLDNN_EXTENSION_NODE(PadImpl, Pad);
#endif

#if GraphGen(Gen_GRN)
MKLDNN_EXTENSION_NODE(GRNImpl, GRN);
#endif

#if GraphGen(Gen_SparseFillEmptyRows)
MKLDNN_EXTENSION_NODE(SparseFillEmptyRowsImpl, SparseFillEmptyRows);
#endif

#if GraphGen(Gen_Bucketize)
MKLDNN_EXTENSION_NODE(BucketizeImpl, Bucketize);
#endif

#if GraphGen(Gen_CTCGreedyDecoder)
MKLDNN_EXTENSION_NODE(CTCGreedyDecoderImpl, CTCGreedyDecoder);
#endif

#if GraphGen(Gen_Gather)
MKLDNN_EXTENSION_NODE(GatherImpl, Gather);
#endif

#if GraphGen(Gen_Proposal)
MKLDNN_EXTENSION_NODE(ProposalImpl, Proposal);
#endif

#if GraphGen(Gen_Range)
MKLDNN_EXTENSION_NODE(RangeImpl, Range);
#endif

#if GraphGen(Gen_Select)
MKLDNN_EXTENSION_NODE(SelectImpl, Select);
#endif

#if GraphGen(Gen_ReduceAnd)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceAnd);
#endif

#if GraphGen(Gen_ReduceL1)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceL1);
#endif

#if GraphGen(Gen_ReduceL2)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceL2);
#endif

#if GraphGen(Gen_ReduceLogSum)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceLogSum);
#endif

#if GraphGen(Gen_ReduceLogSumExp)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceLogSumExp);
#endif

#if GraphGen(Gen_ReduceMax)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceMax);
#endif

#if GraphGen(Gen_ReduceMean)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceMean);
#endif

#if GraphGen(Gen_ReduceMin)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceMin);
#endif

#if GraphGen(Gen_ReduceOr)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceOr);
#endif

#if GraphGen(Gen_ReduceProd)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceProd);
#endif

#if GraphGen(Gen_ReduceSum)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceSum);
#endif

#if GraphGen(Gen_ReduceSumSquare)
MKLDNN_EXTENSION_NODE(ReduceImpl, ReduceSumSquare);
#endif

#if GraphGen(Gen_GatherTree)
MKLDNN_EXTENSION_NODE(GatherTreeImpl, GatherTree);
#endif

#if GraphGen(Gen_PriorBoxClustered)
MKLDNN_EXTENSION_NODE(PriorBoxClusteredImpl, PriorBoxClustered);
#endif

#if GraphGen(Gen_SpaceToBatch)
MKLDNN_EXTENSION_NODE(SpaceToBatchImpl, SpaceToBatch);
#endif

#if GraphGen(Gen_SparseSegmentMean)
MKLDNN_EXTENSION_NODE(SparseSegmentReduceImpl, SparseSegmentMean);
#endif

#if GraphGen(Gen_SparseSegmentSqrtN)
MKLDNN_EXTENSION_NODE(SparseSegmentReduceImpl, SparseSegmentSqrtN);
#endif

#if GraphGen(Gen_SparseSegmentSum)
MKLDNN_EXTENSION_NODE(SparseSegmentReduceImpl, SparseSegmentSum);
#endif

#if GraphGen(Gen_CumSum)
MKLDNN_EXTENSION_NODE(CumSumImpl, CumSum);
#endif
