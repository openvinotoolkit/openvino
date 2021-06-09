// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MKLDNN_EXTENSION_NODE
# warning "MKLDNN_EXTENSION_NODE is not defined"
# define MKLDNN_EXTENSION_NODE(__prim, __type)
#endif

MKLDNN_EXTENSION_NODE(CTCLossImpl, CTCLoss);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronTopKROIsImpl, ExperimentalDetectronTopKROIs);
MKLDNN_EXTENSION_NODE(ExtractImagePatchesImpl, ExtractImagePatches);
MKLDNN_EXTENSION_NODE(ReverseSequenceImpl, ReverseSequence);
MKLDNN_EXTENSION_NODE(DetectionOutputImpl, DetectionOutput);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronDetectionOutputImpl, ExperimentalDetectronDetectionOutput);
MKLDNN_EXTENSION_NODE(LogSoftmaxImpl, LogSoftmax);
MKLDNN_EXTENSION_NODE(ReorgYoloImpl, ReorgYolo);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronROIFeatureExtractorImpl, ExperimentalDetectronROIFeatureExtractor);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronGenerateProposalsSingleImageImpl, ExperimentalDetectronGenerateProposalsSingleImage);
MKLDNN_EXTENSION_NODE(NonMaxSuppressionImpl, NonMaxSuppressionIEInternal);
MKLDNN_EXTENSION_NODE(TopKImpl, TopK);
MKLDNN_EXTENSION_NODE(ExperimentalDetectronPriorGridGeneratorImpl, ExperimentalDetectronPriorGridGenerator);
MKLDNN_EXTENSION_NODE(GRNImpl, GRN);
MKLDNN_EXTENSION_NODE(BucketizeImpl, Bucketize);
MKLDNN_EXTENSION_NODE(CTCGreedyDecoderImpl, CTCGreedyDecoder);
MKLDNN_EXTENSION_NODE(CTCGreedyDecoderSeqLenImpl, CTCGreedyDecoderSeqLen);
MKLDNN_EXTENSION_NODE(ProposalImpl, Proposal);
MKLDNN_EXTENSION_NODE(RangeImpl, Range);
MKLDNN_EXTENSION_NODE(GatherTreeImpl, GatherTree);
MKLDNN_EXTENSION_NODE(CumSumImpl, CumSum);
