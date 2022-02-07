// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // CVS-66280
        R"(.*canLoadCorrectNetworkAndCheckConfig.*)",
        R"(.*canSetCorrectConfigLoadNetworkAndCheckConfig.*)",
        //
        R"(.*ExclusiveAsyncRequests.*)",
        R"(.*ReusableCPUStreamsExecutor.*)",
        R"(.*SplitLayerTest.*numSplits=30.*)",
        // CVS-51758
        R"(.*InferRequestPreprocessConversionTest.*oLT=(NHWC|NCHW).*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*oLT=1.*)",
        //Not Implemented
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(canSetConfigToExecNet|canSetConfigToExecNetAndCheckConfigAndCheck).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution|CheckExecGraphInfoSerialization).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CanCreateTwoExeNetworksAndCheckFunction).*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(checkGetExecGraphInfoIsNotNullptr).*)",
        R"(.*LoadNetworkCreateDefaultExecGraphResult.*)",

        // TODO: Round with f16 is not supported
        R"(.*smoke_Hetero_BehaviorTests.*OVExecGraphImportExportTest.*readFromV10IR.*)",
        // TODO: support import / export of precisions in template plugin
        R"(.*smoke_Hetero_BehaviorTests.*OVExecGraphImportExportTest.ieImportExportedFunction.*)",
        R"(.*smoke_BehaviorTests.*OVExecGraphImportExportTest.ieImportExportedFunction.*)",
        // TODO: Round with f16 is not supported
        R"(.*smoke_Hetero_BehaviorTests.*OVExecGraphImportExportTest.*readFromV10IR.*)",

        R"(.*importExportedIENetworkParameterResultOnly.*elementType=(i8|u8).*)",
        R"(.*importExportedIENetworkParameterResultOnly.*elementType=(i16|u16).*)",
        R"(.*importExportedIENetworkParameterResultOnly.*elementType=(i64|u64).*)",
        R"(.*importExportedIENetworkParameterResultOnly.*elementType=u32.*)",
        R"(.*importExportedIENetworkConstantResultOnly.*elementType=(u32|u64).*)",

        // CVS-64094
        R"(.*ReferenceLogSoftmaxLayerTest.*4.*iType=f16.*axis=.*1.*)",
        //CVS-64012
        R"(.*ReferenceDeformableConvolutionLayerTest.*f16.*real_offset_padding_stride_dialation.*)",
        R"(.*ReferenceDeformableConvolutionLayerTest.*bf16.*)",
        R"(.*ReferenceDeformableConvolutionV8LayerTest.*f16.*real_offset_padding_stride_dialation.*)",
        R"(.*ReferenceDeformableConvolutionV8LayerTest.*bf16.*)",
        R"(.*ReferenceDeformableConvolutionV8LayerTest.*f64.*mask.*)",
        //CVS-63973
        R"(.*ReferencePSROIPoolingLayerTest.*bf16.*)",
        //CVS-63977
        R"(.*ReferenceProposalV1LayerTest.*f16.*)",
        //CVS-64082
        R"(.*ReferenceProposalV4LayerTest.*f16.*)",
        // CVS-64101
        R"(.*ReferenceExperimentalGPLayerTest.*bf16.*)",
        // CVS-64105
        R"(.*ReferenceGatherElementsTestNegative.*)",
        // CVS-64052
        R"(.*ReferenceStridedSliceLayerTest.*strided_slice_stride_optional_dynamic)",
        // CVS-64017
        R"(.*ReferenceGatherTest.*dType=i16.*)",
        R"(.*ReferenceGatherTest.*dType=u16.*)",
        R"(.*ReferenceGatherTest.*dType=bf16.*)",
        R"(.*ReferenceGatherTest.*dType=f64.*)",
        // CVS-64110
        R"(.*ReferenceGatherTestV7.*dType=i16.*)",
        R"(.*ReferenceGatherTestV7.*dType=u16.*)",
        R"(.*ReferenceGatherTestV7.*dType=bf16.*)",
        R"(.*ReferenceGatherTestV7.*dType=f64.*)",
        // CVS-64037
        R"(.*ReferencePadTest.*pad_exterior_2d_0x0)",
        R"(.*ReferencePadTest.*pad_exterior_2d_0x3)",
        R"(.*ReferencePadTest.*pad_exterior_2d_3x0)",
        // CVS-70975
        R"(.*ReferencePadTestParamsTooLarge.*)",
        // CVS-64006
        R"(.*ReferenceBatchToSpaceLayerTest.*dType=i4.*)",
        R"(.*ReferenceBatchToSpaceLayerTest.*dType=u4.*)",
        // CVS-64113
        R"(.*ReferenceRollLayerTest.*dType=i4.*)",
        R"(.*ReferenceRollLayerTest.*dType=u4.*)",
        // CVS-64050
        R"(.*ReferenceSpaceToBatchLayerTest.*dType=i4.*)",
        R"(.*ReferenceSpaceToBatchLayerTest.*dType=u4.*)",
        // CVS-64066
        R"(.*ReferenceGRUCellTestHardsigmoidActivationFunction.*gru_cell_hardsigmoid_activation_function)",
        // CVS-71381
        R"(.*ReferenceExpLayerTest.*u32.*)",
        R"(.*ReferenceExpLayerTest.*u64.*)",
        // CVS-64054
        R"(.*ReferenceTopKTest.*aType=i8.*)",
        R"(.*ReferenceTopKTest.*aType=i16.*)",
        R"(.*ReferenceTopKTest.*aType=u8.*)",
        R"(.*ReferenceTopKTest.*aType=u16.*)",
        R"(.*ReferenceTopKTest.*aType=bf16.*)",
        R"(.*ReferenceTopKTest.*aType=f64.*)",
        // CVS-63947
        R"(.*ReferenceConcatTest.*concat_zero_.*)",
        // CVS-64102
        R"(.*ReferenceExperimentalPGGLayerTest.*iType=bf16.*stride_x=(32|64).*)",
        // CVS-72215
        R"(.*ReferenceTileTest.*aType=i4.*)",
        R"(.*ReferenceTileTest.*aType=u4.*)",
        // CVS-71891
        R"(.*ReferenceTileTest.*rType=i4.*)",
        R"(.*ReferenceTileTest.*rType=u4.*)",
        R"(.*DeviceID.*)",
    };

#ifdef _WIN32
    // CVS-63989
     retVector.emplace_back(R"(.*ReferenceSigmoidLayerTest.*u64.*)");
    // CVS-64054
    retVector.emplace_back(R"(.*ReferenceTopKTest.*topk_max_sort_none)");
    retVector.emplace_back(R"(.*ReferenceTopKTest.*topk_min_sort_none)");
#endif
    return retVector;
}
