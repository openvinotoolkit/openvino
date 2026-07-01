// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"

const std::vector<std::regex>& disabled_test_patterns() {
    const static std::vector<std::regex> patterns{
        // unsupported metrics
        std::regex(R"(.*smoke_OVGetMetricPropsTest.*OVGetMetricPropsTest.*(RANGE_FOR_STREAMS|MAX_BATCH_SIZE).*)"),

        // CVS-64094
        std::regex(R"(.*ReferenceLogSoftmaxLayerTest.*4.*iType=f16.*axis=.*1.*)"),
        // CVS-64012
        std::regex(R"(.*ReferenceDeformableConvolutionLayerTest.*f16.*real_offset_padding_stride_dialation.*)"),
        std::regex(R"(.*ReferenceDeformableConvolutionLayerTest.*bf16.*)"),
        std::regex(R"(.*ReferenceDeformableConvolutionV8LayerTest.*f16.*real_offset_padding_stride_dialation.*)"),
        std::regex(R"(.*ReferenceDeformableConvolutionV8LayerTest.*bf16.*)"),
        std::regex(R"(.*ReferenceDeformableConvolutionV8LayerTest.*f64.*mask.*)"),
        // CVS-63973
        std::regex(R"(.*ReferencePSROIPoolingLayerTest.*bf16.*)"),
        // CVS-63977
        std::regex(R"(.*ReferenceProposalV1LayerTest.*f16.*)"),
        // CVS-64082
        std::regex(R"(.*ReferenceProposalV4LayerTest.*f16.*)"),
        // CVS-64101
        std::regex(R"(.*ReferenceExperimentalGPLayerTest.*bf16.*)"),
        // CVS-64105
        std::regex(R"(.*ReferenceGatherElementsTestNegative.*)"),
        // CVS-64052
        std::regex(R"(.*ReferenceStridedSliceLayerTest.*strided_slice_stride_optional_dynamic)"),
        // CVS-64017
        std::regex(R"(.*ReferenceGatherTest.*dType=i16.*)"),
        std::regex(R"(.*ReferenceGatherTest.*dType=u16.*)"),
        std::regex(R"(.*ReferenceGatherTest.*dType=bf16.*)"),
        std::regex(R"(.*ReferenceGatherTest.*dType=f64.*)"),
        // CVS-64110
        std::regex(R"(.*ReferenceGatherTestV7.*dType=i16.*)"),
        std::regex(R"(.*ReferenceGatherTestV7.*dType=u16.*)"),
        std::regex(R"(.*ReferenceGatherTestV7.*dType=bf16.*)"),
        std::regex(R"(.*ReferenceGatherTestV7.*dType=f64.*)"),
        // CVS-64037
        std::regex(R"(.*ReferencePadTest.*pad_exterior_2d_0x0)"),
        std::regex(R"(.*ReferencePadTest.*pad_exterior_2d_0x3)"),
        std::regex(R"(.*ReferencePadTest.*pad_exterior_2d_3x0)"),
        // CVS-70975
        std::regex(R"(.*ReferencePadTestParamsTooLarge.*)"),
        // CVS-64113
        std::regex(R"(.*ReferenceRollLayerTest.*dType=i4.*)"),
        std::regex(R"(.*ReferenceRollLayerTest.*dType=u4.*)"),
        // CVS-64066
        std::regex(R"(.*ReferenceGRUCellTestHardsigmoidActivationFunction.*gru_cell_hardsigmoid_activation_function)"),
        // CVS-71381
        std::regex(R"(.*ReferenceExpLayerTest.*u32.*)"),
        std::regex(R"(.*ReferenceExpLayerTest.*u64.*)"),
        // CVS-64054
        std::regex(R"(.*ReferenceTopKTest.*aType=i8.*)"),
        std::regex(R"(.*ReferenceTopKTest.*aType=i16.*)"),
        std::regex(R"(.*ReferenceTopKTest.*aType=u8.*)"),
        std::regex(R"(.*ReferenceTopKTest.*aType=u16.*)"),
        std::regex(R"(.*ReferenceTopKTest.*aType=bf16.*)"),
        std::regex(R"(.*ReferenceTopKTest.*aType=f64.*)"),
        // CVS-63947
        std::regex(R"(.*ReferenceConcatTest.*concat_zero_.*)"),
        // CVS-64102
        std::regex(R"(.*ReferenceExperimentalPGGLayerTest.*iType=bf16.*stride_x=(32|64).*)"),

        // New plugin API doesn't support legacy NV12 I420 preprocessing
        std::regex(R"(.*ConvertNV12WithLegacyTest.*)"),
        std::regex(R"(.*ConvertI420WithLegacyTest.*)"),
        // Plugin version was changed to ov::Version
        std::regex(R"(.*VersionTest.*pluginCurrentVersionIsCorrect.*)"),
        // New plugin API doesn't support changes of pre-processing
        std::regex(R"(.*InferRequestPreprocessTest.*SetPreProcessToInputInfo.*)"),
        std::regex(R"(.*InferRequestPreprocessTest.*SetPreProcessToInferRequest.*)"),
        // New plugin work with tensors, so it means that blob in old API can have different pointers
        std::regex(R"(.*InferRequestIOBBlobTest.*secondCallGetInputDoNotReAllocateData.*)"),
        std::regex(R"(.*InferRequestIOBBlobTest.*secondCallGetOutputDoNotReAllocateData.*)"),
        std::regex(R"(.*InferRequestIOBBlobTest.*secondCallGetInputAfterInferSync.*)"),
        std::regex(R"(.*InferRequestIOBBlobTest.*secondCallGetOutputAfterInferSync.*)"),
        // Old API cannot deallocate tensor
        std::regex(R"(.*InferRequestIOBBlobTest.*canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*)"),
        // Why query state should throw an exception
        std::regex(R"(.*InferRequestQueryStateExceptionTest.*inferreq_smoke_QueryState_ExceptionTest.*)"),
        std::regex(
            R"(.*OVInferRequestCheckTensorPrecision.*get(Input|Output|Inputs|Outputs)From.*FunctionWith(Single|Several).*type=(u4|u1|i4|boolean).*)"),
        std::regex(R"(.*LSTMSequence_With_Hardcoded_Refs.*ReferenceLSTMSequenceTest.*iType=f16.*)"),
        // Interpreter backend doesn't implement evaluate method for OP Multiply (by GroupNormalizationDecomposition)
        std::regex(R"(.*ReferenceGroupNormalization.*_f64*)"),
        // Precision not high enough to get exact result for the complex test cases
        // (both tiny values and very high values necessary)
        std::regex(R"(.*ReferenceInverse.*bf16.*[4,4].*)"),
        // model import is not supported
        std::regex(R"(.*OVCompiledModelBaseTest.import_from_.*)"),
#ifdef _WIN32
        // CVS-63989
        std::regex(R"(.*ReferenceSigmoidLayerTest.*u64.*)"),
        // CVS-120988
        std::regex(R"(.*ReferenceTopKTest.*topk_max_sort_none)"),
        std::regex(R"(.*ReferenceTopKTest.*topk_min_sort_none)"),
#endif

#if defined(__APPLE__) && defined(OPENVINO_ARCH_X86_64)
        // CVS-120988
        std::regex(R"(.*ReferenceTopKTest.*aType=(u32|u64).*topk_(max|min)_sort_none)"),
        std::regex(R"(.*ReferenceTopKTest.*aType=(i32|i64|f16|f32).*topk_min_sort_none)"),
#endif

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
        std::regex(R"(.*smoke_TopK_With_Hardcoded_Refs/ReferenceTopKTestMaxMinSort.CompareWithRefs.*)"),
        std::regex(R"(.*smoke_TopK_With_Hardcoded_Refs/ReferenceTopKTestBackend.CompareWithRefs.*)"),
        std::regex(R"(.*smoke_TopK_With_Hardcoded_Refs/ReferenceTopKTestMaxMinSortV3.CompareWithRefs.*)"),
        std::regex(R"(.*smoke_TopK_With_Hardcoded_Refs/ReferenceTopKTestBackendV3.CompareWithRefs.*)"),
        // fails only on Linux arm64

        std::regex(
            R"(.*ReferenceConversionLayerTest.CompareWithHardcodedRefs/conversionType=(Convert|ConvertLike)_shape=.*_iType=(f16|f32|bf16)_oType=u4.*)"),
#endif
    };

    return patterns;
}
