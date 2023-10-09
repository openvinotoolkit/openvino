// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <ie_system_conf.h>

#include <string>
#include <vector>

#include "ie_parallel.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // TODO: Issue 31841
        R"(.*(QuantGroupConvBackpropData3D).*)",
        // TODO: Issue 31843
        R"(.*(QuantConvBackpropData3D).*)",
        R"(.*(QuantConvBackpropData2D).*(QG=Perchannel).*)",
        R"(.*(QuantGroupConvBackpropData2D).*(QG=Perchannel).*)",
        // TODO: Issue 33886
        R"(.*(QuantGroupConv2D).*)",
        R"(.*(QuantGroupConv3D).*)",
        // TODO: Issue: 34518
        R"(.*RangeLayerTest.*)",
        R"(.*(RangeAddSubgraphTest).*Start=1.2.*Stop=(5.2|-5.2).*Step=(0.1|-0.1).*netPRC=FP16.*)",
        R"(.*(RangeNumpyAddSubgraphTest).*netPRC=FP16.*)",
        // TODO: Issue: 43793
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*iPRC=0.*_iLT=1.*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*_oLT=1.*)",
        // TODO: Issue: 63469
        R"(.*ConversionLayerTest.*ConvertLike.*)",
        // TODO: Issue: 34055
        R"(.*ReluShapeOfSubgraphTest.*)",
        // TODO: Issue: 43314
        R"(.*Broadcast.*mode=BIDIRECTIONAL.*inNPrec=BOOL.*)",
        // TODO: Issue 43417 sporadic issue, looks like an issue in test, reproducible only on Windows platform
        R"(.*decomposition1_batch=5_hidden_size=10_input_size=30_.*tanh.relu.*_clip=0_linear_before_reset=1.*_targetDevice=CPU_.*)",
        // Skip platforms that do not support BF16 (i.e. sse, avx, avx2)
        R"(.*(BF|bf)16.*(jit_avx(?!5)|jit_sse).*)",
        // TODO: Incorrect blob sizes for node BinaryConvolution_X
        R"(.*BinaryConvolutionLayerTest.*)",
        // TODO: 53618. BF16 gemm ncsp convolution crash
        R"(.*_GroupConv.*_inFmts=nc.*_primitive=jit_gemm.*ENFORCE_BF16=YES.*)",
        // TODO: 53578. fork DW bf16 convolution does not support 3d cases yet
        R"(.*_DW_GroupConv.*_inFmts=(ndhwc|nCdhw16c).*ENFORCE_BF16=YES.*)",
        // TODO: 56143. Enable nspc convolutions for bf16 precision
        R"(.*ConvolutionLayerCPUTest.*_inFmts=(ndhwc|nhwc).*ENFORCE_BF16=YES.*)",
        // TODO: 56827. Sporadic test failures
        R"(.*smoke_Conv.+_FP32.ConvolutionLayerCPUTest\.CompareWithRefs.*TS=\(\(.\.67.+\).*inFmts=n.+c.*_primitive=jit_avx2.*)",
        // incorrect jit_uni_planar_convolution with dilation = {1, 2, 1} and output channel 1
        R"(.*smoke_Convolution3D.*D=\(1.2.1\)_O=1.*)",

        // TODO: Issue: 35627. CPU Normalize supports from 2D to 4D blobs
        R"(.*NormalizeL2_1D.*)",
        R"(.*NormalizeL2_5D.*)",
        // Issue: 59788. dnnl_normalize_nchw applies eps after sqrt for across_spatial
        R"(.*NormalizeL2_.*axes=\(1.2.*_eps=100.*)",
        R"(.*NormalizeL2_.*axes=\(2.1.*_eps=100.*)",
        R"(.*NormalizeL2_.*axes=\(3.1.2.*_eps=100.*)",

        // Unsupported operation of type: NormalizeL2 name : Doesn't support reduction axes: (2.2)
        R"(.*BF16NetworkRestore1.*)",
        R"(.*MobileNet_ssd_with_branching.*)",

        // TODO: 57562 No dynamic output shape support
        R"(.*NonZeroLayerTest.*)",
        // Not expected behavior
        R"(.*Behavior.*InferRequestSetBlobByType.*Batched.*)",
        R"(.*OVCompiledModelBaseTest.*(CanGetInputsInfoAndCheck|canSetConfigToCompiledModel).*)",
        R"(.*Behavior.*CorrectConfigCheck.*(canSetConfigAndCheckGetConfig|canSetConfigTwiceAndCheckGetConfig).*CPU_BIND_THREAD=YES.*)",
        // Issue: 72021 Unreasonable abs_threshold for comparing bf16 results
        R"(.*smoke_Reduce.*type=(Prod|Min).*INFERENCE_PRECISION_HINT=(BF|bf)16.*)",
        // TODO: 56520 Accuracy mismatch
        R"(.*ReduceOpsLayerTest.*type=Mean_.*netPRC=(I64|I32).*)",
        R"(.*ReduceOpsLayerTest.*type=Mean_.*netPRC=U64.*)",
        // Not implemented yet:
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModel.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canExportModel.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNetWithIncorrectConfig.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModelWithIncorrectConfig.*)",
        R"(.*Hetero.*Behavior.*ExecutableNetworkBaseTest.*ExecGraphInfo.*)",
        R"(.*Hetero.*Behavior.*OVCompiledModelBaseTest.*ExecGraphInfo.*)",
        R"(.*Hetero.*Behavior.*ExecutableNetworkBaseTest.*CanCreateTwoExeNetworksAndCheckFunction.*)",
        R"(.*Hetero.*Behavior.*OVCompiledModelBaseTest.*canCreateTwoCompiledModelAndCheckTheir.*)",
        // CPU does not support dynamic rank
        // Issue: 66778
        R"(.*smoke_BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
        R"(.*smoke_Hetero_BehaviorTests.*InferFullyDynamicNetworkWith(S|G)etTensor.*)",
        R"(.*smoke_BehaviorTests.*DynamicOutputToDynamicInput.*)",
        R"(.*smoke_BehaviorTests.*DynamicInputToDynamicOutput.*)",
        R"(.*smoke_Hetero_BehaviorTests.*DynamicOutputToDynamicInput.*)",
        R"(.*smoke_Hetero_BehaviorTests.*DynamicInputToDynamicOutput.*)",
        // unsupported metrics
        R"(.*OVGetMetricPropsTest.*OVGetMetricPropsTest.*(MAX_BATCH_SIZE).*)",
        R"(.*smoke_HeteroOVGetMetricPropsTest.*OVGetMetricPropsTest.*(AVAILABLE_DEVICES|OPTIMIZATION_CAPABILITIES|RANGE_FOR_ASYNC_INFER_REQUESTS|RANGE_FOR_STREAMS).*)",
        // supports only '' as device id
        R"(.*OVClassQueryModelTest.*QueryModelWithDeviceID.*)",

        // Issue 67214
        R"(smoke_PrePostProcess.*resize_and_convert_layout_i8.*)",
        // TODO: 67255
        R"(smoke_If.*SimpleIf2OutTest.*)",

        // Issue: 69086
        // need to add support convert BIN -> FP32
        // if we set output precision as BIN, when we create output blob precision looks like UNSPECIFIED
        R"(.*smoke_FakeQuantizeLayerCPUTest.*bin.*)",
        // Issue: 69222
        R"(.*smoke_PriorBoxClustered.*PriorBoxClusteredLayerCPUTest.*_netPRC=f16_.*)",
        // Issue: 72005
        // there are some inconsistency between cpu plugin and ng ref
        // for ctcMergeRepeated is true when legal randomized inputs value.
        // Failure happened on win and macos for current seeds.
        R"(.*CTCLossLayerTest.*CMR=1.*)",
        R"(.*CTCLossLayerCPUTest.*ctcMergeRepeated=1.*)",
        // Issue: 71756
        R"(.*GroupDeconv_2D_DW_BF16/GroupDeconvolutionLayerCPUTest.CompareWithRefs.*PRC=f32.*inFmts=nChw16c_outFmts=nChw16c_primitive=jit_avx512_dw_Fused=Multiply\(PerChannel\).Add\(PerChannel\)_PluginConf_ENFORCE_BF16=YES.*)",
        R"(.*smoke_GroupDeconv_(2|3)D_Blocked_BF16.*S=(\(2\.2\)|\(2\.2\.2\))_PB=(\(0\.0\)|\(0\.0\.0\))_PE=(\(0\.0\)|\(0\.0\.0\))_D=(\(1\.1\)|\(1\.1\.1\))_.*_O=64_G=4.*)",
        // Issue: 59594
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*BOOL.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*MIXED.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*Q78.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*U4.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*I4.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*BIN.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*CUSTOM.*)",
        R"(smoke_ConversionLayerTest/ConversionLayerTest.CompareWithRefs.*UNSPECIFIED.*)",
        // Issue:
        // New API tensor tests
        R"(.*OVInferRequestCheckTensorPrecision.*type=i4.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u1.*)",
        R"(.*OVInferRequestCheckTensorPrecision.*type=u4.*)",
        // Issue: 77390
        R"(.*LoopLayerCPUTest.*exec_cond=0.*)",
        R"(.*LoopLayerCPUTest.*trip_count=0.*)",
        R"(.*LoopForDiffShapesLayerCPUTest.*exec_cond=0.*)",
        R"(.*LoopForDiffShapesLayerCPUTest.*trip_count=0.*)",
        R"(.*LoopForConcatLayerCPUTest.*exec_cond=0.*)",
        R"(.*LoopForConcatLayerCPUTest.*trip_count=0.*)",
        // [ INFO ] Can't compile network without cache for ..  with precision ..
        R"(.*CompileModelCacheTestBase.*CompareWithRefImpl.*KSOFunction.*)",
        R"(.*CompileModelCacheTestBase.*CompareWithRefImpl.*NonMaxSuppression.*)",
        R"(.*CompileModelCacheTestBase.*CompareWithRefImpl.*Nms.*)",
        // Issue: 105838
        R"(smoke_NmsLayerTest.*)",
        // Issue: 95590
        R"(.*CachingSupportCase.*CompileModelCacheTestBase.*(TIwithLSTMcell1|MatMulBias|2InputSubtract)_(u|i).*)",
        // Issue: 95607
        R"(.*CachingSupportCase.*LoadNetworkCacheTestBase.*(TIwithLSTMcell1|MatMulBias|2InputSubtract)_(i|u).*)",
        R"(.*CachingSupportCase.*ReadConcatSplitAssign.*)",
        // 94982. FP32->I32 conversion issue in the reference implementation. There can be some garbage in the rest of float values like 0.333333745.
        // The kernel does not have such garbage. The diff 0.000000745 is taken into account in calculations and affects further type conversion.
        // Reorder->GridSample->Reorder also does not work here. Potential fix is to use nearest conversion instead of truncation.
        R"(.*GridSampleLayerTestCPU.*(BILINEAR|BICUBIC).*(i32|i8).*)",
        // AdaptiveAvgPool is converted into Reduce op for suitable parameters. CPU Reduce impl doesn't support non planar layout for 3D case
        R"(.*StaticAdaPoolAvg3DLayoutTest.*OS=\(1\).*_inFmts=(nwc|nCw16c|nCw8c).*)",
        // Issue: 111404
        R"(.*smoke_set1/GatherElementsCPUTest.*)",
        // Issue: 111406
        R"(.*smoke_InterpolateLinearOnnx_Layout_Test/InterpolateLayerCPUTest.*)",
        R"(.*smoke_InterpolateLinear_Layout_Test/InterpolateLayerCPUTest.*)",
        R"(.*smoke_InterpolateCubic_Layout_Test/InterpolateLayerCPUTest.*)",
        // Issue: 111412
        R"(.*smoke_Proposal_(Static|Dynamic)_Test_Case1/ProposalLayerCPUTest.*)",
        // Issue: 111418
        R"(.*smoke_Snippets_ConvertStub/ConvertStub\.CompareWithRefImpl/IS.*_OT=\(bf16\)_#N=2_#S=2_targetDevice=CPU.*)",
        R"(.*smoke_Snippets_Convert/Convert\.CompareWithRefImpl/IS.*_IT=\(f32\)_OT=\(u8\)_#N=1_#S=1_targetDevice=CPU.*)",
        R"(.*smoke_Snippets_ConvertManyOnInputs/ConvertManyOnInputs\.CompareWithRefImpl/IS.*_IT=\(f32\.u8\)_OT=\(\)_#N=1_#S=1_targetDevice=CPU.*)",
        // Issue: 106939
        R"(.*ScatterNDUpdateLayerCPUTest.*-1.-1.-1.-2.-2.-2.*)",
        // New plugin API doesn't support changes of pre-processing
        R"(.*(Hetero).*InferRequestPreprocessTest.*SetPreProcessToInputInfo.*)",
        R"(.*(Hetero).*InferRequestPreprocessTest.*SetPreProcessToInferRequest.*)",
        // TODO: for 22.2 (Issue 68949)
        R"(.*smoke_AutoBatching_CPU/AutoBatching_Test_DetectionOutput.*)",
        // Issue: 117837
        R"(.*smoke_4D_out_of_range/GatherInPlaceLayerTestCPU.*_indices=\(\-15\).*)",
        // Issue: 120222
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=1_axis=3_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=7_axis=3_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=18_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=21_.*_sort=value_modelType=f16_trgDev=CPU.*)",
        // Issue: 121228
        R"(smoke_TestsDFT_(1|2|3|4)d/DFTLayerTest.Inference.*bf16.*)",
        // Issue: 121313
        R"(smoke_GroupConvBackpropData.*paddingDefined/GroupConvBackpropLayerTest.Inference.*f16.*)",
        R"(smoke_GroupConvBackpropData.*paddingDefined/GroupConvBackpropLayerTest.Inference.*f32.*)",
        // Issue: 122094
        R"(smoke_Interpolate_Basic_Down_Sample_Tail/InterpolateLayerTest.Inference.*(asymmetric|align_corners).*f16.*)",
    };
#if defined(__APPLE__) && defined(OPENVINO_ARCH_ARM64)
    // Issue: 120950
    retVector.emplace_back(R"(.*smoke_TensorIteratorCommon/TensorIteratorTest.Inference.*_modelType=f16_targetDevice=CPU.*)");
    retVector.emplace_back(R"(.*smoke_CtcGreedyDecoderBasic/CTCGreedyDecoderLayerTest.Inference.*netPRC=f16.*trgDev=CPU.*)");
    retVector.emplace_back(R"(.*CTCGreedyDecoderSeqLenLayerTest.Inference.*dataPRC=f16.*trgDev=CPU.*)");
#endif

#if defined(OPENVINO_ARCH_X86)
    retVector.emplace_back(R"(.*DetectionOutputLayerTest.*)");
    // WIP: plugin cannot be loaded for some reason
    retVector.emplace_back(R"(.*HeteroSyntheticTest.*)");
    retVector.emplace_back(R"(.*IEClassBasicTestP.*)");
#elif defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
    {
        // Issue: 121709
        retVector.emplace_back(R"(smoke_ConversionLayerTest/ConversionLayerTest.Inference/conversionOpType=Convert_IS.*_inputPRC=f16_targetPRC=(u|i)8_trgDev=CPU.*)");
        // Issue: 121710
        retVector.emplace_back(R"(smoke_GRUCellCommon/GRUCellTest.Inference/decomposition0_batch=5_.*WType=CONSTANT_RType=CONSTANT_BType=CONSTANT_netPRC=f16_targetDevice=CPU_.*)");
        // Issue: 121715
        retVector.emplace_back(R"(smoke_CompareWithRefs_static/EltwiseLayerTest.Inference/IS.*_eltwise_op_type=Div_secondary_input_type=PARAMETER_opType=VECTOR_model_type=i32_InType=undefined_OutType=undefined_trgDev=CPU.*)");
        retVector.emplace_back(R"(smoke_CompareWithRefs_static_check_collapsing/EltwiseLayerTest.Inference/IS.*_eltwise_op_type=Div_secondary_input_type=PARAMETER_opType=VECTOR_model_type=i32_InType=undefined_OutType=undefined_trgDev=CPU.*)");
        // TODO: enable once streams / tput mode is supported
        retVector.emplace_back(R"(OVClassConfigTestCPU.smoke_CpuExecNetworkCheck(Model|Core)StreamsHasHigherPriorityThanLatencyHint.*)");
        retVector.emplace_back(R"(smoke_BehaviorTests/CorrectConfigCheck.canSetConfigAndCheckGetConfig.*CPU_THROUGHPUT_STREAMS=8.*)");
        retVector.emplace_back(R"(smoke_BehaviorTests/CorrectConfigCheck.canSetConfigTwiceAndCheckGetConfig.*CPU_THROUGHPUT_STREAMS=8.*)");
        retVector.emplace_back(R"(smoke_CPU_OVClassLoadNetworkAndCheckWithSecondaryPropertiesTest/OVClassLoadNetworkAndCheckSecondaryPropertiesTest.LoadNetworkAndCheckSecondaryPropertiesTest.*)");
        retVector.emplace_back(R"(smoke_CPU_OVClassLoadNetworkAndCheckWithSecondaryPropertiesDoubleTest/OVClassLoadNetworkAndCheckSecondaryPropertiesTest.LoadNetworkAndCheckSecondaryPropertiesTest.*)");
        retVector.emplace_back(R"(smoke_CPU_OVClassCompileModelAndCheckSecondaryPropertiesTest.*)");
        retVector.emplace_back(R"(smoke_CPU_OVClassCompileModelAndCheckWithSecondaryPropertiesDoubleTest.*)");
    }
    // invalid test: checks u8 precision for runtime graph, while it should be f32
    retVector.emplace_back(R"(smoke_NegativeQuantizedMatMulMultiplyFusion.*)");
    // int8 specific
    retVector.emplace_back(R"(smoke_Quantized.*)");
    // Limitations in ACL Deconvolution layer:
    // https://github.com/ARM-software/ComputeLibrary/blob/874e0c7b3fe93a6764ecb2d8cfad924af19a9d25/src/runtime/NEON/functions/NEDeconvolutionLayer.cpp#L124
    retVector.emplace_back(R"(smoke_Deconv_2D_Planar_FP32.*DeconvolutionLayerCPUTest.CompareWithRefs.*IS.*TS=.*((2.12.5.7)).*)");
    retVector.emplace_back(R"(smoke_Deconv_2D_AutoPadding_FP32.*DeconvolutionLayerCPUTest.CompareWithRefs/IS=.*TS=.*((2.67.5.7)).*)");

#endif

#if defined(OPENVINO_ARCH_ARM)
    // TODO: rounding errors
    retVector.emplace_back(R"(.*iv_secondaryInputType=PARAMETER_opType=VECTOR_NetType=i32.*)");
#endif

#if !defined(OPENVINO_ARCH_X86_64)
    // very time-consuming test
    retVector.emplace_back(R"(.*OVInferConsistencyTest.*)");
    // TODO: generate new 'expected' runtime graph for non-x64 CPU
    retVector.emplace_back(R"(smoke_serialization/ExecGraphSerializationTest.ExecutionGraph.*)");
    retVector.emplace_back(R"(smoke_ExecGraph/ExecGraphRuntimePrecision.CheckRuntimePrecision/Function=(EltwiseWithTwoDynamicInputs|FakeQuantizeRelu).*)");
    // Issue 108803: bug in CPU scalar implementation
    retVector.emplace_back(R"(smoke_TestsDFT_(1|2|3|4)d/DFTLayerTest.CompareWithRefs.*)");
    retVector.emplace_back(R"(smoke_TestsDFT_(1|2|3|4)d/DFTLayerTest.Inference.*)");
    // Issue 88764, 91647, 108802: accuracy issue
    retVector.emplace_back(R"(MultipleLSTMCellTest/MultipleLSTMCellTest.CompareWithRefs.*)");
    // int8 / code-generation specific
    retVector.emplace_back(R"(smoke_LPT.*)");
    retVector.emplace_back(R"(smoke_Snippets.*)");
#endif

#if defined(_WIN32)
    retVector.emplace_back(R"(.*LoadNetworkCompiledKernelsCacheTest.*CanCreateCacheDirAndDumpBinariesUnicodePath.*)");
#endif

    if (!InferenceEngine::with_cpu_x86_avx512_core()) {
        // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
        // tests are useless on such platforms
        retVector.emplace_back(R"(.*(BF|bf)16.*)");
        retVector.emplace_back(R"(.*bfloat16.*)");
        // MatMul in Snippets uses BRGEMM that is supported only on AVX512 platforms
        // Disabled Snippets MHA tests as well because MHA pattern contains MatMul
        retVector.emplace_back(R"(.*Snippets.*MHA.*)");
        retVector.emplace_back(R"(.*Snippets.*(MatMul|Matmul).*)");
    }
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    if (!InferenceEngine::with_cpu_x86_avx512_core_fp16()) {
        // Skip fp16 tests for paltforms that don't support fp16 precision
        retVector.emplace_back(R"(.*INFERENCE_PRECISION_HINT=(F|f)16.*)");
    }
#elif defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
#if !defined(OV_CPU_ARM_ENABLE_FP16)
    // Skip fp16 tests for paltforms that don't support fp16 precision
    retVector.emplace_back(R"(.*INFERENCE_PRECISION_HINT=(F|f)16.*)");
#else
    // Issue 117407
    retVector.emplace_back(R"(.*EltwiseLayerCPUTest.*IS=\(\[1\.\.10\.2\.5\.6\]_\).*eltwiseOpType=SqDiff.*_configItem=INFERENCE_PRECISION_HINT=f16.*)");
#endif // OV_CPU_ARM_ENABLE_FP16
#endif
    if (!InferenceEngine::with_cpu_x86_avx512_core_vnni() && !InferenceEngine::with_cpu_x86_avx512_core_amx_int8()) {
        // MatMul in Snippets uses BRGEMM that supports i8 only on platforms with VNNI or AMX instructions
        retVector.emplace_back(R"(.*Snippets.*MatMulFQ.*)");
        retVector.emplace_back(R"(.*Snippets.*MatMul.*Quantized.*)");
        retVector.emplace_back(R"(.*Snippets.*MHAFQ.*)");
        retVector.emplace_back(R"(.*Snippets.*MHAINT8.*)");
        retVector.emplace_back(R"(.*Snippets.*MHAQuant.*)");
    }
    if (!InferenceEngine::with_cpu_x86_avx512_core_amx_int8())
        //TODO: Issue 92895
        // on platforms which do not support AMX, we are disabling I8 input tests
        retVector.emplace_back(R"(smoke_LPT/FakeQuantizeWithNotOptimalTransformation.CompareWithRefImpl.*CPU.*I8.*)");
    if (!InferenceEngine::with_cpu_x86_avx512_core_amx_bf16() && !InferenceEngine::with_cpu_x86_bfloat16()) {
        // ignored for not supported bf16 platforms
        retVector.emplace_back(R"(.*smoke_Snippets_EnforcePrecision_bf16.*)");
        retVector.emplace_back(R"(.*smoke_Snippets_MHAWOTransposeEnforceBF16.*)");
        retVector.emplace_back(R"(.*smoke_Snippets_MHAEnforceBF16.*)");
    }

    return retVector;
}
