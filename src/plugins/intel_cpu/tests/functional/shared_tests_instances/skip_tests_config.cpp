// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils/precision_support.h"

#include <string>
#include <vector>

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
        R"(.*(RangeAddSubgraphTest).*Start=1.2.*Stop=(5.2|-5.2).*Step=(0.1|-0.1).*ET=f16.*)",
        R"(.*(RangeNumpyAddSubgraphTest).*ET=f16.*)",
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
        R"(.*ConvolutionLayerCPUTest.*_inFmts=(ndhwc|nhwc).*INFERENCE_PRECISION_HINT=bf16.*)",
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

        // Not expected behavior
        R"(.*OVCompiledModelBaseTest.*(CanGetInputsInfoAndCheck|canSetConfigToCompiledModel).*)",
        R"(.*Behavior.*CorrectConfigCheck.*(canSetConfigAndCheckGetConfig|canSetConfigTwiceAndCheckGetConfig).*CPU_BIND_THREAD=YES.*)",
        // Issue: 72021 Unreasonable abs_threshold for comparing bf16 results
        R"(.*smoke_Reduce.*type=(Prod|Min).*INFERENCE_PRECISION_HINT=(BF|bf)16.*)",
        // TODO: 56520 Accuracy mismatch
        R"(.*ReduceOpsLayerTest.*type=Mean_.*netPRC=(I64|I32).*)",
        R"(.*ReduceOpsLayerTest.*type=Mean_.*netPRC=U64.*)",
        // Not implemented yet:
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModel.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canExportModel.*)",
        R"(.*Behavior.*OVCompiledModelBaseTest.*canSetConfigToCompiledModelWithIncorrectConfig.*)",
        R"(.*Hetero.*Behavior.*OVCompiledModelBaseTest.*ExecGraphInfo.*)",
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
        R"(.*GroupDeconv_2D_DW_BF16/GroupDeconvolutionLayerCPUTest.CompareWithRefs.*PRC=f32.*inFmts=nChw16c_outFmts=nChw16c_primitive=jit_avx512_dw_Fused=Multiply\(PerChannel\).Add\(PerChannel\)_PluginConf_INFERENCE_PRECISION_HINT=bf16*)",
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
        // 94982. FP32->I32 conversion issue in the reference implementation. There can be some garbage in the rest of
        // float values like 0.333333745.
        // The kernel does not have such garbage. The diff 0.000000745 is taken into account in calculations and affects
        // further type conversion.
        // Reorder->GridSample->Reorder also does not work here. Potential fix is to use nearest conversion instead of
        // truncation.
        R"(.*GridSampleLayerTestCPU.*(BILINEAR|BICUBIC).*(i32|i8).*)",
        // AdaptiveAvgPool is converted into Reduce op for suitable parameters. CPU Reduce impl doesn't support non
        // planar layout for 3D case
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
        R"(.*InferRequestPreprocessTest.*SetPreProcessToInputInfo.*)",
        R"(.*InferRequestPreprocessTest.*SetPreProcessToInferRequest.*)",
        // Plugin version was changed to ov::Version
        R"(.*VersionTest.pluginCurrentVersionIsCorrect.*)",
        // Issue: 113703, 114763
        R"(.*smoke_If/SimpleIfTest.*Cond=0.*)",
        // Issue: 114765
        R"(.*smoke_PSROIPoolingAverageLayoutTest/PSROIPoolingLayerCPUTest.*bf16.*)",
        R"(.*smoke_PSROIPoolingBilinearLayoutTest/PSROIPoolingLayerCPUTest.*bf16.*)",
        // TODO: for 22.2 (CVS-68949)
        R"(.*smoke_AutoBatching_CPU/AutoBatching_Test_DetectionOutput.*)",
        // Issue: 120222
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=1_axis=3_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=7_axis=3_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=1_axis=1_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=7_axis=1_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=18_.*_modelType=f16_trgDev=CPU.*)",
        R"(.*smoke_TopK/TopKLayerTest.Inference.*_k=21_.*_sort=value_modelType=f16_trgDev=CPU.*)",
        // Issue: 121812
        R"(.*ConvertCPULayerTest.*outFmts=(nhwc|nChw8c|nChw16c).*)",
        // Need to generate sequence exactly in the i64 data type. Enable in scope of i64 enabling.
        R"(.*RandomUniformLayerTestCPU.*OutPrc=i64.*)",
        // Issue: 123815 (Tests are sensintive to available thread count on testing machines)
        R"(.*smoke_Snippets_MHA_.?D_SplitDimensionM.*)",
        // Issue: 122356
        R"(.*NmsRotatedOpTest.*(SortDesc=True|Clockwise=False).*)",
        // Issue: 126095
        R"(^smoke_Multinomial(?:Static|Dynamic)+(?:Log)*.*seed_g=0_seed_o=0.*device=CPU.*)",
        // Issue: 129025
        R"(.*smoke_CpuExecNetworkCheck.*StreamsHasHigherPriorityThanLatencyHint.*)",
        // Issue: 119648
        R"(.*smoke_LPT/InterpolateTransformation.*)",
        // Issue: 129931
        R"(smoke_LPT/ConvolutionTransformation.CompareWithRefImpl/f32_\[.*,3,16,16\]_CPU_f32_rank=4D_fq_on_data=\{level=256_shape=\[1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ .*18.7 \}_output_high\{ 18.8 \}_precision=\}_fq_on_weights=\{_255_\[6,1,1,1\]_\{ .*1.52806e.*39, .*0.2, .*0.3, .*0.3, .*0.2, .*0.1 \}_\{ 1.52806e.*39, 0.2, 0.3, 0.3, 0.2, 0.1 \}\})",
        // Issue: 133173
        R"(.*smoke_ScaledAttn_CPU/ScaledAttnLayerCPUTest.CompareWithRefs/netPRC=bf16.*has_scale=0.*)",
        R"(.*smoke_LPT_4D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/f32_\[1,8,16,16\]_CPU_f32_\[16,16\]_level=256_shape=\[.*\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=__255_\[.*\]_\{ -12.7 \}_\{ 12.7 \}_\{\}.*)",
        R"(.*smoke_LPT_4D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/f32_\[1,8,16,16\]_CPU_f32_\[16,16\]_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ -12.7 \}_output_high\{ 12.8 \}_precision=.*)",
        R"(.*smoke_LPT_3D/ConvolutionBackpropDataTransformation.CompareWithRefImpl/f32_\[1,8,16\]_CPU_f32_\[16\]_.*_input_high=\{ 25.5 \}_.*_precision=__255_\[1,1,1\]_\{ -12.7 \}_\{ 12.7 \}_\{\}.*)",
        R"(.*smoke_LPT/ConvolutionQDqTransformation.CompareWithRefImpl/f32_\[(1,3,4,4|4,3,4,4)\]_CPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ -12.8 \}_input_high=\{ 12.7 \}_output_low=\{ 0 \}_output_high=\{ 255 \}_precision=f32__u8___f32__.*_f32_\[\]_1_1_undefined__\{, 15\}_f32_\[\]__255_\[1,1,1,1\]_\{ -128 \}_\{ 127 \}__i8___f32__\{ -128 \}_.*_1_1_i8_.*)",
        R"(.*smoke_LPT/ConvolutionQDqTransformation.CompareWithRefImpl/f32_\[(1,3,4,4|4,3,4,4)\]_CPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ -12.8 \}_input_high=\{ 12.7 \}_output_low=\{ 0 \}_output_high=\{ 255 \}_precision=f32__u8___f32_\{\}__\{ 0.1 \}_f32_\[\]_1_1_undefined__\{, 15\}_f32_\[\]__255_\[1,1,1,1\]_\{ -128 \}_\{ 127 \}__i8_.*)",
        R"(.*smoke_LPT/MultiplyTransformation.CompareWithRefImpl/f32_\[1,3,16,16\]_CPU_f32_undefined__on_branch1_0_2.55_0_2.55_on_branch2_-1.28_1.27_-1.28_1.27_1.*)",
        R"(.*smoke_LPT/MultiplyTransformation.CompareWithRefImpl/f32_\[1,3,16,16\]_CPU_f32_broadcast1_undefined__on_branch1_-1.28_1.27_-1.28_1.27_on_branch2_0_2.55_0_2.55_0.*)",
        R"(.*smoke_LPT/MultiplyTransformation.CompareWithRefImpl/f32_\[1,3,16,16\]_CPU_f32_broadcast2_undefined__on_branch1_0_2.55_0_2.55_on_branch2_-1.27_1.28_-1.27_1.28_0.*)",
        R"(.*smoke_LPT/ConvolutionTransformation.CompareWithRefImpl/f32_\[(1|4),3,16,16\]_CPU_f32_rank=4D_fq_on_data=\{level=256_shape=\[1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ -18.7 \}_output_high\{ 18.8 \}_precision=\}_fq_on_weights=\{_255_\[1\]_\{ -18.7 \}_\{ 18.7 \}\}.*)",
        R"(.*smoke_LPT/ConvolutionTransformation.CompareWithRefImpl/f32_\[(1|4),3,16,16\]_CPU_f32_rank=4D_fq_on_data=\{level=256_shape=\[1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ -18.7 \}_output_high\{ 18.8 \}_precision=\}_fq_on_weights=\{_255_\[6,1,1,1\].*)",
        R"(.*smoke_LPT/RecurrentCellTransformation.CompareWithRefImpl/f32_\[1,2,16\]_CPU_f32FQ_X_level=256_.*_FQ_W_level=255.*)",
        R"(.*smoke_LPT/SubtractTransformation.CompareWithRefImpl/f16_\[1,3,16,16\]_CPU_f32.*)",
        R"(.*smoke_LPT/FakeQuantizeTransformation.CompareWithRefImpl/f32_\[1,32,72,48\]_CPU_f32_0_level=65536_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 65.535 \}_output_low=\{ 0 \}_output_high=\{ 65.535 \}_precision=.*)",
        R"(.*smoke_LPT/FakeQuantizeTransformation.CompareWithRefImpl/f32_\[1,32,72,48\]_CPU_f32_0_level=65536_shape=\[\]_input_low=\{ -32.768 \}_input_high=\{ 32.767 \}_output_low=\{ -32.768 \}_output_high=\{ 32.767 \}_precision=.*)",
        R"(.*smoke_LPT/MoveFakeQuantizeTransformation.CompareWithRefImpl/f32_\[(1|4),1,16,16\]_CPU_f32SPLIT:0_OP:_FQ:level=256_shape=\[\]_input_low=\{ (0|-1.28) \}_input_high=\{ (2.55|1.27) \}_output_low=\{ (0|-1.28) \}_output_high=\{ (2.55|255|1.27) \}_precision=_DQ:.*)",
        R"(.*smoke_LPT/MoveFakeQuantizeTransformation.CompareWithRefImpl/f32_\[(1|4),1,16,16\]_CPU_f32SPLIT:0_OP:relu_FQ:level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 2.55 \}_output_low=\{ 0 \}_output_high=\{ 255 \}_precision=_DQ:__f32_\{\}__\{ 0.01 \}_undefined_\[\]_0_1_undefined.*)",
        R"(.*smoke_LPT/MoveFakeQuantizeTransformation.CompareWithRefImpl/f32_\[(1|4),1,16,16\]_CPU_f32SPLIT:0_OP:relu_FQ:level=256_shape=\[1,6,1,1\]_input_low=\{ 0, 0, 0, 0, 0, 0 \}_input_high=\{ 2.55, 1.275, 0.85, 0.6375, 0.51, 0.425 \}_output_low=\{ -128, -128, -128, -128, -128, -128 \}_output_high=\{ 127, 127, 127, 127, 127, 127 \}_precision=_DQ:\{\}.*)",
        R"(.*smoke_LPT/MoveFakeQuantizeTransformation.CompareWithRefImpl/f32_\[(1|4),1,16,16\]_CPU_f32SPLIT:(0|1)_OP:_FQ:level=256_shape=\[1,6,1,1\]_input_low=\{ 0, 0, 0, 0, 0, 0 \}_input_high=\{ 2.55, 1.275, 0.85, 0.6375, 0.51, 0.425 \}_output_low=\{ 0, 0, 0, 0, 0, 0 \}_output_high=\{ 255, 127.5, 85, 63.75, 51, 42.5 \}_precision=_DQ:__f32_.*)",
        R"(.*smoke_LPT/EliminateFakeQuantizeTransformation.CompareWithRefImpl/CPU_f32_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ 127.5 \}_output_low=\{ 0 \}_output_high\{ 127.5 \}_precision=f32_level=256_shape=\[\]_input_low=\{ 0 \}_input_high=\{ (127.5|121.429) \}_output_low=\{ 0 \}_output_high\{ (127.5|121.429) \}_precision=f32.*)",
        R"(.*smoke_LPT/MatMulWithOptimizedConstantFq.CompareWithRefImpl/f32_\[1,16\]_\[(10|16),(10|16)\]_CPU_level=256_shape=\[1\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=_level=255_shape=\[1\]_input_low=\{ -12.7 \}_input_high=\{ 12.7 \}_output_low=\{ -12.7 \}_output_high\{ 12.7 \}_precision=.*)",
        R"(.*smoke_LPT/FuseDequantizeToFakeQuantizeTransformation.CompareWithRefImpl/CPU_f32_0_undefined_\[\]_f32__\{\}_\{\}__\{ (0.01|0.01, 0.1, 1) \}_.*)",
        R"(.*smoke_LPT/GroupConvolutionTransformation.CompareWithRefImpl/f32_\[1,6,24,24\]_CPU_f32_4D_\[1,6,24,24\]_\[1,24,18,18\]_3_-1_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=_wo_reshape__255_\[3,8,1,1,1\]_\{ -127 \}_\{ 127 \}.*)",
        R"(.*smoke_LPT/GroupConvolutionTransformation.CompareWithRefImpl/f32_\[1,6,24(,24)*\]_CPU_f32_(3D|4D)_\[1,6,24(,24)*\]_\[1,24,18(,18)*\]_3_-1_level=256_shape=\[1,1,1.*\]_input_low=\{ 0 \}_input_high=\{ 25.5 \}_output_low=\{ 0 \}_output_high\{ 25.5 \}_precision=_wo_reshape__255_\[3,8,1,1(,1)*\]_\{ -127, -12.7, -1.27,.*)",
        R"(.*smoke_LPT/GroupConvolutionTransformation.CompareWithRefImpl/f32_\[1,6,1,24,24\]_CPU_f32_5D_\[1,6,1,24,24\]_\[1,24,1,18,18\]_3_-1_level=256_shape=\[1,1,1,1,1\]_input_low=\{ -12.8 \}_input_high=\{ 12.7 \}_output_low=\{ -12.8 \}_output_high\{ 12.7 \}_precision=_reshape_on_weights__255_\[1,1,1,1,1\]_\{ -127 \}_\{ 127 \}.*)",
        R"(.*smoke_LPT/GroupConvolutionTransformation.CompareWithRefImpl/f32_\[1,24,8,12,12\]_CPU_f32_5D_\[1,24,8,12,12\]_\[1,24,1,1,1\]_3_-1_level=256_shape=\[1,1,1,1,1\]_input_low=\{ -12.8 \}_input_high=\{ 12.7 \}_output_low=\{ -12.8 \}_output_high\{ 12.7 \}_precision=_reshape_on_weights__255_\[1,1,1,1,1\]_\{ -127 \}_\{ 127 \}.*)",
        R"(.*smoke_LPT/GroupConvolutionQDqTransformation.CompareWithRefImpl/f32_\[1,6,24,24\]_CPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ -12.8 \}_input_high=\{ 12.7 \}_output_low=\{ 0 \}_output_high=\{ 255 \}_precision=f32__u8___f32_.*_undefinedoutput_original_f32_multiplyAfter=(false|true).*)",
        R"(.*smoke_LPT/GroupConvolutionQDqTransformation.CompareWithRefImpl/f32_\[1,6,24,24\]_CPU_f32_level=256_.*_precision=f32__u8___f32_\{\}__\{ 0.1 \}.*_f32_\[6,2,5,5\]__255_\[1,1,1,1\]_\{ -128 \}_\{ 127 \}__i8.*undefinedoutput_original_u8_multiplyAfter=(false|true).*)",
        R"(.*smoke_LPT/MatMulWithConstantTransformation.CompareWithRefImpl/\[(2,3,4|1,1,3,4)\]_f32_CPU_.*_shape=\[1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0, 0, 0 \}_output_high=\{ 255, 25.5, 255 \}_precision=_level=256_shape=\[1\]_input_low=\{ -128 \}_.*)",
        R"(.*smoke_LPT/ReduceSumTransformation.CompareWithRefImpl/f32_\[1,3,10,10\]_CPU_f32_level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ 0 \}_output_high\{ 127 \}_precision=_keepDims__reduce_axis_2_3_.*)",
        R"(.*smoke_TestsDFT_3d/DFTLayerTest.Inference/.*TS=.*10.4.20.32.2.*_Precision=bf16.*)",
        R"(.*smoke_TestsDFT_3d/DFTLayerTest.Inference/.*TS=.*2.5.7.8.2.*_Precision=bf16.*)",
        R"(.*smoke_TestsDFT_3d/DFTLayerTest.Inference/.*TS=.*1.120.128.1.2.*_Precision=bf16.*_signal_size=\(\).*)",
        R"(.*smoke_TestsDFT_4d/DFTLayerTest.Inference/.*2.5.7.8.2.*Precision=bf16.*signal_size=\(\).*)",
        R"(.*smoke_TestsDFT_4d/DFTLayerTest.Inference/.*TS=\{\((10.4.20.32.2|1.120.128.1.2)\)\}.*Precision=f32.*signal_size=\(\).*)",
        R"(.*smoke_TestsDFT_4d/DFTLayerTest.Inference/.*1.120.128.1.2.*Precision=bf16.*signal_size=\(\).*)",
        R"(.*smoke_TestsDFT_2d/DFTLayerTest.Inference/.*TS.*1.120.128.1.2.*Precision=bf16_Axes=\(2.1\)_signal_size=\(\).*)",
        // by calc abs_threshold with expected value
        R"(.*smoke_.*_4D.*/GatherLayerTestCPU.CompareWithRefs.*INFERENCE_PRECISION_HINT=bf16.*)",
        R"(.*smoke.*Mvn6LayerTest.Inference/.*TS.*1.10.5.7.8.*_ModelType=f32.*_Ax=\((2.3.4|-3.-2.-1)\).*)",
        R"(.*smoke.*Mvn6LayerTest.Inference/.*TS.*2.55.*_ModelType=f32.*)",
        R"(.*smoke_ConvWithZeroPointFuse/ConvWithZeroPointFuseSubgraphTest.CompareWithRefs.*)",
        R"(.*smoke_FakeQuantize/FakeQuantizeLayerTest.Inference.*TS=.*3.49.7.5.6.*LEVELS=(255|256).*netPRC=f32.*)",
        R"(.*smoke_FakeQuantize/FakeQuantizeLayerTest.Inference.*TS=.*(2.16.4.3.18|3.10.2.5.6|3.49.5.6|2.16.3.18|2.8.5.18|3.10.5.6|2.8.1.5.18).*LEVELS=255.*netPRC=f32.*)",
        R"(.*smoke_FakeQuantize.*/FakeQuantizeLayerTest.Inference.*TS=.*3.4.2.5.*LEVELS=255.*)",
        R"(.*smoke_FakeQuantizePerChannel.*/FakeQuantizeLayerTest.Inference.*TS=.*11.10.22.19.*LEVELS=(255|256).*netPRC=f32.*)",
        R"(.*smoke_MVN_5D/Mvn6LayerTest.Inference.*TS=.*3.4.2.5.*LEVELS=255.*netPRC=f16.*)",
        R"(.*smoke_Snippets_MHAINT8MatMul/MHAINT8MatMul.*)",
        R"(.*smoke_static/ConvertFqRnnToQuantizedRnn.*2.1.5.*2.1.1.*2.1.1.*)",
        R"(.*smoke_InterpolateBicubicPillow_Layout_Test/InterpolateLayerCPUTest.CompareWithRefs/ShapeCalcMode=sizes_IS=\[?.2..20.?.?\]_TS.*1.17.4.4.*2.3.10.12.*1.17.4.4.*Sizes.*4.4.*10.20.*10.4.*PARAMETER.*0.0.0.0.*0.0.1.1.*2.3.*)",
        R"(.*smoke_LoopForCommon/LoopLayerCPUTest.CompareWithRefs/.*_netType=bf16.*)",
        R"(.*smoke_FuseScaleShiftAndFakeQuantize/FuseScaleShiftAndFakeQuantizeTest.CompareWithRefs/.*Scale=\[ 30 \]_Shift=\[ 17 \]_Intervals=\[ -1 \],\[ 5 \],\[ -5 \],\[ 1 \].*)",
        R"(.*smoke_QuantizedConvolutionBatchNorm.*/QuantizedConvolutionBatchNorm.CompareWithRefs/conv_type=convolution_quantize.*)",
        R"(.*smoke_Param/RandomUniformLayerTestCPU.CompareWithRefs/IS=\{3\}_OS=\[4,3,210\]_Min=-50_Max=0_ShapePrc=.*_OutPrc=f32_GlobalSeed=8_OperationalSeed=(0|3).*)",
        R"(.*smoke_Param/RandomUniformLayerTestCPU.CompareWithRefs/IS=\{3\}_OS=\[4,3,210\]_Min=-50_Max=50_ShapePrc=.*_OutPrc=f32_GlobalSeed=8_OperationalSeed=(5|3|0).*)",
        R"(.*smoke_Param/RandomUniformLayerTestCPU.CompareWithRefs/IS=\{3\}_OS=\[4,3,210\]_Min=-50_Max=50_ShapePrc=.*_OutPrc=f32_GlobalSeed=0_OperationalSeed=5.*)",
        R"(.*smoke_Param/RandomUniformLayerTestCPU.CompareWithRefs/IS=\{1\}_OS=\[500\]_Min=-50_Max=50_ShapePrc=.*_OutPrc=f32_GlobalSeed=0_OperationalSeed=5.*)",
        R"(.*smoke.*/RNNCellCPUTest.CompareWithRefs.*activations=.*relu.*INFERENCE_PRECISION_HINT=bf16.*)",
        R"(.*smoke_InterpolateBicubicPillow_Layout_Test/InterpolateLayerCPUTest.CompareWithRefs/ShapeCalcMode=sizes_IS=\[\?.2..20.\?.\?\]_TS=\(1.17.4.4\)_\(2.3.10.12\)_\(1.17.4.4\)_Sizes=\(4.4\)_\(10.20\)_\(10.4\)_PARAMETER.*P.*.1.1.*.*)",
        R"(.*smoke_InterpolateBicubicPillow_Layout_Test/InterpolateLayerCPUTest.CompareWithRefs/ShapeCalcMode=scales_IS=\[\?.2..20.\?.\?\]_TS=\(1.11.4.4\)_\(2.7.6.5\)_\(1.11.4.4\)_Scales=\(1.25.0.75\)_CONSTANT_.*PB=\(0.0.0.0\)_PE=\(0.0.1.1\).*)",
        R"(.*smoke_Conv_Sum_Broadcast_BF16/ConvSumInPlaceTest.CompareWithRefs.*INFERENCE_PRECISION_HINT=bf16.*)",
        R"(.*smoke_Interpolate_Basic_Down_Sample_Tail/InterpolateLayerTest.Inference.*InterpolateMode=cubic_ShapeCalcMode=scales_CoordinateTransformMode=(pytorch_half_pixel|half_pixel).*netType=f32.*)",
        R"(.*smoke_basic/PermConvPermConcat.CompareWithRefs/IS=\(1.1.8.16\)_KS=\(1.5\)_OC=.*_ET=f32_targetDevice=CPU.*)",
        R"(.*smoke_basic/PermConvPermConcat.CompareWithRefs/IS=\(1.1.7.32\)_KS=\(1.3\)_OC=.*_ET=f32_targetDevice=CPU.*)",
        R"(.*smoke_BasicNegative/RangeAddSubgraphTest.*Step=-0.1_ET=f16.*)",
        R"(.*smoke_ConvertRangeSubgraphCPUTest/ConvertRangeSubgraphCPUTest.CompareWithRefs.*bf16.*)",
        R"(.*smoke_FQLayerDQBias_4D.*FQLayerDQBias.smoke_CompareWithRefs.*_TS=\(\(1.3.64.64\)_\)_layer_type=MatMul.*)",
        R"(.*smoke_Snippets_ConvMul/ConvEltwise.CompareWithRefImpl/IS\[0\]=\(1.10.16.16\)_IS\[1\]=\(1.10.16.16\)_Op=Multiply_#N=6_#S=1.*)",
        R"(.*smoke_InterpolateBicubicPillow_LayoutAlign_Test/InterpolateLayerCPUTest.CompareWithRefs/.*Sizes=\(6.8\).*)",
        R"(.*smoke_RDFT_CPU_1D/RDFTTestCPU.CompareWithRefs/prec=f32_.*TS0=\(\((106|246|245|510|1022)\)\).*)",
        R"(.*smoke_RDFT_CPU_2D/RDFTTestCPU.CompareWithRefs/prec=f32_.*_TS0=\(\((1022.64|24.39|126.32|510.64)\)\)_constAxes=true_axes=\(\(0.1\)\)_isInverse=false_primitive=jit_avx2.*)",
        R"(.*smoke_RDFT_CPU_2D/RDFTTestCPU.CompareWithRefs/prec=f32_.*_TS0=\(\((1022.64|126.32|510.64)\)\)_constAxes=true_axes=\(\(0\)\)_isInverse=false_primitive=jit_avx2.*)",
        R"(.*smoke_RDFT_CPU_2D/RDFTTestCPU.CompareWithRefs/prec=f32_.*_isInverse=false_primitive=jit_avx512.*)",
        R"(.*smoke_RDFT_CPU_2D/RDFTTestCPU.CompareWithRefs/prec=f32_.*_TS0=\(\((20.126|20.510|20.1022)\)\)_constAxes=true_axes=\(\(1\)\)_isInverse=false_primitive=jit_avx512.*)",
        R"(.*smoke_TestsDFT_3d/DFTLayerTest.Inference/.*TS=.*1.120.128.1.2.*_Precision=f32.*signal_size=\(\).*)",
        R"(.*smoke_TestsDFT_2d/DFTLayerTest.Inference.*TS=\{\(1.120.128.1.2\)\}_Precision=f32_Axes=\(2.1\)_signal_size=\(\)_Inverse=0.*)",
        R"(.*smoke_FakeQuantizeLayerCPUTest_4D_(jit|ref)/FakeQuantizeLayerCPUTest.CompareWithRefs/IS=\[\?.\?.\?.\?\]_TS=\(\(4.16.6.7\)\).*inPrec=f32.*LEVELS=255.*)",
        R"(.*smoke_FakeQuantizeLayerCPUTest_5D_(jit|ref)/FakeQuantizeLayerCPUTest.CompareWithRefs/IS=\[\?.\?.\?.\?.\?\]_TS=\(\((4|3).16.6.7.8\)\).*inPrec=f32.*LEVELS=255.*)",
        R"(.*smoke_FakeQuantizeLayerCPUTest_Decompos/FakeQuantizeLayerCPUTest.CompareWithRefs/IS.*\(\((4.5.6.7|1.1.6.7|1.1.6.1|1.5.1.6)\)\)_inPrec=f32.*LEVELS=255.*)",
        R"(.*smoke_CompareWithRefs/LRNLayerCPUTest.CompareWithRefs/f32_IS.*axes=\(1.2.3\).*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(9.16.32.126\)\)_constAxes=true_axes=\(\((0.1.2.3|3.1|_2._1)\)\).*isInverse=false.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(1.192.36.64\)\)_constAxes=true_axes=\(\((0.1.2.3|3.2|_2._1|0.1|1)\)\).*isInverse=false.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(1.192.36.64\)\)_constAxes=true_axes=\(\((0|_2._1|0.1.2.3)\)_.*isInverse=false.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=.*_TS0=\(\(1.192.36.64\)_.*constAxes=false.*isInverse=false.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(46.10.128.65\)\)_constAxes=true_axes=\(\((1.0|0.1.2.3|3.1|_2._1)\)\).*isInverse=false.*primitive=jit_avx512.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(10.46.128.65\)\)_constAxes=true_axes=\(\((0.1|1.2)\)\).*isInverse=false.*primitive=jit_avx512.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\?.192.36.64\]_.*_axes=\(\((0|_2._1|_1|1)\)_.*isInverse=false.*)",
        R"(.*smoke_RDFT_CPU_4D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\((1.120.64.64|1.120.96.96|\?.\?.\?.\?|1.192.\?.\?|1..2.\?.\?.1..100)\)\).*isInverse=false.*)",
        R"(.*smoke_RDFT_2d/RDFTLayerTest.Inference/IS=\(100.16\)_modelType=f32_Axes=\((0.1|_2._1|1.0)\)_SignalSize=\(\).*)",
        // Issue: 134470
        R"(.*smoke.*StatefulModelStateInLoopBody.*)",
        // Issue: 123274 (Dynamic Softmax aren't supported)
        R"(smoke_Snippets_(Softmax|AddSoftmax|Reduce).*\[.*\?.*\].*)",
        R"(smoke_Snippets_BroadcastSelect_Dynamic.*)",
    };

#if defined(OPENVINO_ARCH_X86)
    retVector.emplace_back(R"(.*DetectionOutputLayerTest.*)");
    // WIP: plugin cannot be loaded for some reason
    retVector.emplace_back(R"(.*HeteroSyntheticTest.*)");
    retVector.emplace_back(R"(.*IEClassBasicTestP.*)");
#elif defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
    {
        retVector.emplace_back(
            R"(smoke_CompareWithRefs_static_check_collapsing/EltwiseLayerTest.Inference/IS.*_eltwise_op_type=Div_secondary_input_type=PARAMETER_opType=VECTOR_model_type=i32_InType=undefined_OutType=undefined_trgDev=CPU.*)");
        // Issue: 123321
        retVector.emplace_back(
            R"(.*smoke_RNNSequenceCommonZeroClip/RNNSequenceTest.Inference.*hidden_size=1.*relu.*direction=reverse.*)");
        // Ticket: 122769
        retVector.emplace_back(R"(.*smoke_nonzero/NonZeroLayerTest.Inference/IS.*)");
        retVector.emplace_back(R"(.*smoke_NormalizeL2_.*)");
        retVector.emplace_back(R"(.*Extension.XmlModelWithExtensionFromDSO.*)");
        retVector.emplace_back(R"(.*Extension.OnnxModelWithExtensionFromDSO.*)");
        retVector.emplace_back(R"(.*ONNXQuantizedModels/QuantizedModelsTests.MaxPool.*)");
        retVector.emplace_back(R"(.*ONNXQuantizedModels/QuantizedModelsTests.Convolution.*)");
        // Ticket: 134601
        retVector.emplace_back(R"(.*smoke_GroupNormalization.*)");
        // by calc abs_threshold with expected value
        retVector.emplace_back(
            R"(.*smoke_Interpolate_Basic_Down_Sample_Tail/InterpolateLayerTest.Inference.*InterpolateMode=(linear|linear_onnx)_ShapeCalcMode=scales_CoordinateTransformMode=half_pixel.*PE=\(0.0.0.0\).*netType=f32.*)");
        retVector.emplace_back(R"(.*smoke_ConversionLayerTest/ConversionLayerTest.Inference/conversionOpType=Convert_.*_inputPRC=f16_targetPRC=(u8|i8).*)");
        retVector.emplace_back(R"(.*smoke_Decomposition_4D/Mvn6LayerTest.Inference/.*TS=\{\((1.16.5.8|2.19.5.10)\)\}_ModelType=f32_.*_Ax=\(0.1.2.3\)_NormVariance=FALSE.*)");
        retVector.emplace_back(R"(.*smoke_Decomposition_4D/Mvn6LayerTest.Inference/.*TS=\{\(2.19.5.10\)\}_ModelType=f32_.*_Ax=\(1\).*)");
        retVector.emplace_back(R"(.*smoke_LogSoftmax4D/LogSoftmaxLayerTest.Inference/.*TS=\{\(2.3.4.5\)\}_modelType=f32_axis=(-4|0).*)");
        retVector.emplace_back(R"(.*smoke_Interpolate_Basic/InterpolateLayerTest.Inference/.*InterpolateMode=cubic_ShapeCalcMode=sizes_CoordinateTransformMode=tf_half_pixel.*PB=\(0.0.0.0\)_PE=\(0.0.1.1\)_.*netType=f32.*)");
        retVector.emplace_back(R"(.*smoke_CompareWithRefs_4D_Bitwise.*/EltwiseLayerCPUTest.*_eltwise_op_type=Bitwise.*_model_type=i32_.*)");
    }
    // invalid test: checks u8 precision for runtime graph, while it should be f32
    retVector.emplace_back(R"(smoke_NegativeQuantizedMatMulMultiplyFusion.*)");
    // int8 specific
    retVector.emplace_back(R"(smoke_Quantized.*)");

    // Issue: 123019
    retVector.emplace_back(R"(smoke_staticShapes4D.*INFERENCE_PRECISION_HINT=f16.*)");
    retVector.emplace_back(R"(smoke_dynamicShapes4D.*INFERENCE_PRECISION_HINT=f16.*)");
    // Issue: 124309
    retVector.emplace_back(R"(.*InferRequestPreprocessConversionTest.*oLT=NHWC.*)");
    retVector.emplace_back(R"(.*smoke_NoReshape/OVCompiledModelGraphUniqueNodeNamesTest.CheckUniqueNodeNames.*)");
    retVector.emplace_back(R"(.*smoke_BehaviorTests/InferRequestPerfCountersTest.CheckOperationInPerfMap.*)");
    retVector.emplace_back(R"(smoke_BehaviorTests/OVCompiledModelBaseTestOptional.CheckExecGraphInfo.*)");
    retVector.emplace_back(
        R"(smoke_ExecGraph/ExecGraphRuntimePrecision.CheckRuntimePrecision/Function=FakeQuantizeBinaryConvolution.*)");
    // Issue: 124395
    retVector.emplace_back(R"(smoke_VariableStateBasic/InferRequestVariableStateTest.*)");
    retVector.emplace_back(R"(smoke_VariableState/OVInferRequestVariableStateTest.*)");

#endif

#if defined(OPENVINO_ARCH_ARM)
    // TODO: rounding errors
    retVector.emplace_back(R"(.*iv_secondaryInputType=PARAMETER_opType=VECTOR_NetType=i32.*)");
    // not supported
    retVector.emplace_back(R"(.*fma.*EltwiseLayerCPUTest.*)");
    retVector.emplace_back(R"(.*int_jit.*EltwiseLayerCPUTest.*)");
    retVector.emplace_back(R"(.*dyn.*EltwiseChainTest.*)");
    retVector.emplace_back(R"(.*smoke_EltwiseChain_MergeConvert_int8/.*InPRC0=i8.*Conversion=i8.*)");
    retVector.emplace_back(R"(.*smoke_EltwiseChain_MergeConvert_int8/.*InPRC0=u8.*Conversion=i8.*)");
    retVector.emplace_back(R"(.*smoke_EltwiseChain_MergeConvert_int8/.*InPRC0=i16.*Conversion=i8.*)");
    retVector.emplace_back(R"(.*smoke_EltwiseChain_MergeConvert_int8/.*InPRC0=u16.*Conversion=i8.*)");
    retVector.emplace_back(R"(.*smoke_EltwiseChain_MergeConvert_int8/.*InPRC0=i32.*Conversion=i8.*)");
    // by calc abs_threshold with expected value
    retVector.emplace_back(R"(.*smoke_CompareWithRefs_static/EltwiseLayerTest.*_eltwise_op_type=Div_.*_model_type=i32_.*)");
#endif

#if !defined(OPENVINO_ARCH_X86_64)
    // very time-consuming test
    retVector.emplace_back(R"(.*OVInferConsistencyTest.*)");
    // TODO: generate new 'expected' runtime graph for non-x64 CPU
    retVector.emplace_back(R"(smoke_serialization/ExecGraphSerializationTest.ExecutionGraph.*)");
    retVector.emplace_back(
        R"(smoke_ExecGraph/ExecGraphRuntimePrecision.CheckRuntimePrecision/Function=(EltwiseWithTwoDynamicInputs|FakeQuantizeRelu).*)");
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
    retVector.emplace_back(R"(.*smoke_QuantizedConvolutionBatchNormTransposeOnWeights/QuantizedConvolutionBatchNorm.CompareWithRefs/conv_type=convolution_quantize_type=fake_quantize_intervals_type=per_(tensor|channel)_transpose_on_weights=true_device=CPU.*)");
    retVector.emplace_back(R"(.*smoke_LPT/ConvolutionTransformation.CompareWithRefImpl/f32_\[(1|4),3,16,16\]_CPU_f32_rank=4D_fq_on_data=\{level=256_shape=\[1,1,1,1\]_input_low=\{ 0 \}_input_high=\{ 255 \}_output_low=\{ -12.7 \}_output_high\{ 12.8 \}_precision=\}_fq_on_weights=\{_255_\[1,1,1,1\]_\{ -12.7 \}_\{ 12.7 \}\}.*)");
    retVector.emplace_back(R"(.*smoke_LPT/FuseDequantizeToFakeQuantizeTransformation.CompareWithRefImpl/CPU_f32_0_undefined_\[\]_f32__\{\}_\{\}__\{ 0.01, 0.1, 1 \}_f32_\[1,3\]_1_1_.*)");
    retVector.emplace_back(R"(.*smoke_QuantizedConvolutionBatchNorm/QuantizedConvolutionBatchNorm.CompareWithRefs/conv_type=convolution_quantize_.*)");
    retVector.emplace_back(R"(.*smoke_QuantizedConvolutionBatchNorm/QuantizedConvolutionBatchNorm.CompareWithRefs/conv_type=convolution_backprop_quantize_type=(quantize_dequantize_intervals|compressed_weights_intervals).*)");
    retVector.emplace_back(R"(.*smoke_LPT/MatMulTransformation.CompareWithRefImpl/f32_CPU_\[(1|8|1,1,1),4,12,2\]_level=256_shape=\[\]_input_low=\{ (0|-12.8) \}_input_high=\{ (25.5|12.7) \}_output_low=\{ (0|-12.8) \}_output_high\{ (25.5|12.7) \}_.*)");
    retVector.emplace_back(R"(.*smoke_LPT/MatMulTransformation.CompareWithRefImpl/f32_CPU_\[(1|8|1,1,1),4,12,2\]_level=256_shape=\[\]_input_low=\{ (0|-12.8) \}_input_high=\{ (25.5|12.7) \}_output_low=\{ (0|-12.8) \}_output_high\{ (25.5|12.7) \}_.*)");
    retVector.emplace_back(
        R"(.*smoke_MatMulCompressedWeights_corner_cases_basic/MatmulWeightsDecompression.CompareWithRefs/data_shape=\[\?.\?.\?\]_\(\[1,1,4096\]\)_weights_shape=\[4096,4096\]_group_size=128_weights_precision=nf4_decompression_precision=f16_transpose_weights=0_decompression_subtract=full_reshape_on_decompression=1_config=\(\).*)");
    retVector.emplace_back(R"(.*smoke_RDFT_CPU_1D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(126\)\)_constAxes=true_axes=\(\(0\)\)_isInverse=false.*)");
    retVector.emplace_back(R"(.*smoke_RDFT_CPU_2D/RDFTTestCPU.CompareWithRefs/prec=f32_IS0=\[\]_TS0=\(\(16.38\)\)_constAxes=true_axes=\(\(0.1\)\)_isInverse=false.*)");
#endif
    if (!ov::with_cpu_x86_avx512_core()) {
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
    if (!ov::with_cpu_x86_avx512_core_fp16()) {
        // Skip fp16 tests for paltforms that don't support fp16 precision
        retVector.emplace_back(R"(.*INFERENCE_PRECISION_HINT=(F|f)16.*)");
    }
#elif defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
    if (!ov::intel_cpu::hasHardwareSupport(ov::element::f16)) {
        // Skip fp16 tests for paltforms that don't support fp16 precision
        retVector.emplace_back(R"(.*INFERENCE_PRECISION_HINT=(F|f)16.*)");
    } else {
        // Issue 117407
        retVector.emplace_back(
            R"(.*EltwiseLayerCPUTest.*IS=\(\[1\.\.10\.2\.5\.6\]_\).*eltwiseOpType=SqDiff.*_configItem=INFERENCE_PRECISION_HINT=f16.*)");
    }
#endif
    if (!ov::with_cpu_x86_avx512_core_vnni() && !ov::with_cpu_x86_avx512_core_amx_int8()) {
        // MatMul in Snippets uses BRGEMM that supports i8 only on platforms with VNNI or AMX instructions
        retVector.emplace_back(R"(.*Snippets.*MatMulFQ.*)");
        retVector.emplace_back(R"(.*Snippets.*MatMul.*Quantized.*)");
        retVector.emplace_back(R"(.*Snippets.*MHAFQ.*)");
        retVector.emplace_back(R"(.*Snippets.*MHAINT8.*)");
        retVector.emplace_back(R"(.*Snippets.*MHAQuant.*)");
    }
    if (!ov::with_cpu_x86_avx512_core_amx_int8())
        // TODO: Issue 92895
        // on platforms which do not support AMX, we are disabling I8 input tests
        retVector.emplace_back(R"(smoke_LPT/FakeQuantizeWithNotOptimalTransformation.CompareWithRefImpl.*CPU.*i8.*)");
    if (!ov::with_cpu_x86_avx512_core_amx_bf16() && !ov::with_cpu_x86_bfloat16()) {
        // ignored for not supported bf16 platforms
        retVector.emplace_back(R"(.*smoke_Snippets_EnforcePrecision_bf16.*)");
        retVector.emplace_back(R"(.*smoke_Snippets_MHAWOTransposeEnforceBF16.*)");
        retVector.emplace_back(R"(.*smoke_Snippets_MHAEnforceBF16.*)");
    }
#ifdef SNIPPETS_LIBXSMM_TPP
    // TPP performs precision conversion implicitly, it makes all Convert tests irrelevant
    retVector.emplace_back(R"(.*smoke_Snippets_Convert.*)");
    // ABS and ROUND operations are needed for TPP support. Disable, since low precisions are not supported by TPP yet.
    retVector.emplace_back(R"(.*smoke_Snippets_FQ.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_TransposeMatMulFQ.*)");
    // TPP doesn't support op with 2 outs, when one of them is Result (ticket: 130642)
    retVector.emplace_back(R"(.*smoke_Snippets_MaxNumParamsEltwise.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_Eltwise_TwoResults.*)");
    // Accuracy problem with Exp + Reciprocal combination on TPP side (ticket: 130699)
    retVector.emplace_back(R"(.*smoke_Snippets_ExpReciprocal.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_AddSoftmax.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_TransposeSoftmaxEltwise.*)");
    // Low-precision Matmuls are not supported by TPP yet
    retVector.emplace_back(R"(.*smoke_Snippets_MatMulFQ.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_MatMulBiasQuantized.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_MatMulQuantized.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_MatMulQuantizedSoftmax.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_MHAINT8MatMul.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_MHAQuantMatMul0.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_MHAFQ.*)");
    retVector.emplace_back(R"(.*smoke_Snippets_PrecisionPropagation_Convertion.*)");
#endif

    if (ov::with_cpu_x86_avx512_core_amx()) {
        // Issue: 130463
        retVector.emplace_back(R"(smoke_Conv_1D_GEMM_BF16/ConvolutionLayerCPUTest.*K\(1\)_S\(1\)_PB\(0\)_PE\(0\).*O=6.*_Fused=Add\(PerChannel\).*)");
        // Issue: 130466
        retVector.emplace_back(R"(smoke_Conv_1D_BF16/ConvolutionLayerCPUTest.*IS=\[\].*K\(3\).*S\(2\).*PE\(0\).*D=\(1\).*O=6(3|4).*brgconv_avx512_amx.*)");
        // Issue: 130467
        retVector.emplace_back(R"(smoke_MM_Brgemm_Amx_.*/MatMulLayerCPUTest.*TS=\(\(10\.10\.10\)\).*bf16.*_primitive=brgemm_avx512_amx.*)");
        retVector.emplace_back(R"(smoke_MM_Brgemm_Amx_.*/MatMulLayerCPUTest.*IS=\[1.*TS=\(\(10\.10\.10\).*bf16.*_primitive=brgemm_avx512_amx.*)");
        retVector.emplace_back(R"(smoke_MM_Brgemm_Amx_.*/MatMulLayerCPUTest.*TS=\(\(55\.12\)\).*bf16.*_primitive=brgemm_avx512_amx.*)");
        // Issue: 130471
        retVector.emplace_back(R"(smoke_JIT_AVX512_DW_GroupConv/GroupConvolutionLayerCPUTest.*inFmts=nCdhw16c.*INFERENCE_PRECISION_HINT=bf16.*)");
        // Issue: 131475
        retVector.emplace_back(R"(smoke_ExportImportTest/ExportOptimalNumStreams.OptimalNumStreams/.*)");
        // by calc abs_threshold with expected value
        retVector.emplace_back(R"(.*smoke_GatherCompressedWeights_basic/GatherWeightsDecompression.CompareWithRefs.*INFERENCE_PRECISION_HINT.*bf16.*)");
        retVector.emplace_back(R"(.*smoke_Interaction/IntertactionCPUTest.CompareWithRefs.*Prc=i32.*)");
        retVector.emplace_back(R"(.*smoke_MatMulCompressedWeights_(amx|corner_cases_amx)/MatmulWeightsDecompression.CompareWithRefs.*INFERENCE_PRECISION_HINT.*bf16.*)");
        retVector.emplace_back(R"(.*smoke_AutoBatching_CPU/AutoBatching_Test.*)");
        retVector.emplace_back(R"(.*smoke_Snippets_EnforcePrecision_bf16/EnforcePrecisionTest.*)");
        retVector.emplace_back(R"(.*smoke_Snippets_MHABF16_4D/MHA.CompareWithRefImpl/.*\[1.58.16.34\]_IS\[1\]=\[1.58.16.34\]_IS\[2\]=\[1.1.1.58\]_IS\[3\]=\[1.58.16.34\].*)");
        retVector.emplace_back(R"(.*smoke_Snippets_MHAWOTransposeBF16_(3|4)D/MHAWOTranspose.*)");
    }

    if (ov::with_cpu_x86_avx512_core_fp16()) {
        // Issue: 130473
        retVector.emplace_back(R"(smoke_CompareWithRefs_4D.*/EltwiseLayerCPUTest.*Sub_secondary.*INFERENCE_PRECISION_HINT=f16.*FakeQuantize.*enforceSnippets=1.*)");
        retVector.emplace_back(R"(smoke_Reduce.*/ReduceCPULayerTest.*axes=\((0.1|1)\).*Prod_KeepDims.*INFERENCE_PRECISION_HINT=f16.*)");
        retVector.emplace_back(R"(smoke_ConvertRangeSubgraphCPUTest/ConvertRangeSubgraphCPUTest\.CompareWithRefs.*Prc=f16.*)");
    }

    return retVector;
}
