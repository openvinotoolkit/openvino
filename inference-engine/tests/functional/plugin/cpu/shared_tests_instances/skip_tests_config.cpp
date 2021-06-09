// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include <ie_system_conf.h>
#include "functional_test_utils/skip_tests_config.hpp"

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
        R"(.*PreprocessDynamicallyInSetBlobTest.*iPRC=0.*_iLT=1.*)",
        R"(.*PreprocessDynamicallyInSetBlobTest.*oPRC=0.*_oLT=1.*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue: 25533
        R"(.*ConvertLikeLayerTest.*)",
        // TODO: Issue: 34055
        R"(.*ShapeOfLayerTest.*)",
        R"(.*ReluShapeOfSubgraphTest.*)",
        // TODO: Issue: 34805
        R"(.*ActivationLayerTest.*Ceiling.*)",
        // TODO: Issue: 43314
        R"(.*Broadcast.*mode=BIDIRECTIONAL.*inNPrec=BOOL.*)",
        // TODO: Issue 43417 sporadic issue, looks like an issue in test, reproducible only on Windows platform
        R"(.*decomposition1_batch=5_hidden_size=10_input_size=30_.*tanh.relu.*_clip=0_linear_before_reset=1.*_targetDevice=CPU_.*)",
        // Skip platforms that do not support BF16 (i.e. sse, avx, avx2)
        R"(.*BF16.*(jit_avx(?!5)|jit_sse|ref).*)",
        // TODO: Incorrect blob sizes for node BinaryConvolution_X
        R"(.*BinaryConvolutionLayerTest.*)",
        R"(.*ClampLayerTest.*netPrc=(I64|I32).*)",
        R"(.*ClampLayerTest.*netPrc=U64.*)",
        // TODO: 53618. BF16 gemm ncsp convolution crash
        R"(.*_GroupConv.*_inPRC=BF16.*_inFmts=nc.*_primitive=jit_gemm.*)",
        // TODO: 53578. fork DW bf16 convolution does not support 3d cases yet
        R"(.*_DW_GroupConv.*_inPRC=BF16.*_inFmts=(ndhwc|nCdhw16c).*)",
        // TODO: 56143. Enable nspc convolutions for bf16 precision
        R"(.*ConvolutionLayerCPUTest.*BF16.*_inFmts=(ndhwc|nhwc).*)",
        // TODO: 56827. Sporadic test failures
        R"(.*smoke_Conv.+_FP32.ConvolutionLayerCPUTest\.CompareWithRefs.IS=\(1\.67.+\).*inFmts=n.+c.*_primitive=jit_avx2.*)",

        // incorrect reference implementation
        R"(.*NormalizeL2LayerTest.*axes=\(\).*)",
        // lpt transformation produce the same names for MatMul and Multiply
        R"(.*MatMulTransformation.*)",
        // incorrect jit_uni_planar_convolution with dilation = {1, 2, 1} and output channel 1
        R"(.*smoke_Convolution3D.*D=\(1.2.1\)_O=1.*)",

        // Unsupported operation of type: NormalizeL2 name : Doesn't support reduction axes: (2.2)
        R"(.*BF16NetworkRestore1.*)",
        R"(.*MobileNet_ssd_with_branching.*)",

        // TODO: 55656 AUTO plugin and QueryNetwork
        R"(.*CoreThreading.*smoke_QueryNetwork.*targetDevice=AUTO_config.*)",
        // Unsupported config KEY_ENFORCE_BF16 for AUTO plugin
        R"(.*smoke_SetBlobOfKindAUTO.*SetBlobOfKindTest.CompareWithRefs.*)",
        // reference doesn't cover I8, U8 cases. Issue: 55842
        R"(.*Gather7LayerTest.*netPRC=I8.*)",
        // need to implement Export / Import
        R"(.*IEClassImportExportTestP.*)"
    };
#ifdef __APPLE__
        // TODO: Issue 55717
        //retVector.emplace_back(R"(.*smoke_LPT.*ReduceMinTransformation.*f32.*)");
#endif
    if (!InferenceEngine::with_cpu_x86_avx512_core()) {
        // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
        // tests are useless on such platforms
       retVector.emplace_back(R"(.*BF16.*)");
       retVector.emplace_back(R"(.*bfloat16.*)");
    }

    return retVector;
}
