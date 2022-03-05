// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // Not supported activation types
        ".*ActivationLayerTest\\.CompareWithRefs/Tanh.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Exp.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Log.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Sigmoid.*netPRC=FP32.*",
        ".*ActivationLayerTest\\.CompareWithRefs/Relu.*netPRC=FP32.*",
        // Not supported dynamic shapes without upper bound
        ".*InferDynamicNetworkWithGetTensor2times.function.*",
        ".*InferFullyDynamicNetworkWithGetTensor.function.*",
        ".*InferDynamicNetworkWithGetTensor.function.*",
        ".*InferDynamicNetworkWithoutSetShape.function.*",
        ".*InferFullyDynamicNetworkWithSetTensor/function.*",
        ".*InferDynamicNetworkWithSetTensor2times.*",
        ".*InferRequestDynamicTests.GetSameTensor2times.*",
        ".*InferRequestDynamicTests.InferDynamicNetworkWithSetTensor.*",
        // TODO: Issue: 26268
        ".*ConcatLayerTest.*axis=0.*",
        // TODO: Issue 31197
        R"(.*(IEClassBasicTestP).*smoke_registerPluginsXMLUnicodePath.*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue: 40473
        R"(.*TopKLayerTest.*mode=min.*sort=index.*)",
        // TODO: Issue: 42828
        R"(.*DSR_NonMaxSuppression.*NBoxes=(5|20|200).*)",
        // TODO: Issue: 42721
        R"(.*(DSR_GatherND).*)",
        // TODO: Issue 26090
        ".*DSR_GatherStaticDataDynamicIdx.*f32.*1.3.200.304.*",
        // TODO: Issue 47315
        ".*ProposalLayerTest.*",
        // TODO: Issue 51804
        ".*InferRequestPreprocessConversionTest.*oPRC=U8.*",
        // TODO: Issue 54163
        R"(.*ActivationLayerTest.*SoftPlus.*)",
        // TODO: Issue 54722
        R"(.*TS=\(\(16\.16\.96\)_\(96\)_\).*eltwiseOpType=FloorMod_secondaryInputType=PARAMETER_opType=VECTOR_netPRC=FP32.*)",
        // TODO: Issue 57108
        R"(.*QueryNetworkHETEROWithMULTINoThrow_V10.*)",
        R"(.*QueryNetworkMULTIWithHETERONoThrow_V10.*)",
        // Not implemented yet:
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNet.*)",
        R"(.*Behavior.*ExecutableNetworkBaseTest.*canExport.*)",
        // TODO: Issue 69529
        R"(.*Behavior.*OVExecGraphImportExportTest.*ieImport.*)",
        R"(.*Behavior.*OVExecGraphImportExportTest.*ExportedIENetwork.*)",
        // TODO: Issue 73501
        R"(.*_Hetero_Behavior.*OVExecGraphImportExportTest.*)",
        // TODO: Issue 65013
        R"(.*importExportedFunctionConstantResultOnly.*elementType=(f32|f16).*)",
        // Not expected behavior
        R"(.*Behavior.*ExecNetSetPrecision.*canSetOutputPrecisionForNetwork.*U8.*)",
        R"(.*CoreThreadingTestsWithIterations.*)",
        R"(.*OVExecutableNetworkBaseTest.*CanSetConfigToExecNet.*)",
        R"(.*OVClassNetworkTestP.*(SetAffinityWithConstantBranches|SetAffinityWithKSO).*)",
        // TODO: Issue 69640
        R"(.*EltwiseLayerTest.*OpType=Prod.*)",
        R"(.*EltwiseLayerTest.*OpType=SqDiff.*PARAMETER.*SCALAR.*)",
        R"(.*EltwiseLayerTest.*TS=\(\(16\.16\.96\)_\(96\)_\).*OpType=SqDiff.*)",
        R"(.*EltwiseLayerTest.*TS=\(\(52\.1\.52\.3\.2\)_\(2\)_\).*OpType=SqDiff.*)",

        // Tests with unsupported precision
        ".*InferRequestCheckTensorPrecision.*type=boolean.*",
        ".*InferRequestCheckTensorPrecision.*type=bf16.*",
        ".*InferRequestCheckTensorPrecision.*type=f64.*",
        ".*InferRequestCheckTensorPrecision.*type=i4.*",
        ".*InferRequestCheckTensorPrecision.*type=i16.*",
        ".*InferRequestCheckTensorPrecision.*type=i64.*",
        ".*InferRequestCheckTensorPrecision.*type=u1.*",
        ".*InferRequestCheckTensorPrecision.*type=u4.*",
        ".*InferRequestCheckTensorPrecision.*type=u8.*",
        ".*InferRequestCheckTensorPrecision.*type=u16.*",
        ".*InferRequestCheckTensorPrecision.*type=u64.*",

        // TODO: Issue 76209
        R"(.*MultithreadingTests.*canRun.*RequestsConsistentlyFromThreads.*MYRIAD.*)",
        // TODO: CVS-82012
        R"(.*StridedSliceLayerTest\.CompareWithRefs/inShape=\(1\.12\.100\).*)",

        // Issue: 81016
        R"(.*ParameterResultSubgraphTest\.CompareWithRefs.*)",
    };
}
