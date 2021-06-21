// Copyright (C) 2018-2021 Intel Corporation
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
        // TODO: Issue: 26268
        ".*ConcatLayerTest.*axis=0.*",
        // TODO: Issue 31197
        R"(.*(IEClassBasicTestP).*smoke_registerPluginsXMLUnicodePath.*)",
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue: 40473
        R"(.*TopKLayerTest.*mode=min.*sort=index.*)",
        // TODO: Issue: 40961
        R"(.*(ConstantResultSubgraphTest).*)",
        // TODO: Issue: 42828
        R"(.*DSR_NonMaxSuppression.*NBoxes=(5|20|200).*)",
        // TODO: Issue: 42721
        R"(.*(DSR_GatherND).*)",
        // TODO: Issue 26090
        ".*DSR_GatherStaticDataDynamicIdx.*f32.*1.3.200.304.*",
        // TODO: Issue 47315
        ".*ProposalLayerTest.*",
        // TODO: Issue 51804
        ".*PreprocessConversionTest.*oPRC=U8.*",
        // TODO: Issue: 56556
        R"(.*(PreprocessTest).*(SetScalePreProcessSetBlob).*)",
        R"(.*(PreprocessTest).*(SetScalePreProcessGetBlob).*)",
        // TODO: Issue 54163
        R"(.*ActivationLayerTest.*SoftPlus.*)",
        // TODO: Issue 54722
        R"(.*IS=\(16\.16\.96\)\(96\)_eltwiseOpType=FloorMod_secondaryInputType=PARAMETER_opType=VECTOR_netPRC=FP32.*)",
        // TODO: Issue CVS-57108
        R"(.*QueryNetworkHETEROWithMULTINoThrow_V10.*)",
        R"(.*QueryNetworkMULTIWithHETERONoThrow_V10.*)"
    };
}
