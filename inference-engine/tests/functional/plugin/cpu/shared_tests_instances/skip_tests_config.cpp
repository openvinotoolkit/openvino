// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        // TODO: Issue 26264
        R"(.*(MaxPool|AvgPool).*S\(1\.2\).*Rounding=CEIL.*)",
        // TODO: Issue 31841
        R"(.*(QuantGroupConvBackpropData3D).*)",
        // TODO: Issue 31843
        R"(.*(QuantConvBackpropData3D).*)",
        R"(.*(QuantConvBackpropData2D).*(QG=Perchannel).*)",
        R"(.*(QuantGroupConvBackpropData2D).*(QG=Perchannel).*)",
        // TODO: Issue 33886
        R"(.*(QuantGroupConv2D).*)",
        R"(.*(QuantGroupConv3D).*)",
        // TODO: Issue 31845
        R"(.*(FakeQuantizeLayerTest).*)",
        R"(.*(EltwiseLayerTest).*IS=\(.*\..*\..*\..*\..*\).*secondaryInputType=PARAMETER.*opType=SCALAR.*)",
        // TODO: Issue 32756
        R"(.*Transpose.*inputOrder=\(\).*)",
        // TODO: failed to downgrade to opset v0 in interpreter backend
        R"(.*Gather.*axis=-1.*)",
        // TODO: Issue 33151
        R"(.*Reduce.*type=Logical.*)",
        R"(.*Reduce.*axes=\(1\.-1\).*)",
        R"(.*Reduce.*axes=\(0\.3\)_type=Prod.*)",
        // TODO: Issue: 34518
        R"(.*RangeLayerTest.*)",
        R"(.*(RangeAddSubgraphTest).*Start=1.2.*Stop=(5.2|-5.2).*Step=(0.1|-0.1).*netPRC=FP16.*)",
        // TODO: Issue: 34083
#if (defined(_WIN32) || defined(_WIN64))
        R"(.*(CoreThreadingTestsWithIterations).*(smoke_LoadNetworkAccuracy).*)",
#endif
        // TODO: Issue: 34348
        R"(.*IEClassGetAvailableDevices.*)",
        // TODO: Issue: 25533
        R"(.*ConvertLikeLayerTest.*)",
        // TODO: Issue: 34516
        R"(.*ConvertLayerTest.*)",
        // TODO: Issue: 34055
        R"(.*ShapeOfLayerTest.*)",
        R"(.*ReluShapeOfSubgraphTest.*)",
        // TODO: Issue: 34805
        R"(.*ActivationLayerTest.*Ceiling.*)",
        // TODO: Issue: 32032
        R"(.*ActivationParamLayerTest.*)",
        // TODO: Issue: 32959
        R"(.*ActivationLayerTest.*Mish.*)",
        // TODO: Issue: 30999 (Implement Interpolate reference in NGraph)
        R"(.*InterpolateLayerTest.*)"
    };
}