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
        // TODO: Issue 31839
        R"(.*(QuantConvBackpropData3D).*)",
        // TODO: Issue 31841
        R"(.*(QuantGroupConvBackpropData3D).*)",
        // TODO: Issue 31843
        R"(.*(QuantGroupConvBackpropData2D)*QG=Perchannel.*)",
        // TODO: Issue 32023
        R"(.*(QuantGroupConvBackpropData2D)*QG=Pertensor.*)",
        // TODO: Issue 31845
        R"(.*(FakeQuantize/FakeQuantizeLayerTest).*)",
        R"(.*(EltwiseLayerTest).*IS=\(.*\..*\..*\..*\..*\).*secondaryInputType=PARAMETER.*opType=SCALAR.*)",
        // TODO: Issue 32756
        R"(.*Transpose.*inputOrder=\(\).*)",
        // TODO: failed to downgrade to opset v0 in interpreter backend
        R"(.*Gather.*axis=-1.*)",
        // TODO: Issue 33151
        R"(.*Reduce.*type=Logical.*)",
        R"(.*Reduce.*axes=\(1\.-1\).*)",
        R"(.*Reduce.*axes=\(0\.3\)_type=Prod.*)",
    };
}
