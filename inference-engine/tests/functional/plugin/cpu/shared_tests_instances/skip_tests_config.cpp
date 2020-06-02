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
        R"(.*(FakeQuantize).*)",
        // TODO: Issue: 32521
        R"(.*(EltwiseLayerTest).*secondaryInputType=CONSTANT.*netPRC=FP16.*)",
        R"(.*(EltwiseLayerTest).*IS=.*1.1.1.1.*opType=SCALAR.*)"
    };
}