// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/low_precision.hpp"

namespace LowPrecisionTestDefinitions {

    TEST_P(LowPrecisionTest, CompareWithRefs) {
        Run();
    }

}  // namespace LowPrecisionTestDefinitions
