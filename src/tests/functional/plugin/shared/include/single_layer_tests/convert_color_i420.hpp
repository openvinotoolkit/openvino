// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/convert_color_i420.hpp"

namespace LayerTestsDefinitions {

TEST_P(ConvertColorI420LayerTest, CompareWithRefs) {
    Run();
}

TEST_P(ConvertColorI420AccuracyTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions