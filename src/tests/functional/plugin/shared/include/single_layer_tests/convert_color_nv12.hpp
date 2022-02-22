// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/convert_color_nv12.hpp"

namespace LayerTestsDefinitions {

TEST_P(ConvertColorNV12LayerTest, CompareWithRefs) {
    Run();
}

TEST_P(ConvertColorNV12AccuracyTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions