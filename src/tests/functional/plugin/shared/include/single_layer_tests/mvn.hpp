// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/mvn.hpp"

namespace LayerTestsDefinitions {

TEST_P(Mvn1LayerTest, CompareWithRefs) {
    Run();
};

TEST_P(Mvn6LayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions