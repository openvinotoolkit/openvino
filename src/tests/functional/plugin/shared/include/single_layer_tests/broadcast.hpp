// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/broadcast.hpp"

namespace LayerTestsDefinitions {

TEST_P(BroadcastLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

namespace ov {
namespace test {

TEST_P(BroadcastLayerTestNew, CompareWithRefs) {
    run();
}

} //  namespace test
} //  namespace ov
