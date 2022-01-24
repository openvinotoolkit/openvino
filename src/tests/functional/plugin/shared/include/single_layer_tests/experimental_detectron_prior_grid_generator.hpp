// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/single_layer/experimental_detectron_prior_grid_generator.hpp>

namespace ov {
namespace test {
namespace subgraph {

TEST_P(ExperimentalDetectronPriorGridGeneratorLayerTest, ExperimentalDetectronPriorGridGeneratorLayerTests) {
    run();
}

} // namespace subgraph
} // namespace test
} // namespace ov
