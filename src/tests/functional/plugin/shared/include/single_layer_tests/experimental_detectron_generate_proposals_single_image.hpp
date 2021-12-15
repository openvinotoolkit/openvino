// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/single_layer/experimental_detectron_generate_proposals_single_image.hpp>

namespace ov {
namespace test {
namespace subgraph {

TEST_P(ExperimentalDetectronGenerateProposalsSingleImageLayerTest, ExperimentalDetectronGenerateProposalsSingleImageLayerTests) {
    run();
}

} // namespace subgraph
} // namespace test
} // namespace ov
