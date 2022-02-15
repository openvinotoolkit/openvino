// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/single_layer/generate_proposals_single_image.hpp>

namespace ov {
namespace test {
namespace subgraph {

TEST_P(GenerateProposalsSingleImageLayerTest, GenerateProposalsSingleImageLayerTests) {
    run();
}

} // namespace subgraph
} // namespace test
} // namespace ov
