// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/generate_proposals.hpp"

namespace ov {
namespace test {
TEST_P(GenerateProposalsLayerTest, GenerateProposalsLayerTests) {
    run();
}
} // namespace test
} // namespace ov
