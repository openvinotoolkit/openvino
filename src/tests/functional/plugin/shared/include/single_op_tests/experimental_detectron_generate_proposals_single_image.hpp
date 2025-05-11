// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/experimental_detectron_generate_proposals_single_image.hpp"

namespace ov {
namespace test {
TEST_P(ExperimentalDetectronGenerateProposalsSingleImageLayerTest, Inference) {
    run();
}
} // namespace test
} // namespace ov
