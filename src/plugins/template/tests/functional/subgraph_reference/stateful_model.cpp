// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/stateful_model.hpp"

using namespace ov::test;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke, StatefulModelStateInLoopBody, ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

}  // namespace
