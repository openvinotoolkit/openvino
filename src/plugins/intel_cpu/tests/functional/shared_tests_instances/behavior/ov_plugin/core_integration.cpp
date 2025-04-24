// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/properties.hpp>
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

namespace {

// OV Class Query model

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("CPU"));


}  // namespace
