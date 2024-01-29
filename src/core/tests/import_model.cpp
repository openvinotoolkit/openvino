// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/runtime/core.hpp"

// Test is disabled due to issue 128924
TEST(ImportModel, DISABLED_ImportModelWithNullContextThrows) {
    ov::Core core;
    ov::RemoteContext context;
    std::istringstream stream("None");
    ASSERT_THROW(core.import_model(stream, context, {}), ov::Exception);
}